# P 端 A/B：`TP4×PP2` 与 `TP8+LayerSplit` 到底差在哪

> 前提：两者都是 PD 的 **prefill 侧**、都是 8 卡、都假设能跑起来。本页不比可行性，只比差异。
> 适用版本 SGLang **v0.5.23**；所有时间与容量数字由公式推导，**未做运行时实测**。

## TL;DR

1. **聚合算力完全一样**：两者都是 8 卡全用上，单 chunk 的理想 prefill 时间相同。
   PP2 只是把「8 卡切一层」换成「4 卡切两段」，不是加速。
2. **省下的 KV 差 3.4 倍**：PP2 每卡 `⌈L/2⌉` 层，LayerSplit(S=8) 每卡 `⌈L/8⌉+1` 层
   （L=61 时 31 对 9）。
3. **代价可同屏比较**：PP2 浪费 `1/(m+1)` 在 bubble 上；
   LayerSplit 浪费 `(L−⌈L/S⌉)×池容量×b / 带宽` 在广播上。
4. **分水岭是池容量**：按 55 GB/s、L=61、m=4 算，交叉点约 **88k token**。
   池子比它小 → LayerSplit 更便宜；比它大 → PP2 更便宜。

## 术语约定

- **PP2** = `--tp 4 --pp 2`（每 stage 4 卡，两个 stage 覆盖全层）。
- **LayerSplit** = `--tp 8 --attn-cp-size 8 --enable-dsa-cache-layer-split`（PP=1）。
- 两者都带 CP：PP2 那侧 `attn_cp_size=4`，LayerSplit 那侧 `8`。
  因为 `attn_tp = tp // dp // cp`（`runtime_context.py:73-76`），**两者的 attention TP 都等于 1**。
- **P 端** = PD 的 prefill worker。

## 01 两者切的不是同一个东西

| 维度 | TP4 × PP2（CP4） | TP8（CP8）+ LayerSplit |
|---|---|---|
| 切什么 | 计算图：层段 → stage | KV/indexer cache 的归属：层 → owner rank |
| 权重 | 每 stage 持 `L/2` 层，层内按 TP4 切 | 每卡持全部 `L` 层，层内按 TP8 切 |
| 每卡 KV 层数 | `⌈L/2⌉` | `⌈L/8⌉ + 1` |
| attention 并行 | CP4，attn_tp = 4/4 = **1** | CP8，attn_tp = 8/8 = **1** |
| FFN / MoE 并行 | TP4（stage 内） | TP8 |
| 每前向主要通信 | stage 边界 P2P（hidden states）+ TP4 all-reduce | TP8 all-reduce + CP8 all-gather + 每远程层**整层**广播 |
| 时间损失来源 | 流水线 bubble | 广播（预取深度 1） |
| 与 chunk 大小的关系 | 强：micro-batch 数 `m` 决定 bubble | 弱：广播量看池容量，不看 chunk |
| 外部依赖 | 无 | mooncake + 全 CP rank 参与传输 |

## 02 8 张卡上的分布

```text
A · TP4 × PP2
  stage0: [r0 r1 r2 r3]  layers [0, ⌈L/2⌉)   TP4 · CP4
  stage1: [r4 r5 r6 r7]  layers [⌈L/2⌉, L)   TP4 · CP4

B · TP8 + CP8 + LayerSplit
  [r0 r1 r2 r3 r4 r5 r6 r7]  全部 L 层都参与计算（TP8 + CP8）
                              每卡只「存」⌈L/8⌉+1 层 KV
```

反直觉的相同点：**attention 在这一层都不是 TP 的**（`attn_tp = 1`），
所以 attention 侧的并行度全靠 CP —— 4 对 8，LayerSplit 那侧是 PP2 的两倍。
差别在 FFN：TP8 对 TP4×2 段。

L=61 时每卡 KV 层数是 **31 对 9**。但 P 端的 KV 是**过路**的（算完经 mooncake 传走，不承载并发），
所以这笔节省换来的不是「能多跑请求」，而是「同一个池子能装更长的上下文 / 更大的 radix 前缀」。

## 03 时间去哪了：bubble 与广播

```text
PP2 :  bubble_ratio = (pp − 1) / (m + pp − 1) = 1 / (m + 1)
         浪费时间    = bubble_ratio × t_chunk

LS  :  remote_layers = L − ⌈L/S⌉
         浪费字节      = remote_layers × (size + page_size) × b
         浪费时间      = 浪费字节 / 带宽
```

两个式子的形状完全不同：

| | 随什么变 | 能不能消掉 |
|---|---|---|
| PP2 的 bubble | 只随 `m` 变，**与池子大小无关** | 加 `m` 就能压：m=8 → 11%，m=16 → 5.9% |
| LS 的广播 | 随**池容量**线性变，与 chunk 大小无关 | 只能靠「提前一层」预取盖住；盖不住没有别的办法 |

> **为什么 LS 的广播非得是整层？** `_broadcast_tensor_from_owner()` 里
> `comm.broadcast(tensor, src=owner_rank)` 广播的是 `remote_kv_buffer` 完整张量
> （`size + page_size` 行），不是按本批次命中的 page 做 gather。
> 集合通信要求所有 rank 传形状一致的张量，这是实现上的取舍。
> 所以「池子越大、广播越贵」是结构性的，调参绕不开。

## 04 交叉点

设 `bubble = 1/(m+1)`，`remote = L − ⌈L/S⌉`：

```text
pool* = bubble × t_chunk × BW / (remote × b)

默认：bubble=0.2, t_chunk=300ms, BW=55 GB/s, L=61, S=8, b=708B
  remote = 61 − 8 = 53
  pool*  = 0.2 × 0.3 × 55e9 / (53 × 708) ≈ 88k token
```

- 池子 **< 88k** → LS 的广播（< 60 ms）比 PP2 的 bubble（60 ms）便宜，且白拿 3.44× KV 余量。
- 池子 **> 88k** → LS 的广播压过 bubble；此时要么把 `m` 加上去，要么选 PP2。

`m` 一加，交叉点同比右移（m=8 → ~480k）。**LS 那侧的浪费对 `m` 完全不敏感** —— 这是两者的本质差别。

## 05 四个真正会分化的面

### 5.1 单请求 TTFT：两条路径不同

| | TP4 × PP2 | TP8 + LayerSplit |
|---|---|---|
| 单个短请求 | **更差**：要过完两个 stage + 一次 P2P；流水填不满就是纯损失 | **更好**：单 stage 无 P2P；CP8 只切 attention，短上下文收益也低 |
| 单个长请求（多 chunk） | **更好**：不同 chunk 可同时落在不同 stage，等于 2 路流水 | **看情况**：chunk 串行，只有 attention 被 CP8 加速，FFN 不变 |
| attention 占比 | 被 CP4 加速 | 被 CP8 加速（这块是强项） |

短 prompt 为主 → LayerSplit；长 prompt 且能填满流水 → PP2。这条与容量无关，是调度形态的差异。

### 5.2 吞吐：同算力，不同损失项

稳态吞吐 = `1 / (t_ideal + 浪费)`。`t_ideal` 相同，比较退化成比较 03 节的两个式子。

### 5.3 显存：3.4 倍，但 P 端可能用不上

`⌈L/8⌉+1` 对 `⌈L/2⌉`。这笔显存只有在「池子被撑到很大」时才有意义。
反过来：**如果池子撑不大，LayerSplit 的显存优势是白拿的、而广播开销是实付的** —— 这正是交叉点的来源。

### 5.4 运维面

| | TP4 × PP2 | TP8 + LayerSplit |
|---|---|---|
| 外部依赖 | 无 | mooncake（传输后端被硬绑） |
| 正确性风险 | 层段对齐、micro-batch 调度 | owner 广播的可见性（典型症状是「偶发、与 batch 组成相关」的精度下降，不是崩溃） |
| PD 传输 | 每 stage 各发自己那段层 | 全 CP rank 各发自己那段；CP 复制状态不能再只由 rank 0 发 |
| 调参旋钮 | `--pp-max-micro-batch-size`、`--chunked-prefill-size`、`--enable-dynamic-chunking`、`SGLANG_PP_LAYER_PARTITION` | 基本没有：`S` 由拓扑定死，广播量由池容量定死 |

最后一行很实在：**PP2 有旋钮可调，LayerSplit 没有。**
PP 的 bubble 能靠加 micro-batch、动态 chunk、层段对齐压到接近 0；
LayerSplit 的广播量只由「池子多大」和「带宽多宽」决定。

## 06 结论

| 你的处境 | 选 | 为什么 |
|---|---|---|
| 池子 ≤ 交叉点（约 88k token） | **TP8 + LayerSplit** | 广播比 bubble 便宜，且白拿 3–4 倍 KV 余量；短 prompt 上 TTFT 也更好 |
| 池子 ≫ 交叉点（长上下文、大前缀） | **TP4 × PP2** | 广播量 ∝ 池容量，会压过 bubble；PP 还能靠加 `m` 继续压 |
| 短 prompt 为主、延迟敏感 | **TP8 + LayerSplit** | 单 stage 无 P2P、无流水填充 |
| 想把调参空间留给自己 | **TP4 × PP2** | 四个旋钮都能动；LayerSplit 的代价结构是固定的 |

方法论：两个配置在**理想时间**上完全等价（都是 8 卡全用），
所以任何「哪个更快」的结论都必须落到「哪个的浪费更小」——
而浪费的两个来源一个是 `m`、一个是 `池容量/带宽`。把这两个数测出来，选择就唯一了。

## 07 自测

1. **既然都是 8 卡全用上，为什么还说 PP2「不是加速」？**
   → 对单个 chunk 而言，PP2 要把它送过两个 stage 才出结果，每个 stage 只用 4 卡，
   时间与 TP8 的全 8 卡单 stage 大致相同。PP 的加速来自**不同 chunk 同时在飞**。
   所以只有「chunk 数 ≥ pp」时流水才填满，短请求（1 个 chunk）反而是纯损失。
2. **LS 的广播为什么不能只传本批次命中的 page？**
   → 集合通信要求所有 rank 传形状一致的张量，而 `remote_kv_buffer` 形状固定为
   `(size + page_size, 1, kv_dim)`。按 page 做 gather 需要每次协商索引，要多一轮通信。
   代价就是你调大池子，每层广播同比变贵。
3. **交叉点为什么随 `m` 移动？**
   → PP2 的浪费是 `1/(m+1)`，`m` 一加就掉（m=4 → 20%，m=8 → 11%，m=16 → 5.9%），
   交叉点同比右移（~88k → ~480k）。LS 那侧的浪费对 `m` 完全不敏感。
4. **只关心「显存够不够跑更长上下文」时还要看时间吗？**
   → 要，但结论简单得多：LayerSplit 每卡 `⌈L/8⌉+1` 层，PP2 是 `⌈L/2⌉` 层，差 3–4 倍。
   如果目标是「同池装更长序列」而非「更多并发」，LayerSplit 的显存优势是实打实的；
   此时再用 04 节核对广播是否还盖得住。

## 来源

- 源码（v0.5.23）：`mem_cache/dsa_cache_layer_split.py`（`_broadcast_tensor_from_owner` 的整层语义、
  `prefetch_kv_buffer` 的深度 1 预取）、`layers/cp/utils.py`（层归属）、
  `runtime_context.py:73-76`（`attn_tp = tp // dp // cp`）、`disaggregation/prefill.py`（每 rank 只发自己那段层）
- 文档：`docs/docs/advanced_features/pipeline_parallelism.mdx`（bubble 与 dynamic chunking 的口径）

> 本页刻意不谈两者的可跑性（`--pp-size 1` 与 PD prefill 两道门）与 6000D 的后端白名单 ——
> 那些在「单机 8 卡：PP2 还是 LayerSplit」那一篇里。
