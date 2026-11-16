# 单机 8 卡：PP2 还是 LayerSplit —— 以 6000D 为例的选型

> 适用版本：SGLang **v0.5.23**。本页由读源码 + 公式推导产出，**未做运行时实测**；
> 6000D 的互连带宽、模型的层数 L 与每 token KV 字节都不在本页做断言，它们是计算器的输入。

## TL;DR

1. **`TP8 + LayerSplit` 需要 16 卡，8 卡上不是一个可部署配置。**
   LayerSplit 只存在于 PD 的 prefill worker（`model_hook.py:365-380`），
   而 PD 两端的有效 TP 宽度必须一致（prefill 的 `attn_tp_size × attn_cp_size` == decode 的 `attn_tp_size`，
   `conn.py:1376-1388`）—— TP8 prefill 就要 8 卡 decode。
2. **PP 和 LayerSplit 在配置层面互斥**：`pp_size > 1` 时 LayerSplit 直接 raise（`model_hook.py:402-408`）。
3. **PP2 把每卡 KV 砍半（容量 ×1.97），PD 4+4 不会。**
4. **6000D 上第一个会炸的是 DSA 后端**：SM120/SM121 + GLM DSA + FP8 KV 只允许
   `flashinfer_sparse_mla`（默认）或 `triton_sparse_mla`（prefill），`--dsa-prefill-backend trtllm` 被拒绝。

## 术语约定

- **LayerSplit** = `--enable-dsa-cache-layer-split`；**PP** = `--pp-size`；
  **CP** = `--attn-cp-size`（attention context parallel，嵌在 TP 世界内）；**PD** = prefill/decode 分离。
- **6000D** 按 **SM120（sm_120）** 处理 —— 源码里 SM120/SM121 是 GLM DSA + FP8 KV 的一等分支
  （`kernels/ops/attention/flash_mla_sm120.py:698-722`）。

## 01 先把 `TP8 + LayerSplit` 排除掉

三层约束串起来，它在 8 卡上自动不成立。每一层都是源码里的 `raise`：

```text
1. LayerSplit 只允许出现在 PD 的 prefill worker
     arg_groups/model_hook.py:365-380（decode 与非 PD 两条分支各有一个 raise）
2. 因此开 LayerSplit 的部署必然还有一个 decode worker
     docs/docs/advanced_features/prefill_cp.mdx:92
3. PD 两端的有效 TP 宽度必须对得上
     prefill 的 attn_tp_size × attn_cp_size == decode 的 attn_tp_size
     conn.py:1376-1388，对不上直接 RuntimeError
4. 于是 TP8 prefill 需要 8 卡 decode → 合计 16 卡
     8 卡机上唯一能容纳的 PD 切法是 prefill 4 + decode 4
```

还有一条更硬的：LayerSplit 要求 `--pp-size 1`（`model_hook.py:402-408`）。
**PP 和 LayerSplit 在配置层面就是互斥的** —— 这不是二选一的性能题，是先被断言判掉一半。

> 反过来读更有用：LayerSplit 的设计前提是「prefill 和 decode 已经是两拨机器」。
> 单机 unified 里，prefill 用的 KV 就是 decode 要用的 KV，压它没有额外收益。

## 02 真正可比的三张牌

记 `L` = 层数，`b` = 每 token 每层 KV 字节，`V` = 每卡留给 KV 的显存，`S` = `attn_cp_size`。
MLA 后端下 KV 在 TP 组内是**复制**的，所以「每卡持有几层」直接决定能装多少 token。

| 配置 | 每卡持有层数 | 可容纳 token | 相对容量（vs A） | 额外通信 |
|---|---|---|---|---|
| A · TP8 unified | `L` | `V/(L·b)` | 1.00× | all-reduce（本来就有） |
| B · TP4 × PP2 unified | `⌈L/2⌉` | `V/(⌈L/2⌉·b)` | ≈ 1.97× | stage 边界 P2P + bubble |
| C · PD 4+4，prefill 带 LayerSplit(S=4) | decode `L` ／ prefill `⌈L/4⌉+1` | decode `V/(L·b)` | 1.00×（decode 侧） | prefill 每层广播一整层 + mooncake |

**PP2 拿到近乎两倍 token 容量，PD 4+4 一点也没多** —— 它只是把 4 张卡从 decode 挪给 prefill。
除非瓶颈确实在 prefill（长 prompt、TTFT 主导），否则 PD 4+4 是净亏：decode 算力减半、容量不变、
还多一套 mooncake 与故障面。

## 03 6000D 特有的两个坑

### 3.1 DSA 后端：trtllm 被拒绝

LayerSplit 的 e2e 用例跑在 B300 上，用 `--dsa-prefill-backend trtllm`。照抄到 SM120 上启动即失败：

```text
GLM DSA with FP8 KV cache on NVIDIA SM120/SM121 supports only
flashinfer_sparse_mla (default) or triton_sparse_mla (prefill), but got ['trtllm'].
```

`kernels/ops/attention/flash_mla_sm120.py:712-722`；自动选择在 `arg_groups/overrides.py:641-654`
（`is_glm_sm12_fp8` 分支）。**正确做法是完全不写 `--dsa-prefill-backend`**，让它落到
`flashinfer_sparse_mla`。这不是调优项，是「不写才对」。

### 3.2 无 NVLink 时，两条路径的通信量差两个数量级

| | 每次传输的字节 | 随什么增长 | 能否被计算盖住 |
|---|---|---|---|
| PP2 的 stage 边界 | hidden states：`tokens × hidden × dtype`（每 chunk 一次） | chunk 大小 | 能，异步 P2P + 双流重叠 |
| LayerSplit 的远程层读 | **整层 buffer：`(size + page_size) × b`**（每个远程层一次） | **整个 KV 池容量**，与 batch 命中页数无关 | 只能靠「提前一层」预取，深度 1 |

第二行「随容量增长」是这条路径最反直觉的地方：**池子撑得越大，每次前向广播量越大**。
NVLink 机上这没关系（摊到几百毫秒 prefill 里只有几个百分点）；PCIe 直连机上它会和层计算正面竞争。
判断方法很朴素：**测「一层 buffer 的 all-broadcast 时间」和「一层 prefill 计算时间」，前者必须明显小于后者。**

## 04 决策规则

| 你的瓶颈 | 选 | 理由 |
|---|---|---|
| 还没到显存墙 | **A · TP8 unified** | 无 bubble、无 PD 故障面、无 mooncake 依赖 |
| 长上下文 / 高并发把 KV 顶爆 | **B · TP4 × PP2** | 每卡层数减半 → 容量 ×1.97；代价是 `(pp−1)/(m+pp−1)` bubble 与 stage P2P |
| TTFT 主导且 decode 负载轻 | **C · PD 4+4 + LayerSplit** | prefill 侧压到 `⌈L/4⌉+1` 层 + CP4 换 TTFT；接受 decode 只剩 4 卡 |
| 已有第二台机器 | **PD 8+8 + LayerSplit** | 这才是目标场景：prefill TP8×CP8，每卡 `⌈L/8⌉+1` 层 |

PP2 的调优口径（`pipeline_parallelism.mdx`）：bubble 主要来自 stage 间耗时不均，
`--enable-dynamic-chunking` 按拟合的二次模型动态缩 chunk，
`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` 默认 0.75。层数除不尽时把多的那层放**高** PP rank
（`SGLANG_PP_LAYER_PARTITION`，文档例子 `15,15,15,16` 优于 `16,15,15,15`）。

## 05 算例（`L=61, b=708 B, V=24 GiB`）

```text
A · TP8         每卡 61 层      → 24GiB / (61×708B)  ≈ 582k token
B · TP4×PP2     每卡 31 层      → 24GiB / (31×708B)  ≈ 1.15M token   （×1.97）
C · PD decode   每卡 61 层      → 24GiB / (61×708B)  ≈ 582k token   （与 A 相同）
C · PD prefill  每卡 ⌈61/4⌉+1=17 层 → 24GiB / (17×708B) ≈ 2.09M token
PP bubble（pp=2, m=4）= 1/(4+1) = 20%
C 的每层广播      ≈ 582k × 708B ≈ 394 MiB，@55 GB/s ≈ 7.2 ms
```

最后一行是关键：7.2 ms 要和你机器上「一层 prefill 的计算时间」比。
预取深度只有 1 层，7.2 ms 若接近或超过单层计算时间，就再也盖不住了。

## 06 两条可执行命令

### B · TP4 × PP2（unified，8 卡全用上）

```bash
python -m sglang.launch_server \
  --model-path <你的 GLM-DSA 权重> \
  --tp 4 --pp 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --enable-dynamic-chunking \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

不写 `--dsa-prefill-backend`：SM120 上会自动落到 `flashinfer_sparse_mla`。
`--enable-dynamic-chunking` 是给 PP bubble 用的，PP1 时不需要。

### C · PD 4+4（prefill 带 LayerSplit）

```bash
# ── prefill：TP4 × CP4 + LayerSplit，4 卡 ──
python -m sglang.launch_server \
  --model-path <你的 GLM-DSA 权重> \
  --tp 4 --attn-cp-size 4 \
  --enable-prefill-cp --cp-strategy interleave \
  --enable-dsa-cache-layer-split \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --base-gpu-id 0 --port 30000

# ── decode：TP4，4 卡，收完整分片 ──
python -m sglang.launch_server \
  --model-path <你的 GLM-DSA 权重> \
  --tp 4 \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode decode \
  --base-gpu-id 4 --port 30001
```

起服务后确认三件事（任一不成立就说明退回了普通路径）：

1. 日志有 `Layer shard plan (continuous): layer_num=…, shard_size=4, rank=…`
   —— 没有这行最常见的原因是漏了 `--attn-cp-size`，会静默退回。
2. 有 `layer-shard broadcast NCCL communicator`，`world_size` 等于 `--attn-cp-size`。
3. 4 个 rank 的 `global=[a, b)` 区间互不相交且并起来是 `[0, L)`。

## 07 最容易踩的四个坑

### ① SM120 后端白名单
```text
GLM DSA with FP8 KV cache on NVIDIA SM120/SM121 supports only flashinfer_sparse_mla
(default) or triton_sparse_mla (prefill), but got ['trtllm'].
```
`kernels/ops/attention/flash_mla_sm120.py:712-722` —— 删掉 `--dsa-prefill-backend` 即可。

### ② PP 与 LayerSplit 互斥
```text
--enable-dsa-cache-layer-split is not supported with pipeline parallelism (pp_size > 1) yet.
```
`model_hook.py:402-408` —— 8 卡上要么 PP2（unified），要么 4+4 PD + LayerSplit。

### ③ 开在非 PD worker 上
```text
--enable-dsa-cache-layer-split is only supported on PD prefill workers. Non-PD workers also run
decode and require ordinary local decode cache semantics.
```
`model_hook.py:376-380` —— unified 部署（不写 `--disaggregation-mode`）时触发。
根因不是性能而是语义：非 owner rank 上没有完整 cache，而 decode 每步都要全层 KV。

### ④ 无报错但 prefill 变慢，且随 KV 池变大而变慢
广播的是整层 buffer（`size + page_size` 行），不是本批次命中的页。
所以把 `max_total_num_tokens` 调大，每层广播时间会同步变大。
诊断：对比「一层 buffer 的 broadcast 时间」与「一层 prefill 计算时间」。

## 08 常见误解

- **「PP 和 LayerSplit 都是按层切，可以叠加。」** 不行，`pp_size > 1` 时直接 raise。
  PP 切**计算图**（每层权重只在一个 stage），LayerSplit 切 **KV 归属**（每层权重处处都有）。
- **「8 卡上 TP8 比 TP4×PP2 省显存。」** 反了。PP2 让每张卡只放一半的层，KV 容量接近翻倍；
  TP8 每张卡都要放全部 `L` 层。PP2 的代价是 bubble 和 stage 边界 P2P，不是显存。
- **「PD 4+4 因为 prefill 侧有 LayerSplit，所以整体容量变大。」** 没有。
  决定并发的是 **decode 侧** KV 容量，而 decode 是普通 TP4、每卡 `L` 层 —— 和 TP8 unified 一样。
- **「CP 也会把 KV 摊到各 rank 上。」** CP 切的是 **token**（interleave/zigzag），不是层。
  每个 CP rank 仍持有全部层的 KV，只是处理的 token 少。真要摊层必须显式开 LayerSplit。

## 09 自测

1. **为什么 PD 4+4 的容量和 TP8 一样，却少了 4 张卡的算力？**
   → KV 容量由「单卡持有几层」决定，而 PD 两侧都是普通 TP。prefill 侧虽有 LayerSplit，
   但 prefill 的 KV 是**过路**的（算完经 mooncake 传走，不承载并发）；承载并发的是 decode 的 radix cache，
   每卡 `L` 层，与 TP8 相同。PD 4+4 是用一半算力换「prefill 不打断 decode」。
2. **PP2 的 bubble 到底多大？**
   → 粗略 `(pp−1)/(m+pp−1)`。pp=2、m=4 时约 20%；m=8 时约 11%。
   连续批处理下实际损失通常小于这个数；但若 `--chunked-prefill-size` 大到单 chunk 吃满一个 stage，
   bubble 会向 50% 靠近 —— 这就是 `--enable-dynamic-chunking` 的用武之地。
3. **有第二台机器时 8+8 怎么配？**
   → prefill 侧 `--tp 8 --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave
   --enable-dsa-cache-layer-split --disaggregation-mode prefill
   --disaggregation-transfer-backend mooncake`，每卡 `⌈L/8⌉+1` 层（L=61 时 9 层，14.8%）；
   decode 侧 `--tp 8` 普通。前提是那台机器确实有 NVLink 域。
4. **6000D 上能不能用 `--enable-dp-attention` 代替 PP？**
   → 方向不同：DP attention 让每个 DP 组处理不同请求、attention 权重按组切，
   **不减少**每卡持有的层数，不解决 KV 容量。它是为了去掉 attention 的 TP 通信。

## 来源

- 源码（v0.5.23）：`arg_groups/model_hook.py`（五道 LayerSplit 门控）、
  `arg_groups/overrides.py:601-740`（DSA 后端按 arch 解析）、
  `kernels/ops/attention/flash_mla_sm120.py:688-723`（SM120 白名单）、
  `mem_cache/dsa_cache_layer_split.py`（广播语义）、`disaggregation/common/conn.py`（PD 两端拓扑比对）
- 文档：`docs/docs/advanced_features/prefill_cp.mdx`、`docs/docs/advanced_features/pipeline_parallelism.mdx`
- 版本边界：LayerSplit 于 v0.5.16 引入（PR #29421）

> 未做运行时实测。6000D 的互连带宽、模型的层数 L 与每 token KV 字节都是计算器的输入，
> 本页不对它们做断言；所有容量数字都是公式推导，不是 benchmark 读数。
