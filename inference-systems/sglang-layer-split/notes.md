# SGLang 的 LayerSplit：把 DSA 的 KV/indexer cache 按层摊到 prefill CP 的各 rank

> 适用版本：SGLang **v0.5.23**（2026-11-02）。该 flag 在 **v0.5.16** 引入（PR #29421，2026-07-09 合入 main）。
> 本页由读源码产出，**未做运行时实测**；所有数字都由文中式子推出。

SGLang 里叫 `layer split` 的东西只有一个：`--enable-dsa-cache-layer-split`。
它不新增并行维度，也不改变任何 kernel 的计算形状 —— 它只挪动一件事：**第 l 层的 KV 落在哪张卡上**。

## 术语约定

- **LayerSplit**：只指 `--enable-dsa-cache-layer-split` 打开的那套机制。它与 PP 帮助文本里那句
  `"the model is split by layer"` 不是一回事 —— 后者切的是计算图，LayerSplit 切的是 KV cache 的归属。
- **rank / owner**：CP 组内的序号与「拥有某层 KV 的 rank」；**scratch**：每个 rank 保留的那一份整层远程 buffer。
- **DSA**（DeepSeek Sparse Attention）、**MLA**、**PD**（prefill/decode 分离）、**CP**（context parallel）保持英文。

## TL;DR

1. **切的是 cache 的归属，不是计算。** TP 宽度、CP 的 token 分配、请求到 rank 的映射全都不变。
2. **每 rank 只 materialize `ceil(L/S)` 层**（`S = attn_cp_size`），外加 **1 层 remote scratch**；
   单卡 KV 从 `L` 层降到 `ceil(L/S)+1` 层。
3. **只有 PD 的 prefill worker 能开**：`--disaggregation-mode prefill`、
   `--enable-prefill-cp --cp-strategy interleave`、`--attn-cp-size > 1`、`--pp-size 1`、backend 为 mooncake。
4. **非 owner 层的读 = owner 广播整层 buffer**：广播的是整个 `(size + page_size)` 行，不是本批次的 page。
5. **它买的是 prefill 侧容量，不是单请求延迟。** 同显存下 token 容量约 ×`S`。

## 01 要解决的问题：MLA 的 KV 在 TP 组内是复制的

SGLang 的 MLA 后端里，每个 TP rank 都持有**完整的** KV cache —— 不是 `1/TP`。
多头潜在注意力把 K/V 压成一个共享 latent（`kv_lora_rank + qk_rope_head_dim`），
各 head 读同一份 latent，按 head 切没有意义，TP 只切权重与计算。
**加宽 TP 不会让 KV 变小，只会让算力变大。**

DSA 模型在这之上还有一份 indexer 的 index-K cache，同样逐层、同样在 TP 组内复制。
prefill 侧要处理整段长上下文，KV + indexer 一起吃满显存，于是 `max_total_num_tokens` 被压下来，
prefill 吞吐跟着掉。

LayerSplit 的做法：既然 KV 是「层 × token × dim」的三维张量，而层这个维度在 TP 组内本来就一模一样，
那就**把层按 CP rank 分掉**。第 `l` 层的 KV 只由它的 owner 写，别的 rank 要读就向 owner 广播一份到 scratch。
复制变成了按层分持。

> 为什么挂在 CP 上而不是 TP 上？分完之后每张卡都缺一部分层，读远程层必须有一次集合通信把整层搬过来。
> CP 组本来就是「同一次前向里、处理同一批 token 的若干 rank」，天然是这次广播的作用域；
> 源码里广播组的选法就是 `get_parallel().attn_cp_group`（`mem_cache/dsa_cache_layer_split.py:253-269`）。

## 02 角色与边界

| 角色 | 谁 | 持有什么 | 落点 |
|---|---|---|---|
| owner rank | 按连续区间分到的 CP rank | 该层**完整**的 `(size + page_size, 1, kv_dim)` 行 | `dsa_cache_layer_split.py:222` |
| 非 owner rank | 其余 CP rank | 该层 **0 行**占位，保持层号对齐 | `dsa_cache_layer_split.py:306-320` |
| remote scratch | 每个 rank 各一份 | 正好**一整层**（indexer 侧另有 `remote_buffer`） | `dsa_cache_layer_split.py:320-330` |
| 广播通信器 | `attn_cp_group` 上的 `PyNcclCommunicator` | 专用 NCCL communicator + `kv_broadcast_stream` | `dsa_cache_layer_split.py:253-269` |
| PD 传输口 | 每个 prefill CP rank | 只上报**自己那段连续层**（`layer_shard_start` 起） | `disaggregation/prefill.py:272-299` |
| 不参与 | draft worker（MTP / EAGLE） | 保持完整 cache；门控第一项是 `not is_draft_worker` | `layers/cp/utils.py:45-56` |
| 不参与 | decode worker | 普通本地 decode cache；开在 decode 上直接报错 | `arg_groups/model_hook.py:365-380` |

## 03 机制

### 3.1 层归属：连续、均衡、前 rem 个 rank 多一层

```text
get_layer_shard_range(rank, S, L):
  base  = L // S
  rem   = L %  S
  start = rank * base + min(rank, rem)
  end   = start + base + (1 if rank < rem else 0)
  return start, end
```

`layers/cp/utils.py:97-106`；`get_layer_owner()` 是反函数（`:108-126`）。

`L = 10, S = 4`：

```text
r0 → [0, 3)   r1 → [3, 6)   r2 → [6, 8)   r3 → [8, 10)
owner(layer 0..9) = [0,0,0, 1,1,1, 2,2, 3,3]
```

单测逐项断言过：`test/registered/unit/mem_cache/test_dsa_layer_shard_utils.py:14-27`。
`L = 2, S = 4` 时后两个 rank 拿到空区间 `[2, 2)`，它们仍有 scratch，只是所有层都要广播。

### 3.2 分配：owner 拿满，别人拿 0 行，另备一份 scratch

```text
kv_buffer[i] = (size + page_size, 1, kv_dim)   if _is_layer_owned(start_layer + i)
             = (0,              1, kv_dim)     otherwise      # 0 行占位
remote_kv_buffer = (size + page_size, 1, kv_dim)                # 全 rank 各一份
```

`dsa_cache_layer_split.py:306-343`。「0 行占位」让 `kv_buffer[layer_id - start_layer]` 的寻址不用改，
所有按层遍历（`move_kv_cache`、HiCache CPU offload、PD buffer 上报）原样工作，遇到 0 行 `continue`。

`+1` 的 scratch 也进显存账：cell size 按 `effective_num_layers = max(1, ceil(L/S) + 1)` 算
（`layers/cp/utils.py:76-95`，被 `model_executor/pool_configurator.py:349` 调用）。所以：

```text
单卡 KV 占比 = (ceil(L/S) + 1) / L
token 容量倍数 = L / (ceil(L/S) + 1)
```

GLM cookbook 里的「up to 75%」（`docs/cookbook/autoregressive/GLM/GLM-5.2.mdx:337`）
是忽略 scratch 的渐近值 `1 - 1/S`，`S = 4` 时正是 75%。

### 3.3 写：只有 owner 落盘，写会作废远程副本

`set_kv_buffer()` / `set_mla_kv_buffer()` 写完本 rank 的 buffer 后，若 `remote_kv_layer_id == layer_id`
就置 `None`（`:395-420`、`:422-448`）。indexer 侧同理（`invalidate()`，`:129-132`）。

### 3.4 读：owner 广播进 scratch，且提前一层预取

```text
get_key_buffer(l) / get_value_buffer(l)
  └─ _get_broadcastable_kv_buffer(l)                        # :507
       if remote_kv_layer_id == l: return remote_kv_buffer   # 命中
       src = kv_buffer[l] if _is_layer_owned(l) else None
       _broadcast_tensor_from_owner(remote_kv_buffer, l, src_tensor=src)   # :271
       remote_kv_layer_id = l
       return remote_kv_buffer
```

`_broadcast_tensor_from_owner()` 里的 `comm.broadcast(tensor, src=owner_rank)` 广播的是
**`remote_kv_buffer` 整个张量**（`size + page_size` 行全量），与 batch 命中多少 page 无关。

让它可用的是**一层预取**：`DeepseekV2DecoderLayer` 算第 `l` 层注意力之前，先把第 `l+1` 层的广播踢出去
（`layers/attention/dsa/utils.py:312-328` → `prefetch_kv_buffer()`，`:461-505`）：

```text
current stream   : 算第 l 层注意力（读已就位的 remote_kv_buffer）
kv_broadcast_stream: 同时发起第 l+1 层广播（先 wait_stream(current)，再切流 broadcast）
第 l+1 层要读时  : _finalize_pending_kv_broadcast() 把 pending 提升为 remote_kv_layer_id
                   并让当前 stream wait_stream 广播流（:450-459）
```

owner 自己也走这条路：它的 `src_tensor` 是自己那份满 buffer，于是先 `copy_` 进 scratch 再作为 source
参与集合通信 —— **集合通信必须所有 rank 都进，owner 不能缺席，只能多拷一次。**

### 3.5 PD 传输：每个 rank 只报自己那段连续层

| 改动 | 原来 | LayerSplit 下 | 落点 |
|---|---|---|---|
| 上报哪些层 | 全部层的 buffer 指针 | 只报 owned 层（0 行天然跳过） | `dsa_cache_layer_split.py:362-375` |
| 层段标签 | `start_layer` / `end_layer` | `layer_shard_start` 起，长度 = 指针个数 | `disaggregation/prefill.py:282-299` |
| 谁发 | CP 组内只有 rank 0 发 | **全 CP rank 都发**（`cp_sharded_prefill`） | `disaggregation/common/conn.py:385-391` |
| CP 复制状态 | 非 0 号 rank 跳过 | 不跳过 —— 每 rank 拥有不同状态层 | `disaggregation/common/conn.py:513-526` |
| draft cache | — | 只由最后一个 shard rank 发 | `disaggregation/prefill.py:277-279` |

第 4 行的源码注释：`"Cache layer split is the exception because each CP rank owns different state layers."`
这不是优化，是正确性要求。

## 04 配置项、默认值与取值范围

| 参数 | 默认 | 取值 | LayerSplit 下的要求 |
|---|---|---|---|
| `--enable-dsa-cache-layer-split` | `False` | bool | 必须显式打开 |
| `--enable-prefill-cp` | `False` | bool | 必须打开 |
| `--cp-strategy` | `None` | `zigzag` \| `interleave` | 必须 `interleave` |
| `--attn-cp-size` | `1` | int > 1 | 必须 > 1，它就是 `S` |
| `--tp` | `1` | int | 需被 `dp × attn_cp_size` 整除；推荐 DP=1、CP=TP |
| `--disaggregation-mode` | `null` | `prefill` \| `decode` \| `null` | 必须 `prefill` |
| `--disaggregation-transfer-backend` | `mooncake` | `mooncake`\|`nixl`\|`ascend`\|`fake`\|`mori`\|`mooncake_tcp` | 只接受 `mooncake` / `mooncake_tcp` |
| `--pp-size` | `1` | int | 必须为 1 |
| 模型架构 | — | — | 须落在 `is_deepseek_dsa()` 白名单且带 `index_topk` |
| `--kv-cache-dtype` | `auto` | `fp8_e4m3` 等 | 无强制；indexer 的 index-K 另算 128+4 字节/层/token |

架构白名单在 `configs/model_config.py:191-212`，判据是「arch 命中列表 **且** `index_topk is not None`」。
**不是「MLA 就能开」，是「有 DSA indexer 才能开」。**

> **静默退回**：`get_glm_dsa_cp_layer_shard_info()` 在 `attn_cp_size <= 1` 时返回 `(None, 1)`，
> 于是 `kv_cache_configurator.py:2352` 的 `elif` 不成立、退回普通 pool。
> flag 开了但没给 `--attn-cp-size`，服务能起来、日志不吵，只是什么也没切。

## 05 显存 / 通信 / 吞吐 / 延迟

### 5.1 显存

```text
per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes   # MLA latent
                    + (index_head_dim + index_head_dim/128 × 4)        # DSA index-K + scale, fp8
                    = 576 + 132 = 708 B                                  # fp8 示例

单卡 KV = (ceil(L/S) + 1) × tokens × per_token_per_layer
基线    =  L              × tokens × per_token_per_layer
```

indexer 项来自 `model_executor/pool_configurator.py:560-571`（`quant_block_size = 128`，
`memory_pool.py:5280`；`index_kpool` 默认 1）。index-K 用独立 fp8 buffer，不跟 `--kv-cache-dtype` 走。

### 5.2 通信

```text
每次前向的广播量 ≈ (L - ceil(L/S)) × (size + page_size) × per_token_per_layer
```

量级完全由 `size`（`max_total_num_tokens`）决定 —— 而 LayerSplit 恰好把这个池子撑大了 `S` 倍。
**容量与广播量同向增长。** 缓解手段是那层预取，以及不要盲目加大 `S`。

### 5.3 吞吐与延迟

| 维度 | 方向 | 原因 |
|---|---|---|
| prefill 吞吐 | ↑ 约 ×S | 同显存下 `max_total_num_tokens` 涨 |
| prefill 单层延迟 | ↑ 一点 | 远程层的读要等广播；预取把等待压到一次 stream 同步 |
| prefill 显存 | ↓ 到 `(ceil(L/S)+1)/L` | 3.2 的式子 |
| decode 延迟 | 不变 | LayerSplit 不在 decode worker 上跑 |
| PD 传输量 | **不变** | 同一份 KV 换个发送方：从 1 个 rank 发全量 → S 个 rank 各发 1/S |

最后一行容易被误解成「传输量变多」。没有：`get_contiguous_buf_infos()` 报的是 owned 层，
S 个 rank 报的是**互不相交**的层段，加起来正好是原来那一份。变的是**并发度**，不是总量。

### 5.4 示例算例（`L = 61, S = 4, tokens = 65536, fp8`）

```text
per-token/layer   = 576 + 132                 = 708 B
每层 buffer       = 65536 × 708               = 44.25 MiB
每 rank 层数      = ceil(61/4) + 1            = 17
单卡 KV           = 17 × 44.25 MiB            = 752 MiB
基线              = 61 × 44.25 MiB            = 2.64 GiB
省下              = 1 − 17/61                 = 72.1%
远程层            = 61 − 16                   = 45
每次前向广播      = 45 × 44.25 MiB            = 1.94 GiB
按 600 GB/s       ≈ 3.48 ms
token 容量倍数    = 61 / 17                   = 3.59×
```

## 06 它和 TP / PP / DP 到底什么关系

| 维度 | 切什么 | 谁持有第 l 层的**权重** | 谁持有第 l 层的**KV** | 通信原语 |
|---|---|---|---|---|
| TP | 每层权重矩阵（按 head / 列切） | 全部 rank 各持 1/TP | **全部 rank 各持一份完整**（MLA） | all-reduce / all-gather |
| PP「model is split by layer」 | 计算图按层段切 stage | 只有该 stage 的 rank | 只有该 stage 的 rank | 点对点 `send_tensor_dict` |
| DP | 请求批次 | 每个副本都完整 | 每个副本都完整 | 无（负载均衡层） |
| CP（prefill CP） | 一条序列的 token | 每个 CP rank 都完整 | 每个 CP rank 都完整 | all-gather |
| **LayerSplit** | **KV/indexer cache 的层归属** | 不变（仍由 TP 决定） | **只有 owner 持有，其余读 scratch** | broadcast（owner → CP 组） |

**TP / PP / DP / CP 是「谁来算」，LayerSplit 是「谁存」。** 前四者改变计算图与 kernel 形状；
LayerSplit 一个 kernel 都不改，它只是给 `kv_buffer` 装了一个「按层查表 — 查不到就去 owner 拿」的间接层。

| 组合 | 能不能 | 说明 |
|---|---|---|
| × TP | 必须 | `S = attn_cp_size` 嵌在 TP 里；推荐 CP=TP、DP=1 |
| × CP | 寄生 | 没有 CP 就没有广播作用域；`--enable-prefill-cp` 是硬前提 |
| × PP | **不行** | `pp_size > 1` 直接 raise |
| × DP attention | 受限 | DSA interleave 文档口径是「keep DP=1」 |
| × PD | 必须 | 只有 prefill worker 能开 |
| × EAGLE / MTP | 并存但不切 | draft worker 保持完整 cache |
| × EP / MoE A2A | 正交 | e2e 的 mixed-MoE 变体只在 decode 侧叠 |

## 07 最小配置示例

取自 `test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit.py:38-91`，4 卡 prefill + 4 卡 decode。
`--tp 4` 与 `--attn-cp-size 4` 相等，即文档推荐的「CP=TP」拓扑，`S = 4`。

```bash
# ── prefill worker：开 LayerSplit 的那一侧 ──
python -m sglang.launch_server \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 --attn-cp-size 4 \
  --enable-prefill-cp --cp-strategy interleave \
  --enable-dsa-cache-layer-split \
  --dsa-prefill-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6

# ── decode worker：普通本地 cache，靠 PD 收完整分片 ──
python -m sglang.launch_server \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode decode \
  --base-gpu-id 4 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
```

起服务后确认三件事：

1. 日志有 `Layer shard plan (continuous): layer_num=…, shard_size=4, rank=…, local=[…), global=[…), partitions=…`
   （`dsa_cache_layer_split.py:237-251`）；没有这行 = pool 没换成 LayerSplit 版本。
2. 日志有 `Initialized dedicated layer-shard broadcast NCCL communicator: rank=…, world_size=…`
   （`:253-269`）；`world_size` 应等于 `--attn-cp-size`。
3. 各 rank 的 `global=[a, b)` 区间**互不相交且并起来是 `[0, L)`**。

## 08 限制、注意事项与易错点

每条带原文错误串，诊断顺序是「先看报错属于哪一层」——
`ServerArgs` 的校验有先后顺序，同时违反两条时只会看到先触发的那条。

### ① 模型架构门
```text
--enable-dsa-cache-layer-split is only supported for DSA (DeepSeek Sparse Attention) models.
```
`arg_groups/model_hook.py:187`
1. 确认 `architectures[0]` 在 `is_deepseek_dsa()` 白名单里。
2. 确认 `hf_config.index_topk is not None` —— 白名单命中但没有 `index_topk` 同样为假。
3. 普通 MLA 模型（无 indexer）没有这条路。

### ② PD 角色门（decode 分支）
```text
--enable-dsa-cache-layer-split is not supported on decode workers. This flag is a prefill-CP
optimization; decode receives full cache shards through PD transfer.
```
`arg_groups/model_hook.py:365-375` —— 把 flag 从 decode 侧的启动命令里删掉。

### ③ PD 角色门（非 PD 分支）
```text
--enable-dsa-cache-layer-split is only supported on PD prefill workers. Non-PD workers also run
decode and require ordinary local decode cache semantics.
```
`arg_groups/model_hook.py:376-380` —— 没写 `--disaggregation-mode`（unified 部署）时触发。
unified 里同一 worker 既 prefill 又 decode，而非 owner rank 没有完整 cache，decode 会读到错的层。

### ④ CP 形状门
```text
--enable-dsa-cache-layer-split requires --enable-prefill-cp and --cp-strategy interleave.
```
`arg_groups/model_hook.py:381-387` —— 两个 flag 缺一不可；`zigzag` 同样触发。
GLM 的 cookbook 另有一条：GLM-5.2 的 zigzag prefill CP 本身暂不可用。

### ⑤ 传输后端门
```text
--enable-dsa-cache-layer-split currently only supports the mooncake transfer backend
(mooncake / mooncake_tcp). Got --disaggregation-transfer-backend 'nixl'.
mori/nixl support will be added later by the community.
```
`arg_groups/model_hook.py:388-401` —— 依赖「mooncake 的全 CP rank KV/indexer 传输」，
源码注释直接写明 mori/nixl 是临时限制。默认值就是 `mooncake`，只有显式改过后端才会撞上。

### ⑥ PP 门
```text
--enable-dsa-cache-layer-split is not supported with pipeline parallelism (pp_size > 1) yet.
It requires prefill context parallelism, and CP + PP has not been validated for this feature.
```
`arg_groups/model_hook.py:402-408` —— PP 与 CP 的组合在别处是支持的，叠上 LayerSplit 就没验证过。

### ⑦ 构造断言
```text
LayerSplitDSATokenToKVPool requires layer_shard_size > 1
```
`mem_cache/dsa_cache_layer_split.py:203-205` —— 正常路径被 `get_glm_dsa_cp_layer_shard_info()`
的 `(None, 1)` 挡在外面；看到它是有人绕过门控直接构造池子。

### ⑧ 无报错，但结果错乱
`dsa_cache_layer_split.py:289-295` 的源码注释描述了这个坑：
1. `PyNcclCommunicator` 默认 `disabled=True`，只在 CUDA graph capture 期间由 `change_state` 打开。
2. 直接 `comm.broadcast(...)` 是**静默 no-op**：非 owner rank 读到 scratch 里的旧内容，
   不报错、不崩、只是算错。
3. 源码在广播处显式包了 `with comm.change_state(enable=True)`。**改这条路径时这行不能省。**
4. 诊断信号：GSM8K 之类的精度「偶发、与 batch 组成相关」地掉；单请求、短序列可能完全正常。

## 09 常见误解

- **「LayerSplit 就是 PP 的另一种叫法。」** PP 的帮助文本确实写着 `"the model is split by layer"`，
  但那是**计算图切段**：第 l 层的权重只在一个 stage 上。LayerSplit 不动权重、不动计算图，
  切的是 KV cache 的**归属**，而且恰恰要求 `pp_size == 1`。
- **「S 越大省得越多，所以往大了开。」** 显存确实单调下降，但远程层占比同时在涨，
  而每次远程层读要搬一整层。S 从 4 加到 8，显存再多省一点，广播量接近翻倍。
- **「广播只传这批请求命中的那几页 KV。」** 不是。广播的是 `remote_kv_buffer` 整个张量，
  与 batch 命中多少 page 无关。
- **「decode 侧也能开。」** 被显式拒绝。原因不是性能而是语义：非 owner rank 上没有完整 cache，
  而 decode 每一步都要全层 KV。
- **「开了 prefill CP 就等于配好了 PD。」** 文档专门写了
  `"enabling prefill CP alone does not configure a PD deployment"`。

## 10 自测

1. `L = 61, S = 8`，每张卡几层？省多少？和「up to 75%」对得上吗？
   → `ceil(61/8)+1 = 9` 层，占 14.8%，省 **85.9%**。75% 是 `S = 4` 的渐近值 `1 - 1/S`（忽略 scratch）。
   同一式子 `S = 4` 得 `(16+1)/61 ≈ 27.9%`，省 72.1%。引用时要写清 `S` 和是否含 scratch。
2. owner 自己也读自己那层，为什么还要 `copy_` 进 scratch？
   → 集合通信**所有 rank 都必须进**，包括 source。少进一个就是死锁，不是「少拷一次」。
3. 为什么 `get_contiguous_buf_infos()` 要重写，而 `get_kv_size_bytes()` 可以直接累加？
   → 非 owner 层是 **0 行**而不是不存在，0 行贡献 0 字节所以求和天然正确；
   但 PD 传输要把「第几层对应哪个指针」交出去，必须只报 owned 层再配 `layer_shard_start`。
4. 为什么 draft（MTP / EAGLE）不跟着切？
   → draft 是另一套池子，且 `is_glm_dsa_cache_layer_split_enabled()` 第一项就是
   `not is_draft_worker`。draft 层数少、KV 占比低，切它收益有限却要在 draft 前向里引入广播。
   对应地 PD 传输侧由**最后一个 shard rank** 替 draft 发全量。
5. LayerSplit 之后 PD 传输总量会变吗？
   → 不会。S 个 rank 上报互不相交的层段，并起来等于原来那一份。变的是并发度。
   代价是 CP 复制状态不能再只由 rank 0 发（`conn.py:513-526`）。

## 来源

- 源码（v0.5.23）：`python/sglang/srt/mem_cache/dsa_cache_layer_split.py`、`layers/cp/utils.py`、
  `mem_cache/kv_cache_configurator.py`、`model_executor/pool_configurator.py`、
  `arg_groups/{fields/parallel.py, model_hook.py, choices.py}`、
  `disaggregation/{prefill.py, common/conn.py}`、`layers/attention/dsa/utils.py`、`configs/model_config.py`
- 文档：`docs/docs/advanced_features/prefill_cp.mdx`、`docs/cookbook/autoregressive/GLM/GLM-5.2.mdx`
- 用例：`test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit.py`、
  `test/registered/unit/mem_cache/test_dsa_layer_shard_utils.py`、
  `test/registered/unit/mem_cache/test_dsa_layer_split_broadcast.py`
- 版本边界：v0.5.16 引入（PR #29421，2026-07-09 合入 main）；v0.5.15 及更早无此文件与 flag。

> 未做运行时实测。5.4 的所有数字都是 5.1 / 5.2 两个式子推出的派生量，不是 benchmark 读数；
> 带宽是自填的有效值，不是任何一张卡的规格。
