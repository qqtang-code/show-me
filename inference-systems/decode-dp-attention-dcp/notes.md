# DCP 与 DPAttention：TP8×DCP4 与 TP16×attnDP4 两条 1M 路线

> 本文是 [index.html](index.html) 的文字版。术语保留 `rank`、`group`、`scheduler`、`KV pool`、`target`、`draft`、`partial output`、`LSE`、`page`，便于直接检索代码。
>
> 证据口径：**事实**来自工作区源码、配置或日志；**推导**由代码形状与分配公式计算；**实测**引用保存的历史日志，不是本轮重新压测。公开归档隐去内部地址与 run id。

## 01 核心结论

1. **两个机制回答两个不同问题。** DPAttention 决定“哪些请求放在同一组 rank 上”；DCP 决定“同一请求的上下文放在哪些 rank 上”。
2. **TP8×DCP4 用位置拼长度。** `attn_dp_size=1`、`dcp_size=4`，每 rank 269,952 行，逻辑 target 容量 `4 × 269,952 = 1,079,808`，纯 GPU，实测完成过 1,048,188 + 384 的真实 1M 请求。
3. **TP16×attnDP4 用分池换并发与容量。** `attn_dp_size=4`，每池 4 rank；DCP1 时靠 HiSparse 的 `host_to_device_ratio=3` 得 `480,000 × 3 = 1,440,000`，DCP2 时靠组内分片得 `600,000 × 2 = 1,200,000`，两条都实测完成过 1M。
4. **DPAttention 单独不足以上 1M。** 它把容量拆成 4 个独立池，单池仍是 480,000 行；单个 1M 请求只落在一个池，所以必须叠加 host 分层或组内 DCP。
5. **DCP 不减少核心 attention FLOPs。** gathered Q heads ×c、本地 KV ÷c，乘积不变；真正变化的是主 KV 副本数与单请求容量。
6. **两个容易被忽略的代价。** DSA index-K 不随 DCP 分片，TP8×DCP4 的 index-K 仍是 8 份全历史；host 分层路线里，同一份 1M 序列在每个 attnTP rank 的 host 上各存一份（DCP1 时 4 份）。

## 02 两套形态的部署依据

| 维度 | A · TP8 × DCP4 | B · TP16 × attnDP4 |
|---|---|---|
| 硬件 | 单节点 8 卡 | 双节点 16 卡 |
| 并行 | `TP8 × PP1 × DCP4 × attnDP1` | `TP16 × PP1 × attnDP4 × attnTP4`，DCP1 或 DCP2 |
| 每 rank 物理行 | 269,952 | 480,000（B1）/ 600,000（B2） |
| 逻辑 target 容量 | 1,079,808 | 1,440,000（B1）/ 1,200,000（B2） |
| 主 KV 副本 | `attnTP/c = 8/4 = 2` | `4/1 = 4`（B1）或 `4/2 = 2`（B2） |
| 每 rank KV | 16.4481 GiB（全 GPU） | 26.5946 GPU + 72.3424 host GiB（B1） |
| 推测解码 | NEXTN 3 steps / 4 draft | EAGLE 或 NEXTN 3/4 |

**A 的日志证据：** `max_total_num_tokens=1079808`、`max_running_requests=1`、`context_len=1048576`、`--cuda-graph-max-bs-decode 1`；真实请求 `1,048,188 input + 384 output + 4 模板 = 1,048,576` 返回 HTTP 200，bs=1 graph 耗时 85.86 s，no-graph 124.55 s；1P2D 组合 85.89 s、`spec_accept_rate=0.960`。

**B 的日志证据：** GLM 侧为 TP16 × attnDP4 × DCP1 + HiSparse，GPU 主池 480,000、host ratio 3、每请求热集 16,384，8 条 C1 的 1M 请求按 DP0–3 各 2 条完成；Hy4 侧 pd1m 为 `TP16 × DCP2 × attnDP4`、每 rank 600,000、逻辑 1,200,000，1,048,000 + 8 约 86 s、完整上下文约 90 s。

### 与 stock SGLang 的差异（必须先说清）

两套 1M 形态里，**只有 B1 是 stock 允许的组合**。stock 源码对 DSA 模型的 DCP 有三条硬限制，全部在 `arg_groups/model_overrides/deepseek_v2.py: _dsa_dcp_overrides()`：

```python
if cfg.dcp_size <= 1:
    return {}
if cfg.enable_hisparse or cfg.enable_hierarchical_cache:
    raise ValueError("DSA DCP does not support HiSparse or HiCache.")
if cfg.disaggregation_mode != "null":
    raise ValueError("DSA DCP does not support PD disaggregation.")
if cfg.speculative_eagle_topk not in (None, 1):
    raise ValueError("DSA DCP requires --speculative-eagle-topk 1.")
```

PD 解码端还有一条前置条件，在 `disaggregation/common/conn.py`：

```python
if self.dcp_size > 1:
    if not (self.is_mla_backend or self.is_hybrid_mla_backend):
        raise RuntimeError("PD decode DCP requires an MLA or hybrid-MLA KV pool.")
    if info.attn_cp_size != 1:
        raise RuntimeError(
            "PD decode DCP currently requires prefill attention CP=1, "
            f"got {info.attn_cp_size}.")
```

对照本页两套形态：

| 形态 | DCP | PD | HiSparse/HiCache | P 侧 CP | stock 是否允许 |
|---|---:|---:|---:|---:|---|
| A · TP8×DCP4 | 4 | 是 | 否 | CP4 | **否**（PD guard + prefill CP≠1） |
| B1 · TP16×attnDP4 | 1 | 是 | 是（HiSparse） | CP4 | **是**（`dcp_size≤1` 直接返回） |
| B2 · TP16×attnDP4 | 2 | 是 | 否 | CP4 | **否**（PD guard + prefill CP≠1） |

所以 A 与 B2 的 1M 结果是**带 overlay 的构建**跑出来的；本页讲的是这些构建里的机制与实测，但机制本身（owner rule、LSE 归并、加宽容量的公式）都是 stock 代码，overlay 改的是“允许在 PD 里开 DCP”这一层门禁。

### rank 划分

A：`attn_tp_size = 8/1/1 = 8`；DCP 组是 TP 组的连续 4 卡切片，所以 8 卡内有两个组：

```text
DCP 组 1 = ranks 0,1,2,3
DCP 组 2 = ranks 4,5,6,7

position p → DCP 组内 rank (p % 4)，局部行 p // 4
```

每组各持一份完整位置空间，所以主 KV 存 2 份。

B：`attn_tp_size = 16/4/1 = 4`：

```text
DP0 = ranks  0– 3     DP1 = ranks  4– 7
DP2 = ranks  8–11     DP3 = ranks 12–15
```

DCP2 时每组再分两个 pair。

### 配置与约束

| 配置 | 默认 | 约束 |
|---|---:|---|
| `--dcp-size` | 1 | 整除 TP；不能与 `--enable-prefill-cp` 同进程；PD 要求 MLA/hybrid-MLA |
| `--attn-dp-size` | 1 | 整除 TP；不能与 `dp_size>1` 同用 |
| `--dcp-comm-backend` | 未设置 | DCP1 → `ag_rs`；SM120 不满足 `fi_a2a`，DCP>1 解析为 `a2a` |
| `--dcp-replicate-q-proj` | 未设置 | DSA + a2a 默认 true，Q 计算约 ×c |
| `--enable-hisparse` + `--hisparse-config` | false | B1 用它做 host 分层：`{"host_to_device_ratio":3,"device_buffer_size":16384}`；stock DSA DCP 与 HiSparse 互斥 |

源码：`arg_groups/fields/parallel.py`、`arg_groups/parallel_hook.py: handle_decode_context_parallelism()`、`arg_groups/overrides.py: _dcp_comm_backend_default()`、`arg_groups/model_overrides/deepseek_v2.py: _dsa_dcp_overrides()`。

## 03 机制

### 3.1 A：DCP 把位置切给 4 个 rank

```text
owner(slot) = slot % 4
local_row   = slot // 4
logical page = 64 × 4 = 256 tokens
```

一次 decode 步骤：

```text
Q projection（每 rank 8 local heads）
  → 若未开 Q 权重复制：DCP all-gather Q → 32 heads
  → 32 gathered heads × 本地 1/4 KV → partial output + LSE
  → 组内 a2a 交换 output + FP32 LSE
  → Triton log-sum-exp 合并 → 本 rank 8 heads
  → o_proj / 下一层
```

`attn_dp_size=1` 时 `num_dp_ranks = dp_size × attn_dp_size = 1`，layer boundary 的 `move_rows()` 只在布局确有 `ATTN_DP` 差异时才插 gather，所以没有跨 DP 的 hidden gather/scatter。但每层 DCP 侧仍是两次集合通信：Q all-gather（开 `dcp_replicate_q_proj` 可省）+ partial output/LSE 的 a2a。这是 A 与 B 在通信结构上最本质的区别。

### 3.2 B：DPAttention 把请求分进 4 个池

```text
请求 → 某个 DP 组 scheduler（只此一组）
     → 组内 attnTP4 切 head（每 rank 16 heads）
     → attention output 归约
     → 四个组的 token 聚合到完整 TP16 算 FFN/MoE
     → 散回各 DP 组 → 下一层
```

`MLPSyncBatchInfo` 同步 token 数、forward mode、graph eligibility；最慢的组拖关键路径。

### 3.3 组合

A 是 `DP1+DCP4`，B1 是 `DP4+DCP1`，B2 是 `DP4+DCP2`。组合时先按请求分 DP，再在组内按位置分 DCP；DCP 不跨 DP 组共享 KV，也不替代 MoE 前的 DP gather/combine。

## 04 对推理行为的影响

核心账：

```text
每 rank gathered Q heads = Hq / attn_tp_size × c
每 rank 本地 KV 位置     = S / c
核心工作量 ∝ (Hq/t × c) × (S/c) = Hq × S / t
```

所以 DCP 不减少每 rank 的核心 attention FLOPs。DSA 下把 `S` 换成 `K=min(S,2048)` 也近似成立；indexer 全历史扫描与 top-k remap 另算。

| 维度 | A · TP8 × DCP4 | B · TP16 × attnDP4 |
|---|---|---|
| 每 rank Q heads | 8 → gathered 32 | 16（DCP1）/ 16 → 32（DCP2） |
| 每 rank 主 KV | S/4 | S / S/2 |
| 每层通信 | DCP Q gather + partial/LSE a2a，无跨 DP 交换 | 跨 DP hidden gather + reduce-scatter（+DCP2 归并） |
| 同步面 | 单 scheduler，无 DP skew | 4 个 scheduler，慢组拖全局 |
| KV 副本 | 主 KV 2 份；index 8 份 | 主 KV 4 份（B1）/ 2 份（B2）；index 4 份 |
| 单请求 1M 占用 | 15.1994 GiB/rank，8 卡 ≈121.6 GiB | 3.491 GPU + 52.6759 host GiB/rank，4 卡 ≈14.0 + 210.7 GiB |
| 1M 并发 | 1 | 每池 1 条，四池 4 条 |

**Prefill：** 两套 D 侧都不做长 Prefill；P 侧 CP4 完成 prompt 计算，A 走 CP0-only 传输后按 DCP4 重排，B 按普通 MLA 传输。

**Decode：** A 每层多一次 DCP 归并、少读 3/4 主 KV；B 每层多一次跨 DP 交换、读全量 KV。

**小 batch：** A 只有 1 个 scheduler，`max_running_requests=1` 时几乎没有调度收益；B 天然 4 组并发，但单条长请求只占 1 组，其余组空转。

**不要写成“DCP 更快”。** A 的收益是纯 GPU 塞进 1M，代价是 index-K 8 份 + 每层 LSE 归并；B 的收益是 4 个独立池与更大聚合容量，代价是跨 DP 通信、host 带宽与 4 份复制。两者都是容量路线。

## 05 容量公式

### 5.1 每层每 token 字节数

```text
m = 512(FP8 latent) + (512/128)×4(scale) + 64×2(BF16 RoPE) = 656 B
i = 128(FP8 index-K) + 4(FP32 scale) = 132 B

target 主 KV = 78m = 51,168 B/token/rank
target index = 21i =  2,772 B/token/rank
NEXTN draft  = m+i  =    788 B/token/rank
```

### 5.2 通用式

```text
Target main  = (N + 64) × 78m
Ibuf         = (N + 64) × c − 64
Pidx         = floor((Ibuf + 65)/64)
Target index = Pidx × 64 × 21i
Draft main   = (cN + 64) × m
Draft index  = floor((cN + 65)/64) × 64 × i

逻辑容量   = c × N
主 KV 副本 = attn_tp_size / c
1M 门槛    = 1,048,576
```

B1 还要把 `N` 换成 host 槽位数 `floor(3N/64)×64 + 64`，并单独加设备侧热集。

### 5.3 算例 A：TP8 × DCP4

`N=269,952, c=4, p=64`：

```text
Target main  = (269,952+64) × 78 × 656 = 13,816,178,688 B = 12.8673 GiB
Ibuf = 1,080,000 → Pidx = 16,876
Target index = 16,876 × 64 × 21 × 132  =  2,993,937,408 B =  2.7883 GiB
Draft main   = 1,079,872 × 656         =    708,396,032 B =  0.6597 GiB
Draft index  = 16,873 × 64 × 132       =    142,543,104 B =  0.1328 GiB
合计                                   = 17,661,055,232 B = 16.4481 GiB

逻辑容量 = 269,952 × 4 = 1,079,808  (> 1,048,576 ✓)
物理行合计 = 8 × 269,952 = 2,159,616 = 2 × 1,079,808
num_pages = 4,218；逻辑 page = 256
```

### 5.4 算例 B1：TP16 × attnDP4，DCP1 + host ratio 3

```text
GPU main  = (480,000+64) × 78 × 656 = 24,563,914,752 B = 22.8769 GiB
GPU index = 22,501 × 64 × 21 × 132  =  3,991,857,408 B =  3.7177 GiB
GPU 合计                             = 28,555,772,160 B = 26.5946 GiB

host 槽位 = 1,440,064
host main = 73,685,194,752 B = 68.6247 GiB
host 合计 = 77,677,052,160 B = 72.3424 GiB

逻辑容量 = 3 × 480,000 = 1,440,000  (> 1,048,576 ✓)
```

把 draft 层也加上（`c=1, N=480,000` → draft main 314,921,984 B、draft index 63,368,448 B），每 rank 合计约 `99.2895 GiB`；页面计算器的 B 模式用的就是这个口径：GPU 主池 + 覆盖整个 host 逻辑池的 index + draft + host 主池与 index。

### 5.5 变体 B2：TP16 × attnDP4，DCP2

`N=600,000, c=2`：

```text
每 rank 合计 = 34,976,480,000 B = 32.5744 GiB
逻辑容量     = 2 × 600,000 = 1,200,000  (> 1,048,576 ✓)
```

### 5.6 同一个 1M 请求的占用

| 形态 | 每 rank | 组/机合计 | 说明 |
|---|---:|---:|---|
| A · DCP4 | 15.1994 GiB | 8 卡 ≈121.6 GiB | main 12.4922 + index 2.7072；index 全历史 8 份 |
| B1 · DCP1+host3 | 3.491 GPU + 52.6759 host GiB | 4 卡 ≈14.0 + 210.7 GiB | GPU 只放 16,448 行热集 |
| B2 · DCP2 | ≈16.29 GiB | 4 卡 ≈65.2 GiB | 纯 GPU；S/2 主 KV + 全历史 index |

单副本理论值 = `1,048,576 × (51,168 + 2,772) = 52.66 GiB`。A 的 121.6 GiB 来自主 KV 2 份 + index 8 份；B1 的 host 210.7 GiB 来自 DCP1 的 4 份复制。

### 5.7 容量口径必须分开

| 口径 | 含义 |
|---|---|
| 每 rank 物理行 | 决定单卡 KV buffer |
| 单池逻辑容量 | `c×N`（A 即整机；B 只是 4 个池之一） |
| 整机逻辑容量 | `d×cN`，均衡上界 |
| 单请求最大长度 | `min(context_length, Prefill 容量, 逻辑容量, draft/index/metadata 限制)` |
| 等长请求并发 | `min(maxRunning 约束, floor(Prefill/S), d×floor(逻辑/S))` |

## 06 与 SGLang 源码的逐条核对

| 结论 | 源码位置 | 核对 |
|---|---|---|
| `attn_tp = tp / attn_dp / attn_cp` | `runtime_context.py: derive_attn_tp_size()` | ✓ |
| DCP 组是 TP 组的连续 `dcp_size` 切片 | `parallel_state.py`：`tp_group[start:start+dcp_size]` | ✓ TP8/DCP4 有 2 组；TP16/DCP2 有 8 组 |
| `dcp` 必须整除 `tp_size` | `parallel_state.py`：`tp % dcp != 0` → RuntimeError | ✓ 只校验 tp_size |
| 每 rank head = `num_heads / attn_tp` | `deepseek_v2.py:2057` | ✓ |
| DCP 把 head 放大到 local × c | `deepseek_v2.py`：`q.view(-1, num_local_heads*dcp_size, …)`；`dsa_backend.py`：`num_dcp_q_heads` | ✓ |
| Q gather 沿 head 维 | `dcp/comm.py: all_gather_q_for_mla_decode()` | ✓ |
| a2a 单次调用打包 output+LSE | `dcp/comm.py: dcp_a2a_lse_reduce()`，buffer `[N,B,H/N,D+lpd]` | ✓ `lpd=2` |
| 按 LSE 做 log-sum-exp | `dcp_lse_combine_triton()` + `is_mla_dcp_lse_base_on_e()` | ✓ base-e / base-2 分后端 |
| 合并后回到 local heads | `forward_mla.py`：`attn_output.view(-1, num_local_heads, kv_lora_rank)` | ✓ |
| owner 规则 | `layers/dcp/layout.py: maybe_dcp_kernel_indices()` | ✓ `indices[rank::c] // c` |
| 写 KV 时解析 owner | `memory_pool.py: _scatter_mla_rows()` → `set_mla_kv_buffer_dcp_sharded_triton()` | ✓ |
| page table 按 owner 重写 | `dsa/dsa_dcp.py: localize_dcp_page_table()` | ✓ 带 topk 缓存复用 |
| indexer 无 DCP 分支 | `dsa_indexer.py` 无 `dcp`；`dsa_backend.py` 仅两处 | ✓ index-K 每 rank 全历史 |
| index 覆盖加宽空间 | `kv_cache_configurator.py: index_size = max_total_num_tokens * dcp_size`；`memory_pool.py: index_buf_size=(size+page)*c−page` | ✓ |
| allocator size/page 加宽 | `kv_cache_configurator.py`：`*dcp` 两处 | ✓ 逻辑 page `64×c` |
| 逻辑容量 = `N × c` | `kv_cache_configurator.py: logical_token_capacity()` | ✓ `loc_space_scale` 只对 draft worker |
| HiSparse 逻辑池 = 设备池 × ratio | `allocator/hisparse.py`：`_size_full = size*host_to_device_ratio` | ✓ |
| host 槽位向上取整 | `pool_host/base.py: HostKVCache.__init__` | ✓ 1,440,064 |
| DCP 后端默认解析 | `overrides.py: _dcp_comm_backend_default()` | ✓ SM120 → a2a |
| DSA+a2a 默认复制 Q 权重 | `model_overrides/deepseek_v2.py:32-33` | ✓ 仅 decode/target_verify |
| **stock 禁止 DSA DCP + PD/HiSparse** | 同文件 `:19-24` | ⚠ A/B2 需 overlay |
| **PD 解码端要求 prefill CP=1** | `disaggregation/common/conn.py` | ⚠ P 用 CP4，需 overlay |
| attnDP1 无跨 DP 交换 | `layer_boundary/layout.py: move_rows()`；`num_dp_ranks=dp*attn_dp` | ✓ |

## 07 仍需确认

1. **B 的 1M 归属：** GLM 的 8 条 1M 请求是每池 2 条并行完成，单条仍独占一个池；不能说单池支持 2 条 1M。
2. **index-K 与 DCP 的交互：** A 形态 index-K 按全历史每 rank 一份，由 `index_buf_size=(N+64)c−64` 推得，但缺逐 rank 实测字节对照。
3. **通信成本：** Q gather、partial/LSE a2a、DP gather 的真实耗时需 trace；本页只给 buffer 形状。
4. **单因子 A/B：** A 与 B 的差异同时包含 DCP、attnDP、host 分层与 kernel 形状，不能互相归因。
5. **B1 的 host 归属：** draft host pool 与 target host pool 是否同一 packed pool，需按实际 pool 日志确认。
6. **overlay 的具体实现：** A 与 B2 需要去掉 `_dsa_dcp_overrides()` 的 PD guard 和 `conn.py` 的 prefill-CP=1 前置；工作区没有完整保留这两份 decode overlay，所以只能确认“stock 会拒绝”，不能逐行确认改法。

## 附：源码锚点

- `sglang/python/sglang/srt/arg_groups/fields/parallel.py`：`attn_dp_size`、`dcp_size`、`dcp_comm_backend`、`dcp_replicate_q_proj`
- `sglang/python/sglang/srt/runtime_context.py`：`derive_attn_tp_size`、`derive_attention_ranks`、`logical_token_capacity`
- `sglang/python/sglang/srt/distributed/parallel_state.py: initialize_model_parallel()`：TP / attnTP / DCP 组构造
- `sglang/python/sglang/srt/layers/dp_attention.py`：DP gather/scatter
- `sglang/python/sglang/srt/managers/data_parallel_controller.py`：DP scheduler
- `sglang/python/sglang/srt/layers/dcp/{layout,comm,planner}.py`：owner rule、Q gather、LSE combine
- `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`：DCP decode 分支
- `sglang/python/sglang/srt/mem_cache/{memory_pool,kv_cache_configurator,index_key_cache,pool_host/base}.py`：MLA/DSA buffer、DCP 容量、index-K、host 池
- `sglang/python/sglang/srt/disaggregation/common/utils.py`：P→D DCP relayout
