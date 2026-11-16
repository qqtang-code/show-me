# 把上下文做到 1M：DCP、DPAttention 与 HiSparse 的三种切法

> 本文是 [index.html](index.html) 的文字版。术语保留 `rank`、`group`、`scheduler`、`KV pool`、`target`、`draft`、`partial output`、`LSE`、`hot set`、`page`。
>
> 证据口径：**事实**来自工作区源码、配置或日志；**推导**由代码形状与分配公式计算；**实测**引用保存的历史日志，不是本轮重新压测。公开归档隐去内部地址与 run id。

## 01 核心结论

1. **1M 是一个硬指标**：逻辑 target 位置必须 `≥ 1,048,576`。三条实现路径都在这个数字上做乘法，但乘的对象不同。
2. **三种乘法**：DCP 乘位置分片宽度 `c`（`逻辑 = c × N`）；DPAttention 把容量拆成 `d` 个独立池（每池独立 scheduler 与 KV pool）；HiSparse 乘 host/device 倍数 `r`（`逻辑 = r × N`）。
3. **DCP 是上下文维度**：`owner(slot)=slot%c`、`local_row=slot//c`。gathered Q heads ×c 与本地 KV ÷c 相抵，**核心 attention FLOPs 不变**，换的是主 KV 副本数与单请求容量。
4. **DPAttention 是请求维度**：`attn_tp = tp/d/attn_cp`。它**单独不足以上 1M**——单池仍是 480,000 行，必须叠加 host 分层或组内 DCP。
5. **HiSparse 是层级维度**：GPU 只留每请求热集，逻辑池 = 设备池 × ratio；代价是每个 rank 在 host 各存一份全量，以及页交换带宽。
6. **两个最容易被漏的代价**：DSA index-K 不随 DCP 分片（全历史每 rank 一份），也不随热集缩小（覆盖整个逻辑池）。

## 02 三种形态

| 维度 | A · TP8 × DCP4 | B · TP16 × attnDP4 | C · TP8 + HiSparse |
|---|---|---|---|
| 硬件 | 单节点 8 卡 | 双节点 16 卡 | 单节点 8 卡 |
| 并行 | `TP8 × PP1 × DCP4 × attnDP1` | `TP16 × PP1 × attnDP4 × attnTP4`，DCP1 或 DCP2 | `TP8 × PP1 × DCP1 × attnDP1` |
| 1M 的乘法 | 物理行 × DCP4 | 物理行 × host 3 或 × DCP2（每池 4 rank） | 物理行 × host/device 16 |
| 每 rank 物理行 | 269,952 | 480,000（B1）/ 600,000（B2） | 132,096 |
| 逻辑 target 容量 | 1,079,808 | 1,440,000 / 1,200,000 | 2,113,536 |
| 每请求热集 | 无（全驻留） | 16,384（B1）/ 无（B2） | 24,576 |
| 主 KV 副本 | 2（attnTP/c = 8/4） | 4（DCP1）或 2（DCP2） | 8（DCP1，attnTP 8） |
| 每 rank KV | 16.4481 GiB GPU | 26.9471 GPU + 72.3424 host GiB | 11.8515 GPU + 106.1778 host GiB |
| 1M 单请求占用 | 15.1994 GiB/rank | 3.491 GPU + 52.6759 host | 3.8814 GPU + 52.6759 host |
| 可驻留 1M 请求 | 1 | 每池 1，四池共 4 | 2 |

### 与 stock SGLang 的差异

stock 源码对 DSA 模型的 DCP 有三条硬限制，全在 `arg_groups/model_overrides/deepseek_v2.py: _dsa_dcp_overrides()`：

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

PD 解码端另有一条前置条件，在 `disaggregation/common/conn.py`：

```python
if self.dcp_size > 1:
    if info.attn_cp_size != 1:
        raise RuntimeError(
            "PD decode DCP currently requires prefill attention CP=1, "
            f"got {info.attn_cp_size}.")
```

对照：

| 形态 | DCP | PD | HiSparse | P 侧 CP | stock 允许 |
|---|---:|---:|---:|---:|---|
| A · TP8×DCP4 | 4 | 是 | 否 | CP4 | **否** |
| B1 · TP16×attnDP4 | 1 | 是 | 是 | CP4 | **是**（`dcp_size≤1` 直接返回） |
| B2 · TP16×attnDP4 | 2 | 是 | 否 | CP4 | **否** |
| C · TP8+HiSparse | 1 | 是 | 是 | CP4 | **是** |

机制本身（owner rule、LSE 归并、加宽公式、双分配器）都是 stock 代码；overlay 改的是「允许在 PD 里开 DCP」这一层门禁。

### rank 布局

A：`attn_tp_size = 8/1/1 = 8`，DCP 组是 TP 组的连续 4 卡切片，8 卡内有两个组：

```text
DCP 组 1 = ranks 0,1,2,3      DCP 组 2 = ranks 4,5,6,7
position p → 组内 rank (p % 4)，局部行 p // 4
```

每组各持一份完整位置空间，所以主 KV 存 2 份。

B：`attn_tp_size = 16/4/1 = 4`：

```text
DP0 = ranks 0–3    DP1 = ranks 4–7    DP2 = ranks 8–11    DP3 = ranks 12–15
```

DCP2 时每组再分两个 pair。

C：`attn_tp_size = 8`，DCP1、attnDP1，同一批请求由 8 个 rank 协同，每 rank 持一份完整主 KV 副本（8 份）。

### 关键配置

| 配置 | 默认 | 约束与用途 |
|---|---:|---|
| `--dcp-size` | 1 | 整除 TP；不能与 `--enable-prefill-cp` 同进程；PD 要求 MLA/hybrid-MLA |
| `--attn-dp-size` | 1 | 整除 TP；不能与 `dp_size>1` 同用 |
| `--dcp-comm-backend` | 未设置 | DCP1 → `ag_rs`；SM120 不满足 `fi_a2a`，DCP>1 → `a2a` |
| `--dcp-replicate-q-proj` | 未设置 | DSA + a2a 默认 true，Q 计算约 ×c |
| `--enable-hisparse` + `--hisparse-config` | false | B1 用 `{"host_to_device_ratio":3,"device_buffer_size":16384}`；C 用 `{"host_to_device_ratio":16,"device_buffer_size":24576}` |

## 03 机制

### 3.1 A：DCP 把位置切给 4 个 rank

```text
owner(slot) = slot % 4
local_row   = slot // 4
logical page = 64 × 4 = 256 tokens

Q projection（每 rank 8 local heads）
  → 未开 Q 权重复制时：DCP all-gather Q → 32 heads
  → 32 gathered heads × 本地 1/4 KV → partial output + LSE
  → 组内 a2a 交换 output + FP32 LSE → Triton log-sum-exp 合并
  → 本 rank 8 heads → o_proj
```

`attn_dp_size=1` 时 `num_dp_ranks = dp_size × attn_dp_size = 1`，`layer_boundary/layout.py: move_rows()` 不会插 `ATTN_DP` 的 gather。每层 DCP 侧是两次集合通信：Q all-gather（可省）+ partial/LSE a2a。

### 3.2 B：DPAttention 把请求分进 4 个池

```text
请求 → 某个 DP 组 scheduler
     → 组内 attnTP4 切 head（每 rank 16 heads）
     → 四个组的 token 聚合到完整 TP16 算 FFN/MoE → 散回各 DP 组
```

`MLPSyncBatchInfo` 同步 token 数、forward mode、graph eligibility。单池 480,000 行 < 1M，所以还要 host 3×（B1）或组内 DCP2（B2）。

### 3.3 C：HiSparse 把历史挪到 host

```text
logical_allocator: size_full = N × r = 132,096 × 16 = 2,113,536
device_allocator : size_hisparse = N = 132,096
每请求 GPU 热集 = device_buffer_size + page_size = 24,576 + 64 = 24,640

DSA top-k 选页 → swap_in_selected_pages() 搬进热集 → LRU 淘汰
index_buf_size = size × host_to_device_ratio  → index 也覆盖整个逻辑池
```

`HiSparseTokenToKVPoolAllocator` 同时维护 logical 与 device 两个 allocator，并保存 `full_to_hisparse_device_index_mapping`；DCP1 意味着每 rank 都是完整副本。

### 3.4 三者能叠加吗

DCP 与 attnDP 正交（B2 就是 `DP4+DCP2`）；DCP 与 HiSparse 在 stock 里互斥；`dcp_size` 又不能与 prefill CP 同进程。现实组合：A = `DP1+DCP4`，B1 = `DP4+DCP1+HiSparse(3)`，B2 = `DP4+DCP2`，C = `DP1+DCP1+HiSparse(16)`。

## 04 对推理行为的影响

核心账：

```text
每 rank gathered Q heads = Hq / attn_tp_size × c
每 rank 本地 KV 位置     = S / c
核心工作量 ∝ (Hq/t × c) × (S/c) = Hq × S / t
```

| 维度 | A · TP8 × DCP4 | B · TP16 × attnDP4 | C · TP8 + HiSparse |
|---|---|---|---|
| 每 rank Q heads | 8 → gathered 32 | 16；DCP2 时 16 → 32 | 8 |
| 每 rank 主 KV | S/4 | S 或 S/2 | S 的副本，GPU 只留热集 |
| 每层集合通信 | Q gather + partial/LSE a2a | 跨 DP gather + reduce-scatter（+DCP2 归并） | 无；有 host↔device 页搬运 |
| 关键路径 | LSE 归并每层都在 | DP 组间同步 + MoE | 页换入 I/O + host 带宽 |
| KV 副本 | 主 2 份，index 8 份 | 主 4 或 2 份，index 4 份 | 主 8 份，index 8 份 |
| 1M 并发 | 1 | 每池 1，共 4 | 2 |
| 主要风险 | index 8 份；每层归并 | 4 份复制；DP 长尾 | host 带宽；热集命中率 |

三条都是容量路线，不是算力路线：同样的 1M 请求，A 占 15.2 GiB/rank 显存，C 只占 3.88 GiB 显存但要 52.7 GiB host。

**分阶段**：Prefill 三条都由 P 侧 CP4 完成，DCP 不改变 Prefill 计算；Decode 时 A 每层多一次归并、少读 3/4 主 KV，B 每层多一次跨 DP 交换、读全量 KV，C 读热集、命中之外等页搬运。长输出下 A 的归并与被读历史无关，C 的历史越长页命中越主导 TPOT。

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
Ibuf         = (N + 64) × c − 64          # 未开 host 分层时
Pidx         = floor((Ibuf + 65) / 64)
Target index = Pidx × 64 × 21i
Draft        = (cN + 64) × m  +  floor((cN+65)/64) × 64 × i

逻辑容量   = c × N      # DCP
           = r × N      # HiSparse
主 KV 副本 = attn_tp_size / c
1M 门槛    = 1,048,576
```

HiSparse 形态下 index 覆盖整个逻辑池（`index_buf_size = N × r`），所以 index 字节随 `r` 增长。

### 5.3 算例 A：TP8 × DCP4

`N=269,952, c=4`：

```text
Target main  = (269,952+64) × 78 × 656 = 13,816,178,688 B = 12.8673 GiB
Ibuf = 1,080,000 → Pidx = 16,876
Target index = 16,876 × 64 × 21 × 132  =  2,993,937,408 B =  2.7883 GiB
Draft        =    708,396,032 + 142,543,104 B             =  0.7925 GiB
每 rank 合计                                               = 16.4481 GiB（全 GPU）

逻辑容量 = 269,952 × 4 = 1,079,808（高 3.0%）
物理行合计 = 8 × 269,952 = 2,159,616 = 2 × 1,079,808
```

### 5.4 算例 B1：TP16 × attnDP4，DCP1 + host ratio 3

```text
GPU main  = (480,000+64) × 78 × 656 = 24,563,914,752 B = 22.8769 GiB
GPU index = 22,501 × 64 × 21 × 132  =  3,991,857,408 B =  3.7177 GiB
GPU draft = 314,921,984 + 63,368,448 B                =  0.3527 GiB
GPU 合计                                              = 26.9471 GiB

host 槽位 = 1,440,064 → host 合计 = 72.3424 GiB
每 rank GPU+host = 99.2895 GiB
逻辑容量 = 3 × 480,000 = 1,440,000（高 37.3%）
```

### 5.5 变体 B2：TP16 × attnDP4，DCP2

`N=600,000, c=2`：每 rank 合计 `32.5744 GiB`，逻辑容量 `2 × 600,000 = 1,200,000`（高 14.5%）。

### 5.6 算例 C：TP8 + HiSparse，N=132,096，r=16

```text
GPU main  = (132,096+64) × 78 × 656 = 6,762,362,880 B =  6.2979 GiB
index 页数 = floor((2,113,536+65)/64) = 33,025
GPU index = 33,025 × 64 × 21 × 132  = 5,858,899,200 B =  5.4565 GiB
Draft     = 86,696,960 + 17,445,120 B               =  0.0970 GiB
GPU 合计                                            = 11.8515 GiB

host 槽位 = 2,113,600
host main = 100.7213 GiB；host index = 5.4565 GiB
host 合计 = 106.1778 GiB
每 rank GPU+host = 118.0293 GiB

逻辑容量 = 132,096 × 16 = 2,113,536（高 101.6%）
可驻留 1M 请求 = floor(2,113,536 / 1,048,576) = 2
```

### 5.7 同一个 1M 请求的占用

| 形态 | 每 rank | 合计 | 说明 |
|---|---:|---:|---|
| A · DCP4 | 15.1994 GiB GPU | 8 卡 ≈121.6 GiB | main 12.4922 + index 2.7072；index 8 份 |
| B1 · DCP1+host3 | 3.491 GPU + 52.6759 host | 4 卡 ≈14.0 + 210.7 GiB | GPU 只放 16,448 行热集 |
| B2 · DCP2 | ≈16.29 GiB GPU | 4 卡 ≈65.2 GiB | S/2 主 KV + 全历史 index |
| C · HiSparse16 | 3.8814 GPU + 52.6759 host | 8 卡 ≈31.1 + 421.4 GiB | 热集 24,640 行；host 每 rank 一份全量 |

单副本理论值 = `1,048,576 × (51,168 + 2,772) = 52.66 GiB`。A 的 121.6 GiB 来自主 KV 2 份 + index 8 份；B1/C 的 host 侧都是 DCP1 的整份复制。

### 5.8 容量口径

| 口径 | 含义 |
|---|---|
| 每 rank 物理行 | 决定单卡 KV buffer |
| 单池逻辑容量 | `c×N`（DCP）或 `r×N`（HiSparse） |
| 整机逻辑容量 | `d×单池`，均衡上界 |
| 单请求最大长度 | `min(context_length, Prefill 容量, 单池逻辑容量, draft/index 限制)` |
| 等长请求并发 | `min(maxRunning 约束, floor(Prefill/S), d×floor(单池/S))` |

## 06 与 SGLang 源码的逐条核对

| 结论 | 源码位置 | 核对 |
|---|---|---|
| `attn_tp = tp / attn_dp / attn_cp` | `runtime_context.py: derive_attn_tp_size()` | ✓ |
| DCP 组是 TP 组连续切片 | `parallel_state.py`：`tp_group[start:start+dcp_size]` | ✓ TP8/DCP4 → 2 组 |
| `dcp` 只校验整除 `tp_size` | 同文件 RuntimeError | ✓ |
| 每 rank head = `num_heads/attn_tp` | `deepseek_v2.py:2057` | ✓ |
| DCP head 放大到 local×c | `q.view(-1, num_local_heads*dcp_size, …)`；`num_dcp_q_heads` | ✓ 8→32 |
| Q gather 沿 head 维 | `dcp/comm.py: all_gather_q_for_mla_decode()` | ✓ |
| a2a 单次打包 output+LSE | `dcp_a2a_lse_reduce()`，buffer `[N,B,H/N,D+lpd]` | ✓ lpd=2 |
| 按 LSE 做 log-sum-exp | `dcp_lse_combine_triton()` + `is_mla_dcp_lse_base_on_e()` | ✓ |
| 合并后回 local heads | `forward_mla.py`：`.view(-1, num_local_heads, kv_lora_rank)` | ✓ |
| owner 规则 | `layers/dcp/layout.py: maybe_dcp_kernel_indices()` | ✓ |
| 写 KV 时解析 owner | `memory_pool.py: _scatter_mla_rows()` | ✓ |
| page table 按 owner 重写 | `dsa/dsa_dcp.py: localize_dcp_page_table()` | ✓ |
| indexer 无 DCP 分支 | `dsa_indexer.py` 无 `dcp`；`dsa_backend.py` 仅两处 | ✓ index 每 rank 全历史 |
| index 覆盖加宽/放大的池 | `index_size = max_total_num_tokens*dcp_size`；HiSparse `index_buf_size = size*ratio` | ✓ C 的 index 到 5.46 GiB |
| allocator size/page 加宽 | `kv_cache_configurator.py`：`*dcp` 两处 | ✓ 逻辑 page 64×c |
| 逻辑容量 = `N×c` | `logical_token_capacity()` | ✓ `loc_space_scale` 只对 draft |
| HiSparse 双分配器 | `allocator/hisparse.py`：`_size_full`、`_size_hisparse` | ✓ 2,113,536 / 132,096 |
| 热集换入 | `hisparse_coordinator.py`：`padded_buffer_size`、LRU | ✓ 24,576 → 24,640 |
| host 槽位向上取整 | `pool_host/base.py: HostKVCache.__init__` | ✓ 2,113,600 |
| DCP 后端默认解析 | `overrides.py: _dcp_comm_backend_default()` | ✓ SM120 → a2a |
| DSA+a2a 默认复制 Q 权重 | `model_overrides/deepseek_v2.py` | ✓ 仅 decode/target_verify |
| **stock 禁止 DSA DCP + PD/HiSparse** | `_dsa_dcp_overrides()` | ⚠ A/B2 需 overlay |
| **PD 解码端要求 prefill CP=1** | `disaggregation/common/conn.py` | ⚠ P 用 CP4，需 overlay |
| attnDP1 无跨 DP 交换 | `layer_boundary/layout.py: move_rows()` | ✓ |

## 07 仍需确认

1. **C 形态的实测：** TP8 + HiSparse（132,096 槽 / ratio 16 / 热集 24,576）的容量按公式推导；工作区没有对应的运行日志，所以不给该配置的 TTFT/TPOT。
2. **B 的 1M 归属：** 8 条 1M 请求是每池 2 条并行完成，单条仍独占一个池。
3. **index-K 与 DCP 的交互：** index 按全历史每 rank 一份，由 `index_buf_size` 公式推得，缺逐 rank 实测字节对照。
4. **通信与 I/O 成本：** Q gather、partial/LSE a2a、DP gather、页交换的真实耗时需 trace。
5. **overlay 实现：** A 与 B2 需去掉 `_dsa_dcp_overrides()` 的 PD guard 与 `conn.py` 的 prefill-CP=1 前置；两份 decode overlay 未完整保留，只能确认「stock 会拒绝」。

## 附：源码锚点

- `sglang/python/sglang/srt/arg_groups/fields/parallel.py`：`attn_dp_size`、`dcp_size`、`dcp_comm_backend`、`dcp_replicate_q_proj`
- `sglang/python/sglang/srt/runtime_context.py`：`derive_attn_tp_size`、`derive_attention_ranks`、`logical_token_capacity`
- `sglang/python/sglang/srt/distributed/parallel_state.py: initialize_model_parallel()`：TP / attnTP / DCP 组构造
- `sglang/python/sglang/srt/layers/dp_attention.py`：DP gather/scatter
- `sglang/python/sglang/srt/layers/dcp/{layout,comm,planner}.py`：owner rule、Q gather、LSE combine
- `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`：DCP decode 分支
- `sglang/python/sglang/srt/mem_cache/{memory_pool,kv_cache_configurator,index_key_cache,hisparse_memory_pool,allocator/hisparse,pool_host/base}.py`：MLA/DSA buffer、DCP 容量、index-K、HiSparse 双分配器与 host 池
- `sglang/python/sglang/srt/managers/hisparse_coordinator.py`：热集槽位、LRU 与页换入
- `sglang/python/sglang/srt/disaggregation/common/{utils,conn}.py`：P→D DCP relayout 与 PD 前置条件
