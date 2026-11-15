# Decode 的 DPAttention 与 DCP：请求分池、位置分片与 KV 容量

> 本文是 [index.html](index.html) 的文字版。术语约定：`rank`、`group`、`scheduler`、`backend`、`block/page`、`target/draft`、`partial output` 与 `LSE` 保留英文，便于直接 grep 代码。
>
> 证据口径：**事实**来自工作区的启动配置、代码或历史文档；**推导**是从这些张量形状和分配公式计算得到；**历史实测**只引用保存下来的日志，不是本次重新压测。公开归档中隐去内部地址、run id 和未公开模型版本，分别写作「GLM 系列」与「Hy 系列」。

## 01 先给结论

1. **D 侧就是 PD 分离中的 Decode 角色，不是 draft，也不是“第 D 个节点”。** 两套代表部署都把 Prefill 与 Decode 放在不同节点组，由 router 连接，Mooncake 负责 P→D 的 KV 传输。
2. **DPAttention = attention data parallelism。** `attn_dp_size` 默认 1；大于 1 时在同一个 TP world 内建立多个 attention 请求/scheduler 组。`attn_tp_size = tp_size / attn_dp_size / attn_cp_size`。它切的是**请求与请求的 KV 池**，不是把单条请求的序列切开。FFN/MoE 仍可在完整 TP 组上运行。
3. **DCP = decode context parallelism。** `dcp_size` 默认 1；大于 1 时在 attention TP 内按位置交错分配同一请求的 target MLA 主 KV。它不是 Prefill CP；当前 P 侧的 CP4 与 D 侧 DCP2 是两个独立配置。
4. **两套代表配置的 D 侧拓扑同为 TP16 × attnDP4 × attnTP4，但 GLM 系列是 DCP1，Hy 系列 r15 才是 DCP2。** 因而不能把 Hy 的 DCP2 数据或结论套到 GLM 当前配方上。
5. **DPAttention 增加的是独立请求池数，DCP 才增加单请求的逻辑位置空间。** 在每 rank 有 `N` 个 target 主 KV 物理行时，单个 DP 组的 target 逻辑容量是 `cN`；`d` 个 DP 组在均衡调度下为 `d×cN`。但端到端仍受 Prefill 池、模型 `context_length`、draft、metadata 与调度上限约束。
6. **Hy 的 DCP2 不是“所有 KV 都不复制”。** attnTP4 内有两个 DCP pair；一个 token 在每个 pair 各有一个 owner，所以 target 主 KV 副本数由 DCP1 的 4 降为 2，而不是 1。DSA index-K 和 NEXTN draft 仍是 4 份。
7. **计算量不能笼统说下降。** DCP 的 gathered Q head 数乘 `c`，本地 KV 长度除 `c`，核心 attention 的 `heads × positions` 理想上不变。新增成本是 Q 准备、partial output + LSE 归并、page table 重写和同步；收益主要来自容量、kernel 形状及更多可驻留请求，成立与否取决于负载。
8. **历史结果不支持“DCP 必然更快”。** Hy r6→r15 是组合改动，不能把 TPOT 改善全部归因于 DCP；DCP4 虽扩大逻辑容量，实测 TPOT、输出 TPS 与 NEXTN accept 均退化。

## 02 部署与实现依据

### 2.1 代表配置及边界

| 项目 | GLM 系列 · unified release（2026-11-10 配方） | Hy 系列 · r15（2026-11-12 配方） |
|---|---|---|
| 总拓扑 | 四台 8-GPU 节点，P 两台 + D 两台；不是单一 32-rank communicator | 同左 |
| Prefill | `TP4 × PP4 × CP4 × EP4`，`attnDP1`，PP 层分配 `18,21,21,18` | `TP4 × PP4 × CP4 × EP4`，`attnDP1`，默认 PP 分层 `19,19,20,20` |
| Decode | `TP16 × PP1 × attnDP4 × attnTP4 × DCP1`，`EP1` | `TP16 × PP1 × attnDP4 × attnTP4 × DCP2`，`EP1` |
| Decode backend | DSA + HiSparse；GPU 主池 480,000；host ratio 3；每请求热集 16,384 | 定制 SM120 sparse MLA + fused sink/LSE；target 每 rank 400,000 行 |
| 推测解码 | EAGLE，steps=3 / topk=1 / draft=4 | NEXTN，steps=3 / topk=1 / draft=4 |
| P→D | Mooncake | Mooncake；CP0-only 发送 + 异步固定槽池 |
| 配置出处 | `recipes/v1/recipe.glm53-pd-unified-hisparse-r1.json` | `docs/HY4_CURRENT_LAUNCH_DCP2_20261112.sh`（公开归档省略内部路径细节） |

**模型执行路径：**当前 GLM checkpoint 的实际 `config.json` 为 `GlmMoeDsaForCausalLM / glm_moe_dsa`，78 层、64 query heads、`hidden_size=6144`；`glm4_moe.py` 中该类直接继承 `DeepseekV2ForCausalLM`。它不是 `glm5_next.py` 的 45 层 KDA 混合模型，不能混用 KDA/Flash 的 KV 结构。Hy 的 `HYV4Attention` 也继承 `DeepseekV2AttentionMLA`，但额外有 `linear_gate` 和 `learnable_sink_param`，所以 DCP 需要 single-owner sink 语义。

**补充时间线：** GLM 的 11 月 13 日 literal 部署有 GPU 运行证据；11 月 14 日更新的 Responses release 仅做 CPU/构建验收，未宣称替换在线服务。因此本文比较的是“最近 GPU 运行配置及与其 DP/DCP/KV 参数一致的 release”，不断言实时容器状态。Hy 在 2026-11-12 早些时候的文档还记录“stock trtllm kernel 不支持 SM120，DCP 被阻断”；后续 r6–r15 通过自定义 sparse-MLA/LSE 路径运行 DCP2。不能拿早期阻断记录否定后续 r15，也不能拿 r15 反推 stock kernel 已支持 SM120。11 月 13–14 日还有 DCP4、TP8/DP1 与 PD1M 容量实验；它们是不同拓扑，不能混进 r15 的延迟或容量对比。

### 2.2 rank 划分

GLM 与 Hy 的 D 侧 `attn_dp_size=4`、`attn_cp_size=1`，所以：

```text
attn_tp_size = tp_size / attn_dp_size / attn_cp_size = 16 / 4 / 1 = 4

DP0: ranks  0,  1,  2,  3   — 节点 0
DP1: ranks  4,  5,  6,  7   — 节点 0
DP2: ranks  8,  9, 10, 11   — 节点 1
DP3: ranks 12, 13, 14, 15   — 节点 1
```

依据：`runtime_context.py: derive_attn_tp_size()` 与 `derive_attention_ranks()`，rank 布局 `(dp, cp, tp)`，TP rank 变化最快。P/D 各有自己的 world；`world_rank = tp_size * pp_rank + tp_rank`。

Hy DCP2 在每个 attnTP4 内再建立两个 contiguous DCP group：

```text
DP0 / attnTP4
  DCP pair A = ranks 0,1   → dcp_rank = 0,1
  DCP pair B = ranks 2,3   → dcp_rank = 0,1

token position p mod 2 = 0 → ranks 0,2 持有 target 主 KV
token position p mod 2 = 1 → ranks 1,3 持有 target 主 KV
```

依据：`runtime_context.py` 中 `dcp_rank = tp_rank % dcp_size`，`parallel_state.py: initialize_model_parallel()` 在每个 TP 组内按连续切片创建 `_DCP`。这也说明 target 主 KV 的副本因子是 `attn_tp_size / dcp_size = 2`。

### 2.3 配置项、默认值与启用条件

| 配置 | 默认值 | 代码确认的启用条件与关键约束 |
|---|---:|---|
| `--attn-dp-size` | 1 | 大于 1 启用 DPAttention；`tp_size % attn_dp_size == 0`。不能与独立的 `dp_size>1` 同时使用。旧的 `--dp-size N --enable-dp-attention` 会解析为 `attn_dp_size=N, dp_size=1`。 |
| `--dcp-size` / `--decode-context-parallel-size` | 1 | 大于 1 启用；不得与 `--enable-prefill-cp` 同用；必须能整除 attention TP 宽度。PD 路径要求 MLA/hybrid-MLA。 |
| `--dcp-comm-backend` | 未显式设置 | DCP1 解析为 `ag_rs`（实际不起作用）；DCP>1 在 CUDA/ROCm/XPU 上，若 `fi_a2a` 不满足则选 `a2a`，其他环境为 `ag_rs`。本批 SM120 不满足 `is_sm100`，所以走 `a2a`。 |
| `--dcp-replicate-q-proj` | 未显式设置 | DSA + `a2a/fi_a2a` 时默认 true：预先把 Q projection 与 `w_kc` 的 head shard gather 成 full-head 权重，省去每层 Q all-gather；代价是更多静态权重与重复 Q 计算。仅 decode / target-verify，extend 不走这一优化。 |
| `--enable-prefill-cp` | false | P 侧开关，需配 `--cp-strategy interleave`。不是 DCP。 |
| `--enable-cp-decode-attn-tp` | false | GLM 配方显式开启，控制 CP 场景下 decode 的 attention 权重切片；不等于 `--dcp-size>1`，也不表示 KV 按 DCP 分片。 |
| `--enable-hisparse` | false | GLM 当前启用。stock DSA DCP 明确拒绝 HiSparse/HiCache 与 PD disaggregation，故当前 GLM 不能直接叠加 DCP。 |

文件与符号：

- `arg_groups/fields/parallel.py`：`tp_size`、`dcp_size`、`attn_dp_size`、`attn_cp_size`、`dcp_comm_backend`、`dcp_replicate_q_proj` 的声明和默认值。
- `arg_groups/parallel_hook.py: handle_decode_context_parallelism()`、`handle_data_parallelism()`：互斥与整除检查；DPAttention 还把 `chunked_prefill_size` 除以 `attn_dp_size`。
- `arg_groups/model_overrides/deepseek_v2.py: _dsa_dcp_overrides()`：stock DSA DCP 拒绝 HiSparse/HiCache、PD 与非 `speculative_eagle_topk=1`，并在 `a2a` 下默认启用 Q 权重复制。
- `arg_groups/overrides.py: _dcp_comm_backend_default()`：通信后端解析。

### 2.4 作用于哪个阶段

- **长输入 Prefill：** 两套都由 P 侧 CP4 完成；D 侧的 DPAttention/DCP 不加速这段计算。GLM 的 DCP1 无影响；Hy DCP2 只改变 P→D target KV 的落盘位置与 metadata。
- **Decode：** 两者都生效于 target attention。DCP 还覆盖 `target_verify`，以及 DSA 的 `extend` 分支；`is_dcp_mla_decode_phase()` 明确列出 `decode / target_verify / DSA extend`。
- **NEXTN draft：** `DeepseekV2AttentionMLA.__init__` 在 `is_nextn and use_dsa` 时把 `self.dcp_size` 固定为 1，因此 target 可以 DCP2，draft 不随 target DCP 分片。
- **FFN/MoE：** DPAttention 不把 FFN 限制到 attnTP4；FFN/MoE 仍在完整 TP16 上运行。DCP 只改变 attention/KV 路径，不把 MoE 改为 context parallel。

## 03 DPAttention 的具体工作机制

### 3.1 请求、权重、KV 的分布

| 内容 | 在一个 attnTP4 / DP 组内 | 跨四个 DP 组 |
|---|---|---|
| 请求 | 由一个 scheduler 管理，只处理路由到本 DP 组的请求 | 不同组处理不同请求；支持按 `routed_dp_rank` 定向，常规策略由 load-balance 配置决定 |
| attention Q/KV_B/O projection | 按 attnTP4 切 head，每 rank 持本组 1/4 | 同一套 head shard 在其他 DP 组重复，attention 权重总副本数增加 |
| Q_A / latent KV 投影等 replicated linear | 按相应 replicated layer 保留 | 不是“整模型 DP”；不能据此推断所有权重只复制 4 份 |
| FFN/MoE 权重与计算 | 仍按完整 TP16 / EP 配置组织 | DP gather 把四个组的 token 放回全 TP MoE 路径 |
| target 主 KV（DCP1） | 本组请求的完整 latent KV 在 4 个 rank 各一份 | 只存在于请求所属组，不在其他组复制 |
| allocator / req_to_token | 本组独立 | 没有一个跨四组共享的 token allocator |

依据：`managers/data_parallel_controller.py` 按 `compute_dp_attention_world_info()` 为进程确定 DP rank；`disaggregation/kv_events.py` 明确写出每个 attention-DP rank 拥有自己的 KV shard；`models/deepseek_v2.py` 的 Q_B/KV_B/O projection 使用 `parallel_group="attn_tp"`。

### 3.2 一次 forward 的数据流

以下是代码路径的简化，不声称每一层都只有这些 collective：

```text
每个 DP 组独立 scheduler
  └─ 本地 batch 与 metadata 在 TP 组同步；无请求的组可构造 idle batch
       └─ attention：本地 token × 本地 head shard × 本组 KV
            └─ attnTP 内完成 output partial sum
                 └─ DP gather：四个 DP 组的 token 重新进入完整 TP 的 FFN/MoE 输入
                      └─ MoE / dense FFN（TP16 / EP 配置）
                           └─ reduce-scatter + 必要的 attnTP gather，回到 DP-local 行
                                └─ 下一层
```

通信原语由 padding 和模型边界决定，不能预先固定为某一种：

- `dp_attention.py` 的 MAX_LEN 路径可有 attention-TP 内的 `reduce_scatter_tensor` 加 TP `all_gather_into_tensor`；combine 在 `tp_size != attn_dp_size` 时是全 TP reduce-scatter 后再在 attnTP 内 gather。
- SUM_LEN 可走 zero-buffer `all_reduce`；另有显式 opt-in 的 `all_gatherv/reduce_scatterv`，不能把 gatherv 当成默认。

**可复核的 buffer 形状（不是网络实测流量）：**当前 GLM 的 hidden size 为 6144，BF16 每行 12,288 B。MAX_LEN 下全局 hidden buffer 为 `4×max(n_d)` 行；若四个 DP 各 4 行，则形状 `[16,6144]`、196,608 B，单 rank scattered chunk 为 `[1,6144]`、12,288 B。`MLPSyncBatchInfo` 本地为 11 个 int64（88 B），全 TP16 gather buffer 为 `[16,11]`（1,408 B），随后读取四个 DP 的 tp0 信息为 `[4,11]`（352 B）。这些数字说明张量尺寸，不能直接当作 NCCL 算法在网络上唯一、固定的总字节数。
- attention output 的 sum 由 `layer_boundary` 完成；GLM/Hy 的 `o_proj(reduce_results=False)` 并不意味着“attention 不需要归约”。
- `MLPSyncBatchInfo.all_gather()` 同步 token 数、forward mode、graph eligibility 等；任意 DP 组慢或需要 eager，可能影响其他组。DPAttention 并非四条完全独立、永不互相等待的流水线。

**GLM unified 的特化：**当前配方另设 `SGLANG_PD_D_HIERARCHY_MAX_ROWS=4`。在实际镜像对应的 `d_hierarchy.py` 中，满足“全 TP16、连续二维 BF16、宽度 6144、输入行数 1–4、输出行数 16×输入”时，AllGather 先做跨节点 pair `[r,r+8]`，再做节点内 8-rank 组集合通信，最后按 `rank_order` 重排；ReduceScatter 反向执行。未命中则回退通用 NCCL。`03-dp-attention.patch` 在通用 DP gather/scatter 前调用这条路径。它优化的是 DPAttention 的跨 DP 激活交换，**不是 DCP，也不改变 KV owner**。镜像哈希对账见 `docs/results/glm53/pd-four-host/20261110-unified-release-r1/report-source.json`，实现来源为 Git `a405481e:runtime/glm53_pd_d_hierarchy/patches/01-d-hierarchy.patch`。

## 04 DCP 的具体工作机制

**两者是独立可组合的轴，而非互相前提。**GLM 证明 `DP4+DCP1` 可独立使用；源码拓扑也允许 `DP1+DCP>1`，Hy r15 则是 `DP4+DCP2`。组合时先把请求分给 DP 组，再在该组的 attention TP 内做 DCP；DCP 不在四个 DP 组之间共享 KV，也不替代 MoE 前的 DP gather/combine。

### 4.1 target 主 KV 的 owner rule 与页映射

代码使用 widened logical slot。对 DCP 宽度 `c`、rank `r`：

```text
owner(slot) = slot % c
local_row   = slot // c
logical page size = physical_page_size × c
```

- `layers/dcp/layout.py: maybe_dcp_kernel_indices()`：以 `indices[r::c] // c` 选取并折叠到本 rank 的物理行。
- `mem_cache/memory_pool.py: MLATokenToKVPool._scatter_mla_rows()`：DCP 写入走 `set_mla_kv_buffer_dcp_sharded_triton`，而不是每 rank 全写。
- `layers/attention/dsa/dsa_dcp.py: localize_dcp_page_table()`：把 page table 中 `slot % c == dcp_rank` 的项保留，改写为 `slot // c` 并压缩有效项。
- `disaggregation/common/utils.py: build_dcp_token_transfer_plan()`：P→D 时按 `(dcp_rank-chunk_start) % c` 选择 source token，destination 位置除以 `c`。draft 仍发送完整序列。
- **target index-K 并不按相同比例分片。** `DSATokenToKVPool` 在未指定 `index_buf_size` 时使用 `(size + page_size) × c - page_size`；因此 target 主 KV 行数为 `N` 时，index-K 的逻辑覆盖约为 `cN`，仍由每个 attnTP rank 持有全历史。

### 4.2 partial attention 如何归并

target decode / verify 的执行顺序：

```text
Q projection（可选 full-head replicated Q，否则先 attn-TP shard）
  → 如未启用 Q 权重复制：DCP all-gather Q head
  → 每 rank 用 gathered Q × 本地 KV shard 计算 partial output + LSE
  → Hy 的 learnable sink 以 single-owner 语义进入归并
  → a2a 交换 partial output + LSE，Triton LSE combine 后回本 rank 的 head block
     （ag_rs 备选是 LSE all-gather + output correction + reduce-scatter）
  → V correction / output projection / 后续 attention-TP 与 MoE 边界
```

`layers/dcp/comm.py: dcp_a2a_lse_reduce()` 证明 stock `a2a` 将 output 与 FP32 LSE 打包进一次 `all_to_all_single`，再执行 `dcp_lse_combine_triton`。softmax 合并不是直接平均 output，而是按 LSE 做 log-sum-exp 加权。

**stock `a2a` 的 packed buffer 账：**设 query 行数为 `B`，聚合后 head 数为 `H=c×Hq/t`，`D=kv_lora_rank`，output 元素字节数为 `e`，`lpd=2`（BF16/FP16 时）。代码分配 `[N,B,H/N,D+lpd]`，所以每 rank 的 send/recv packed buffer 各为：

```text
bytes_rank = B × H × (D + lpd) × e
```

当 `t=4, Hq=64, D=512, e=2` 时，DCP2 为 `B×32,896 B`，DCP4 为 `B×65,792 B`；NEXTN verify 的 `B` 还需按实际 verify 行数计。它是每 rank 的 buffer 尺寸，不是对 NCCL 算法总流量的实测，也不能直接套作 r15 自定义 fused-combiner 的精确线上字节数。若 Q 权重复制关闭，Q all-gather 的输入为 `B×(Hq/t)×Dq×e`，输出再乘 `c`；启用复制后这一项被 full-head Q 计算取代。

Hy 的 learnable sink 不能简单在 each rank 重复加入：它会给每个 query head 的 softmax 分母额外贡献一项。早期 patch `_dcp_apply_single_owner_sink` 证明必须只由拥有相应 gathered head block 的 rank 注入一次；r15 文档进一步描述为“零输出 shard 融合进 Triton combiner”。

### 4.3 P→D 与 decode 的交互发生在哪里

- P 侧 CP4 是 Prefill 的计算并行；Hy 当前 `cp_sharded_prefill=False`（非 hybrid MLA 且未开 DSA cache layer split），所以 CP0 持有完整 KV，且环境变量关闭 all-CP transfer，只有 CP0 发送。
- `send_kvcache_dcp()` 在 P→D 传输时完成 full KV → DCP owner 的映射，因此**decode 每层不需要把全量 KV broadcast 到其他 rank**。
- 每层真正跨 rank 的是 Q（仅在未复制 Q 权重时）、partial output + LSE，以及 page table/metadata；主 KV 保留在 owner rank 本地。
- r15 的 standalone 记录给出 full sparse attention 与 DCP partial + fused combine 的数值差：output max abs `3.05e-5`，合并 LSE max abs `9.54e-7`。这是保存下来的微基准结果，不是本次重跑。

## 05 对推理行为的影响

### 5.1 计算与 kernel：理想化账目

设 query head 总数 `Hq=64`，attention TP 宽 `t=attn_tp_size`，DCP 宽 `c`，序列位置数 `S`。

```text
每 rank 的 gathered Q heads = Hq / t × c
每 rank 的本地 KV positions = S / c
核心 attention 工作量 ∝ (Hq/t × c) × (S/c) = Hq×S/t
```

所以 DCP 在理想均匀分片下**不减少每 rank 的核心 attention FLOPs**，主要改变的是 KV ownership 与 kernel 形状。对 DSA top-k，若 `K=min(S,2048)` 且 top-k 近似均匀落在 DCP rank，则把 `S` 换成 `K` 仍成立；indexer 扫描全历史与 top-k remap 是额外工作，不能省略。

| D 配置 | 每 rank Q heads | 每 rank 主 KV 位置 | 单请求主 KV 副本数（一个 DP 组） | 单个 DP 组逻辑容量 |
|---|---:|---:|---:|---:|
| TP16 / DP1 / DCP1 | 4 | S | 16 | N |
| TP16 / DP4 / DCP1 | 16 | S | 4 | N |
| TP16 / DP1 / DCP2 | 8 | S/2 | 8 | 2N |
| TP16 / DP4 / DCP2 | 32 | S/2 | 2 | 2N |

这里 `N` 是**每 rank target 主 KV 物理行数**，不是整个 DP 组的唯一 token 数。按各自的 DP 组数 `d` 聚合且负载均衡，整机逻辑容量分别约为 `N / 4N / 2N / 8N`；单请求上限分别约为 `N / N / 2N / 2N`，还未考虑 P 侧。

### 5.2 DPAttention 的正面与代价

- **有效并行度：** 全局并发均衡时，每 rank batch 约为原来的 1/4、head 数变为 4 倍，理想 attention FLOPs 总量不变；MoE 仍看到全局 batch。C1 或低并发时，请求只落在一个 DP 组，其他组可能在 attention 阶段空闲，收益不能按 4 倍估计。另需注意，**TP16、attnDP1 在这里主要是理论基线**：历史 warmup 已证明当前 SM120 sparse-MLA 没有 `num_heads=4, topk=2048, d_qk=576` 的 decode kernel，不能把“关掉 DP”写成当前可运行的回退方案。
- **kernel 形状：** 从 TP16 的 4 个 Q head/rank 到 DP4 的 16 个，可能改善 attention kernel 的 tile 利用；同时本地 batch 变小、attention 权重的静态副本和权重复读增加，不能只看 head 数。
- **通信：** attention-TP 集合从 16 rank 缩到 4 rank，但 MoE 前后增加 DP gather/combine 与全局 scheduler 同步。最慢 DP 组会拉长整个 TP 的关键路径。
- **显存与 KV：** 单请求在 4 而非 16 个 rank 持有主 KV，P→D 复制目标和重复存储减少；四个独立池提高聚合唯一 token 容量，但单请求仍受所属池限制，且不均衡会搁浅其他池的容量。
- **TTFT/TPOT：** 长输入 Prefill 在 P 侧，D 的 DPAttention 对 TTFT 主要是传输与元数据影响；TPOT 是否改善取决于并发、MoE 边界通信与 attention kernel 的相对占比，不能只凭池容量判断。

### 5.3 DCP 的正面与代价

- **容量：** target 主 KV 的每 rank 占用约按 `1/c`，主 KV 副本因子从 `t` 降为 `t/c`，逻辑位置空间乘 `c`。这是 DCP 最确定的收益。
- **Q 计算与静态权重：** 默认 `dcp_replicate_q_proj` 路径把 Q projection 和 `w_kc` 在 DCP 组内复制为 full-head，Q 计算约乘 `c`，静态权重增加；省去每层 Q all-gather。关闭该选项则保留较少计算，但多一次 Q 集合通信。
- **partial output + LSE：** 每个 DCP target attention 层都需要归并。payload 随 query row 数、gathered head 数、output dtype 与 `c` 增长，不随历史 KV 总长线性增长；在 NEXTN verify 下，行数还需乘 draft token 数。
- **indexer 与 draft：** DSA index-K 仍覆盖全历史，NEXTN draft 仍 DCP1，因此总显存收益小于“主 KV 除以 c”。长历史下 indexer 成本与 index-K 占用不会随 DCP 同步消失。
- **同步：** DCP collective 位于每个 target attention 的关键路径；`ag_rs` 与 `a2a` 的次数/字节不同，不能混算。stock `a2a` 把 output+LSE 合成一次 all-to-all，随后本地 combine。
- **长输入 TTFT：** P→D 传输要额外做 DCP relayout，但不改变 P 的 CP4 计算。Hy r6→r15 的 TTFT P50 从 10130.3ms 到 10156.1ms，未见改善；这与“DCP 主要作用在 decode”一致，但不能把微小差值当成严格单因子测量。

### 5.4 随负载变化的趋势

| 变量 | 对 DPAttention 的影响 | 对 DCP 的影响 |
|---|---|---|
| 并发/batch 增加 | 各 DP 组更易填满，weight/kernel 利用提高；但 KV 占用、全局 MoE batch 和慢组同步增加 | partial/LSE payload 近似随行数增加；大 batch 更容易摊薄集合通信固定开销，也更早遇到 KV 容量上限 |
| 输入长度增加 | 不直接改变 P 侧 CP4 计算；提高每请求池占用与 P→D 传输量 | 主 KV 容量收益更重要；partial/LSE 归并不随历史长线性增加，但 indexer 全历史成本与 relayout 增大 |
| 已生成长度增加 | 每个 resident 请求持续占池，影响可承载并发与负载均衡 | DSA `K=min(S,2048)` 后核心 sparse attention 近似饱和，indexer 仍随历史增长；DCP 的容量收益通常比 FLOPs 收益更可靠 |
| DCP 宽度增加 | 不改变 DP 组数 | 主 KV 每 rank 约 `1/c`，逻辑容量 `×c`；gathered heads、Q 复制、partial/LSE 与同步增加，DCP4 已出现退化；DCP8 超出 attnTP4 的有效布局 |
| KV 占用增加 | 非均衡池更容易成为瓶颈，其他池空闲也不能直接挪给忙组 | 可容纳更多/更长请求，反而增加 decode 工作集与通信压力，不能把“省显存”直接等同于“降 TPOT” |
| NEXTN | 四个 DP 组仍需同步 graph/forward mode | verify 行数放大约 draft token 数倍，partial/LSE 与 workspace 随之增加；accept rate 会改变最终 TPOT，必须单列 |

## 06 KV 容量与显存公式

### 6.1 先看普通 MHA，防止把公式套错

对普通 MHA/GQA，定义：

- `Lloc`：本 rank 持有的层数（PP 后）；
- `Hkv`：总 KV head 数，`Hkv,loc` 为本 rank 的 KV head 数；
- `dh`：head dimension；`e`：每元素字节数；`S`：序列长度；`R`：本 rank 所属请求数。

未分页、未考虑 DCP 时：

```text
M_request,rank = 2 × Lloc × Hkv,loc × dh × e × S
M_card         = Σrequest M_request,rank + M_page_reserve + M_runtime
```

其中 `2` 是 K/V 两份。若 `Hkv` 小于 attention TP 宽或 kernel 要求复制，必须按实际 `Hkv,loc` 修正，不能永远除以 TP。page 管理下 `S → ceil(S/page)×page`，KV pool 常额外保留 dummy/padding page。

**GLM 与 Hy 的 target 不是这种普通 MHA 布局。** 它们使用 MLA latent KV：K/V 共用 latent，不应再机械乘 2；DSA 另有 index-K。

### 6.2 两项目实际 target 布局

已核对配置/文档与 `DSATokenToKVPool`、`IndexKeyCache`：

```text
每层每 token 主 MLA =
  FP8 latent       512 × 1 B = 512 B
  block scales     (512/128) × 4 B = 16 B
  BF16 RoPE        64 × 2 B = 128 B
                  ------------------
                   m = 656 B

每个 full-indexer 层每 token index-K =
  FP8 index key    128 × 1 B = 128 B
  FP32 scale       1 × 4 B = 4 B
                  ------------------
                   i = 132 B

target：78 层，其中 21 个 full-indexer 层；NEXTN draft：1 层。
```

```text
target 主 KV / token / rank = 78m = 51,168 B
target index-K / token / rank = 21i = 2,772 B
NEXTN main + index / token / rank = m+i = 788 B
```

### 6.3 单请求：DCP target + NEXTN

令物理 page `p=64`、DCP 宽度 `c`：

```text
S_target = ceil(S / (p×c)) × p       # 每 rank 的 target 主 KV 行
S_full   = ceil(S / p) × p           # index-K 与 DCP1 draft 的覆盖
```

则一个请求在所属 DP 组的**每个 attnTP rank**上：

```text
M_req,rank = 78m × S_target           # target 主 KV：按位置分片
           + 21i × S_full             # target index-K：全历史
           + (m+i) × S_full           # NEXTN draft：DCP1，全历史
```

一个请求在整个 attnTP 组的物理占用：

```text
M_req,group = (attn_tp_size/c) × 78m × S_target
            + attn_tp_size × 21i × S_full
            + attn_tp_size × (m+i) × S_full
```

所以 target 主 KV 副本因子是 `attn_tp_size/c`，而 index-K 与 draft 副本因子仍是 `attn_tp_size`。

### 6.4 单卡池与单个并行组

设每 rank target 主 KV 配置为 `N` 行。target、draft 的实际 buffer 可按代码写成：

```text
Target main = (N + p) × 78m

Ibuf = (N + p) × c - p
Pidx = floor((Ibuf + p + 1) / p)
Target index = Pidx × (p × 21i)

Draft main = (cN + p) × m
Pdraft = floor((cN + p + 1) / p)
Draft index = Pdraft × (p × i)
```

单卡 target+draft payload：

```text
M_card = Target main + Target index + Draft main + Draft index
```

单个 DP 组有 `attn_tp_size=4` 张卡，所以 `M_group = 4 × M_card`（权重相同、请求数不同的模型层不重复乘）。D 侧 16 卡为 `M_decode = 16 × M_card`。这些是**KV buffer payload**，不是进程总显存；还需加本地权重、activation、communication/workspace、CUDA graph、allocator 与 host pool。

容量口径必须分开：

| 口径 | target 主 KV 行/容量 | 说明 |
|---|---|---|
| 每 rank 物理行 | N | 决定每卡主 KV buffer；index 与 draft 另算 |
| 单 DP 组逻辑 token | cN | 不能误算成 4N；DCP group 的 owner 已去重，但不同 DCP group 仍有副本 |
| 四个 DP 组逻辑 token | 4cN | 仅在请求可均衡分配到四组时成立；不等价于物理行总数 16N |
| 单请求最大 target 长度 | 约 cN | 还需取 `context_length`、P 侧容量、draft/index/metadata 的最小值 |

### 6.5 GLM 的 HiSparse 修正

GLM 当前不是普通 DCP 池，而是 DCP1 + HiSparse：

- GPU 主池 `N=480,000` 行/rank；`host_to_device_ratio=3`，所以逻辑主池与 index 地址空间为 `3N=1,440,000`。
- 每请求 GPU 热集 `h=16,384`，长序列其余主 KV 在 host；热集限制的是 resident workspace，不是逻辑上下文上限。
- `HiSparseTokenToKVPoolAllocator` 同时维护 `size_full=3N` 的 logical allocator 与 `N` 的 device allocator，并保存 logical→physical mapping。
- `HiSparseDSATokenToKVPool` 把 `index_buf_size` 设为 `3N`，所以 index-K 的 GPU 地址空间也按 3N 计。

```text
GLM 单请求逻辑 target 长度 ≈ min(context_length, Prefill 容量, 3N, draft/host/metadata 容量)
GLM 单 DP 组并发（等长 S） ≈ min(
  max_running_requests 约束,
  floor(3N / S),             # 逻辑地址
  floor(N / hot_tokens_per_request), # 若热集成为限制
  Prefill 并发约束
)
```

stock `_dsa_dcp_overrides` 拒绝 DCP + HiSparse/PD，因此**不能把 Hy 的 DCP2 公式直接叠加到这套 GLM HiSparse 配置上**。

### 6.6 可复核算例 A：GLM unified release

只计算 target KV payload；`N=480,000, p=64, c=1`，HiSparse index ratio 改为 3：

```text
Target main = (480000+64) × 78 × 656
            = 24,563,914,752 B = 22.8769 GiB / rank

Ibuf = 480000 × 3 = 1,440,000
Pidx = floor((1,440,000 + 64 + 1) / 64) = 22,501
Target index = 22,501 × 64 × 21 × 132
             = 3,991,857,408 B = 3.7177 GiB / rank

Target total = 28,555,772,160 B = 26.5946 GiB / rank
```

`HostKVCache` 还按 `page_num=size//64+1` 向上预留，所以 host 槽位为 `22,501×64=1,440,064`，而不是 1,440,000：

```text
Host target main  = 1,440,064 × 78 × 656
                  = 73,685,194,752 B = 68.6247 GiB / rank
Host target index = 1,440,064 × 21 × 132
                  = 3,991,857,408 B = 3.7177 GiB / rank
Host target total = 72.3424 GiB / rank
```

host 主池本身约 `68.6247×8=548.9975 GiB` 每 8 卡节点；target main+index 的 host payload 约 `578.7391 GiB` 每节点。若把 1 个 draft 层也按同一 1,440,064 槽估算，draft main+index 为 `1,440,064×788=1.0568 GiB`，device/host 各自归属仍需按 packed pool 现场确认，不能直接把它当成在线完整占用。

热集还有 page reserve：`16,384+64=16,448 slots/request`；EAGLE scratch 在该配置下为 1,024 slots/request。每组最多 `64//4=16` 个运行请求时，仅这两项名义为 `16×(16,448+1,024)=279,552 slots < 480,000`，但常驻 KV、在途 prealloc 和其他 reserve 仍可能先成为限制。

四个 DP 组：

- target GPU 主池物理行合计：`16 × 480,000 = 7,680,000` 行；这是 buffer 行数，不是唯一 token 容量。
- target 逻辑主池：`4 × 1,440,000 = 5,760,000` tokens（均衡、host+device 口径）。
- target index 的 GPU 地址空间也按每 rank `3N` 计，不能把 host ratio 再重复乘一次到“并发数”上。
- 单请求名义上限：约 `min(1,048,576, Prefill 1,200,000, 1,440,000)`，即模型 1M 级；但发布文档明确保留“完整配方的新 1M 回执仍需补齐”，所以不能把公式当成新实测通过。
- 196,608-token 等长请求：D 逻辑上每组 `floor(1,440,000/196,608)=7`，四组 28；P 侧 `floor(1,200,000/196,608)=6`，所以端到端名义上首先受 P 侧约 6 限制。GPU 热集/工作区、运行上限与生命周期仍可能更早限制。

**未计入：** NEXTN draft、host target/index pool、页级生命周期、权重、activation、通信/graph 预留，因此不能将 26.5946 GiB 当成每卡完整显存占用。

### 6.7 可复核算例 B：Hy r15，DCP2

`N=400,000, p=64, c=2`：

```text
Target main = (400000+64) × 78 × 656
            = 20,470,474,752 B = 19.0646 GiB

Ibuf = (400000+64)×2 - 64 = 800,064
Pidx = floor((800064+64+1)/64) = 12,502
Target index = 12,502 × 64 × 21 × 132
             = 2,217,954,816 B = 2.0656 GiB

Draft main = (2×400000+64) × 656
           = 524,841,984 B = 0.4888 GiB
Pdraft = floor((800000+64+1)/64) = 12,501
Draft index = 12,501 × 64 × 132
            = 105,608,448 B = 0.0984 GiB

Target+draft = 23,318,880,000 B = 21.7174 GiB / rank
```

容量与并发：

- 每 DP 组 target 逻辑容量 `2N=800,000`，四组为 3,200,000；target 主 KV 副本因子为 2。
- D 侧 16 卡 target 主 KV 物理行合计 `16×400,000=6,400,000`，唯一逻辑容量不是 6.4M，而是约 3.2M。
- 196,608-token 请求：D 每组 `floor(800,000/196,608)=4`，四组名义 16；但 r15 的 P 实际池 549,952，只能容纳 2 条同长请求，所以**端到端约 2，而不是 16**。配置写了 `context_length=1,048,576`，也不能越过 P 侧限制。
- 后续 PD1M 配方将 P/D 都调到 1.2M 逻辑量级，D 每 rank 600,000、DCP2 逻辑 1,200,000，才有 `min(P 1.2M, D 1.2M, context 1.048M)` 的 1M 容量条件；它是另一套配置，不与 r15 混合。

**交叉校验：** 另一个单节点 TP8/DCP4 探针报告 target 物理行 `243,712`、逻辑 `974,848`、KV `14.1342344`（约 GiB）。按同一公式：

```text
main  = (243712+64)×78×656 = 12,473,530,368 B
index = 15,236×64×21×132   =  2,702,988,288 B
total = 15,176,518,656 B = 14.134234 GiB
```

与保存的运行时字段吻合，说明 target pool 的分解口径可复核；这不是本次重新实测。

### 6.8 自动容量与显式 `max-total-tokens`

若未给显式上限，可把可用 KV 预算写成：

```text
B = HBM_total - W_local - A_activation - A_graph - A_comm - A_slack - A_other
N ≈ floor_page(B / bytes_per_target_row)
```

GLM/Hy 的 `bytes_per_target_row` 必须包括主 KV、index-K、DCP 和 draft 修正，而不是只用 `2×Hkv×dh×e`。`mem_fraction_static` 影响自动测算的 slack；但一旦显式设置 `max_total_tokens`，降低 mem fraction 不会自动缩小或扩大固定池。Hy 的 OOM 调查已用相同 KV size 的 mem90/mem85 日志证明这一点。

## 07 最终对比表

| 维度 | GLM 系列当前代表配置 | Hy 系列 r15 |
|---|---|---|
| D 侧实现 | TP16 + attnDP4 + attnTP4 + DCP1；DSA + HiSparse host ratio 3 | 同拓扑但 DCP2；定制 SM120 sparse MLA、single-owner sink、fused LSE combiner |
| 请求划分 | 4 个 attention-DP scheduler / KV 池 | 同左 |
| 单请求 target 主 KV | 不按 DCP 分片；hot/cold 由 HiSparse 映射 | 按 token 位置 mod 2 分片，P→D 直接写入 owner |
| 主 KV 副本数（每组） | 4 个 rank（hot/cold 分层另计） | 2 个 rank |
| index-K / draft | full-indexer 21 层；EAGLE draft 不随 DCP | index-K 与 NEXTN draft 仍 4 个 rank，DCP1 |
| 计算 | DP 均衡时核心 attention FLOPs 不因此减少；DCP1 无 LSE 归并 | DCP 理想核心 FLOPs 不减少；Q 复制、partial/LSE、page remap 是增量 |
| 通信 | attnTP + DP gather/combine + 全 TP MoE/scheduler 同步 | 同左，另加 DCP partial output+LSE；Q all-gather 是否省掉取决于 Q 权重复制 |
| 容量 | 每组逻辑 1.44M（host ratio 3）；每组 GPU 主池 480K | 每组 target 逻辑 800K；每 rank 主 KV 400K |
| 单请求 | 公式具备 1M 条件，但完整配方的新 1M 实测回执仍需补齐 | r15 受 P 549,952 限制，不能宣称 1M；PD1M 是另一配方 |
| 适用场景 | 长上下文、host offload 可接受、需要大逻辑池与前缀驻留 | 需要增加 target 主 KV 容量/副本效率，且能接受自定义 kernel 与逐层 LSE 归并成本 |
| 不适用/风险 | 直接叠加 stock DCP 会被 HiSparse/PD guard 拒绝；热集 I/O 与 host 带宽影响 TPOT | 小 batch/短序列、通信或 Q 复制占主导；DCP4 已退化，DCP8 布局不成立；自定义路径需独立正确性验证 |

## 08 仍需通过代码或实测确认

1. **r15 精确 overlay 源码：** 当前工作区保存的 r15 plan 固定了 `dsa_backend.py`、`forward_mla.py`、`layers/dcp/comm.py`、`dcp_kernels.py` 的 SHA；decode 挂载对账为 8 匹配、4 缺失、4 版本不符。`dcp_kernels.py`、`layers/dcp/comm.py`、`arg_groups/model_overrides/deepseek_v2.py`、`disaggregation/common/conn.py` 在本地缺失，另四个文件哈希不一致。因此 fused zero-output-shard sink、SM120 partial-LSE kernel 的精确实现与字节级 payload 只能由 plan、文档和 standalone 日志佐证，不能声称已逐行复核。
2. **r15 resolved 参数：** HYV4 属于 `is_deepseek_dsa`，SM120 在非 `is_sm100` 下将未显式设置的通信后端解析为 `a2a`，而 stock `_dsa_dcp_overrides()` 在该条件下将 `dcp_replicate_q_proj` 置 true；因此**合理推断为 true**。但 r15 未保留 resolved 参数和远程 overlay 字节，仍需运行中 `/get_server_info` 或启动日志作最终确认。
3. **真实通信成本：** 需要在相同 batch、accept length、序列长度、Q 复制设置下采集 NCCL/FlashInfer trace，分别测 Q gather、partial/LSE、DP gather/combine 与 MoE，而不是用张量形状推算出“实测毫秒”。
4. **单因子 A/B：** 至少固定 P 侧、权重、NEXTN accept 分布，比较 DCP1/2、Q replicate on/off、a2a/ag_rs；r6→r15 是组合结果，DCP4 的 accept 也显著变化，不能作单因子性能归因。
5. **并发与长序列容量：** 需分别测每 DP 组非均衡、等长/变长、长输出、prefix reuse 下的驻留与 retract；理论 `4cN` 是均衡上界，不是调度保证。
6. **GLM 完整配方 1M：** 发布记录明确缺新的端到端 1M 回执，需补正确性、输出完整性和性能；不能把历史 D1M 验收直接当成当前完整组合的实测。
7. **总显存：** 还缺每 rank 的 target/draft host pool、权重、activation、workspace、CUDA graph、Mooncake/IPC、radix metadata 的现场账目，不能从 KV payload 推出完整 HBM 余量。
8. **基线可运行性：** TP16、attnDP1 只用于说明并行语义和公式。当前 4-head sparse-MLA kernel 不可运行；若要测“去 DP”的真实收益，必须换支持该 head 形状的 kernel，而不是直接改一个启动参数。
9. **实时在线状态：** 本次没有连接服务器检查容器。GLM 最新 Responses release 只有 CPU 验收，Hy r15 也是历史 plan/日志；不能把任何一份配置称为此刻仍在运行的唯一线上状态。

## 附：主要源码锚点

- `sglang/python/sglang/srt/arg_groups/fields/parallel.py`：并行参数与默认值。
- `sglang/python/sglang/srt/runtime_context.py`：`derive_attn_tp_size`、`derive_attention_ranks`、DCP rank 与 logical capacity 的拓扑基础。
- `sglang/python/sglang/srt/distributed/parallel_state.py: initialize_model_parallel()`：TP / attention TP / DCP group 构造。
- `sglang/python/sglang/srt/layers/dp_attention.py`：DP gather/scatter、padding 与 collective 选择。
- `sglang/python/sglang/srt/managers/data_parallel_controller.py`：scheduler 进程与请求路由。
- `sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py: MLPSyncBatchInfo`：跨 DP 组的 metadata 同步。
- `sglang/python/sglang/srt/models/deepseek_v2.py`：attn-TP projection、`dcp_size`、NextN DCP1。
- `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`：Q gather、partial attention 与 LSE 归并入口。
- `sglang/python/sglang/srt/layers/dcp/{layout,comm,planner}.py`：owner rule、Q/KV gather、LSE combine 与 decode metadata。
- `sglang/python/sglang/srt/layers/attention/dsa/dsa_dcp.py`：DCP page table 与 sparse attention。
- `sglang/python/sglang/srt/mem_cache/{memory_pool,kv_cache_configurator,index_key_cache}.py`：MLA/DSA buffer、DCP 容量、index-K 与 draft。
- `sglang/python/sglang/srt/mem_cache/{hisparse_memory_pool,allocator/hisparse,pool_host/mla}.py`：GLM HiSparse logical/device/host 映射。
- `sglang/python/sglang/srt/disaggregation/{common/utils,mooncake/conn,common/conn}.py`：P→D DCP relayout、CP 条件与 rank mapping。
