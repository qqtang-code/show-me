# 🧠 show-me 知识库 · 图解与概念解释合集

每次用 `show-me` skill 把概念讲清楚时产出的图解、推导与解释，都归档在这里。一个主题一个目录，`index.html` 是产物本体。

- **定位**：和 [Paper-Reading-Collection](https://qqtang-code.github.io/Paper-Reading-Collection/) 并列——那边是读别人的论文，这边是讲清楚一个概念、一次改动或一个系统设计。
- **在线门户**：[qqtang-code.github.io/show-me](https://qqtang-code.github.io/show-me/)（本地直接 `open index.html` 也有搜索与标签筛选）。
- **每篇的结构**：速览 → 前置知识 → 关键推导（带真实数字）→ 常见误解 → 自测题 → 延伸阅读。自测题用 `<details>` 折叠，答案就在下面。

## ⚠️ 公开仓库的脱敏约定

这个仓库是**公开**的，并按学习资料来写。归档前必须去掉这些内容：

- 内部**主机名 / IP / 绝对路径 / 内部目录名**
- 内部**文档链接与文档 ID**、内部代码库路径
- **未公开的模型版本号**（可写「GLM 系列模型」这类家族描述）与内部项目代号
- 内部**环境变量名**、内部补丁编号（改写成中性描述，如 `DFLASH_PP_INIT_PROBE`）
- 会话/任务 ID（`meta.json` 里只保留 `sess_xxxxxxxx` 前 12 位作为 `session_ref`，便于本地回溯）

**保留**：全部实测数字、推导过程、代码结构、结论与代价分析——学习价值恰恰在这里。

每条目的 `meta.json` 用两个字段分开记：
`visibility`（`public` / `internal`，描述**原始材料**的敏感度）与 `publish`（`true` / `false`，描述**这一页是否已批准公开**）。
`publish: false` 的条目不会进门户、不进 README、不参与上一篇/下一篇。`redactions[]` 记录脱敏了哪些东西。

构建期还会兜底：`cwd` 若写成绝对路径会被改写为仓库相对路径（仓库外则清空），
其余字段里若还留着本机绝对路径或完整 session id，`build_index.py` 直接中止 —— 这类内容推上去就进 git 历史了，宁可卡在本地。

## 目录约定

```
show-me/
├── build_index.py             # 扫 meta.json + 解析文章页 → 重建门户、README 表格、各页站点外壳
├── theme.py                   # 全站设计令牌（颜色/尺寸的唯一来源，门户与文章页外壳共用）
├── shell.py                   # 给每个文章页注入站点外壳（吸顶条 / 分类侧栏 / 上一篇下一篇），幂等
├── templates/portal.html      # 门户模板（占位符由 build_index.py 填充，别改根目录的 index.html）
├── curation.json              # 阅读路线（routes，手工维护）与「确认不进路线」清单（unlisted）
├── check_links.py             # 校验相对链接**与页内锚点**是否可达（sync.sh 会自动跑，失败即中止）
├── taxonomy.json              # 标签受控词表：同义写法在构建期归并到规范写法
├── verify_site.py             # 验收：链接 / 六档宽度溢出 / 门户交互 / 外壳幂等
├── sync.sh                    # commit + push（skill 每次写完自动调用）
├── DESIGN.md                  # 站点结构与信息架构说明（改版时先看这份）
├── <category>/<slug>/
│   ├── index.html             # 产物本体（固定文件名）
│   ├── notes.md               # 纯文字回答也要留档
│   ├── meta.json              # 元数据（见下）
│   └── *.png                  # 配图等附加文件
```

分类固定为：`inference-systems` 推理系统 · `comm-parallel` 通信与并行 · `training-algorithms` 训练与算法 · `tooling` 工具与方法 · `misc` 其他。

`meta.json` 字段：`title` `slug` `category` `date` `tags[]` `one_liner` `artifact` `extra_files[]` `session_ref` `sources[]` `visibility` `publish` `redactions[]` `enriched`。

`one_liner` 是卡片摘要，控制在 240 字符以内（当前 119–234，均值 186）：卡片负责帮读者**选择**，完整论证留在正文。

## 条目总览

<!-- ENTRIES:BEGIN -->
| 日期 | 主题 | 分类 | 标签 | 一句话 | 出处 |
|---|---|---|---|---|---|
| 2026-11-16 | [把上下文做到 1M：DCP、DPAttention 与 HiSparse 的三种切法](inference-systems/decode-dp-attention-dcp/index.html) ([notes.md](inference-systems/decode-dp-attention-dcp/notes.md)) | inference-systems | DCP · DPAttention · HiSparse · 1M 上下文 · KV Cache · MLA · DSA · 容量模型 | 单请求 1M 上下文要求逻辑 target 位置 ≥ 1,048,576，三条路乘的对象不同：DCP 乘位置分片宽度（TP8×DCP4，269,952×4=1,079,808，纯 GPU）、DPAttention 拆成 4 个独立池（每池 480,000×host3 或 600,000×DCP2）、HiSparse 乘 host/device 倍数（TP8，132,096×16=2,113,536）。index-K 不随 DCP 分片，显存收益小于主 KV ÷c。 | — |
| 2026-11-16 | [SGLang 的 LayerSplit：把 DSA 的 KV/indexer cache 按层摊到 prefill CP 的各 rank](inference-systems/sglang-layer-split/index.html) ([notes.md](inference-systems/sglang-layer-split/notes.md)) | inference-systems | SGLang · LayerSplit · DSA · prefill CP · PD 分离 · KV Cache · MLA · 并行策略 | LayerSplit 不新增并行维度，也不改任何 kernel 的计算形状 —— 它只挪动「第 l 层的 KV 落在哪张卡上」。每 rank 只 materialize ceil(L/S) 层加 1 层 remote scratch，非 owner 层靠 owner 广播整层 buffer 来读（搬的是整个 token pool，不是本批次 page）。只对 PD prefill worker 生效，要求 interleave CP、PP=1、mooncake 后端；S=4 时省 72%，同显存下 token 容量约 ×3.6。附层归属公式、五道门控的原文错误串、以及「PyNcclCommunicator 默认 disabled 导致 broadcast 静默 no-op」这个坑。 | — |
| 2026-10-27 | [SGLang prefix cache 的三类注意力：Linear · SWA · Sparse](inference-systems/sglang-prefix-cache-three-attentions/index.html) ([notes.md](inference-systems/sglang-prefix-cache-three-attentions/notes.md)) | inference-systems | SGLang · prefix cache · Unified Radix Cache · 线性注意力 · KDA · 滑动窗口注意力 · DSA · tombstone · GLM 系列 | 同一棵 radix 树上，线性注意力、滑动窗口、稀疏注意力各自缓存什么、怎么判定命中、怎么更新、在什么场景下会失效：MAMBA 存一个 slot id（O(1)/序列）、SWA 的复用是尾窗全有或全无（5000-token 前缀在 L<4999 时复用 0）、稀疏注意力缓存的是 indexer K 而不是 top-k 模式。附 GLM5 系列模型的完整推理路径。 | — |
| 2026-10-21 | [DeepInfer × SGLang 深度对比：同一条请求，两种引擎](inference-systems/deepinfer-vs-sglang/index.html) | inference-systems | DeepInfer · SGLang · 调度 · KV Cache · DSpark · 投机解码 · MoE · 并行策略 · 源码走读 | 把 DeepInfer 与 SGLang 放进同一条请求里对照：进程内 C++ 调度器 + 单模型离线内核，对 Python 调度器 + ZMQ 三进程全栈服务。异步调度都靠「结果延后一步」，但一边把 seqlens 修正搬上 GPU；前缀复用一个靠 radix 自动命中、一个靠 fork 显式共享加外部 3FS；投机解码一边在线预测验证长度，一边用离线 SPS/STS 表。 | — |
| 2026-10-21 | [SGLang 并行策略怎样改变 KV 池容量：分子 B 与分母 b](inference-systems/kv-pool-parallel-strategies/index.html) ([notes.md](inference-systems/kv-pool-parallel-strategies/notes.md)) | inference-systems | SGLang · KV Cache · TP / PP / CP / DCP / EP / DP · MLA · DSA · GLM 系列 · 容量模型 · Attention DP | 把 KV 池容量写成 B/b 之后，每条并行策略只需回答它动了分子还是分母：TP 只靠权重分片放大 B，PP 把本地层数 78→39 使 8 卡容量翻倍，CP 挤掉 attention TP 会吃掉 B，EP 在固定 TP 组宽时只换存法，Attention DP 把 8 份重复 latent 换成 8 个独立池（唯一 token 容量约 N→8N）；而 PP 多 stage 取 MIN，分层不均衡会吃掉整个实例。 | — |
| 2026-09-24 | [--stream-interval 为什么在长输出场景有提升](inference-systems/sglang-stream-interval/index.html) | inference-systems | SGLang · 流式输出 · SSE · 调度 · O(T²) · 推理服务 | --stream-interval 在源码里只有一个判定点（output_streamer.accept 的 len(output_ids) % N == 1），但真正被它砍掉的是默认非增量流式路径里的 O(T²)：每个 SSE 事件都要重建累计文本、并携带完整 text + output_ids 向下游搬运；实测每翻倍输出长度开销涨 3.9 倍，间隔 2 恒定砍半 —— 所以只有长输出才看得见。 | — |
| 2026-09-21 | [MoE 的通信：MegaMoE 内核、A2A 开关，与各后端的真实差别](inference-systems/moe-a2a-megamoe/index.html) | inference-systems | SGLang · DeepGEMM · MegaMoE · MoE · EP / A2A · DeepEP · FlashInfer · 对称内存 · 源码走读 | MegaMoE 把一个 MoE 层的 dispatch、两次分组 GEMM、激活与 combine 归约压进同一个全网格持久内核（拉取式 dispatch + L1 输出直落 L2 池 + 结果直推源 rank）；设与不设 --moe-a2a-backend 之间，换的是「层尾那次稠密 all-reduce」与「dispatch/combine」的取舍，并顺带改写 ep=tp、共享专家融合、CUDA graph 等七项参数。 | — |
| 2026-09-19 | [KV 预取与预测：该不该提前搬，以及提前搬什么](inference-systems/kv-prefetch-prediction/index.html) | inference-systems | KV Cache · 预取 · 稀疏注意力 · DSA · 投机解码 · 评测方法 · HiCache | 把 KV 预取拆成两个问题——该不该提前搬（系统层四类信号）与提前搬什么（模型层预测），并给出判据「提前量 ≥ 传输时间」、五个必报指标与一套跨并发梯度的测试矩阵；模型层部分拆解 SparDA（层间 lookahead，提前量一层）与 DualDecoder（用投机 token 预测下一 decode step 的检索索引，KV 预测准确率 88%）。 | `sess_028051f7` |
| 2026-09-19 | [SGLang 推理全流程：一条请求从进到出的每一步](inference-systems/sglang-request-lifecycle/index.html) | inference-systems | SGLang · 调度 · DSA · MoE · NVFP4 · 投机解码 · CUDA Graph · 量化 AllReduce · 连续批处理 | 以一台 8 卡 PCIe 单机的 GLM 系列 MoE 部署（TP8 · DSA · NVFP4 · EAGLE · 270K 上下文）为例，把一条请求从 HTTP 接入到流式输出的五段流程拆开讲：调度器每轮怎么组批、78 层里到底算了什么、每轮上百次 all-reduce 怎么被压下去、容量与延迟各由什么决定。 | `sess_028051f7` |
| 2026-09-18 | [L1–L4 缓存与 LLM 推理的 KV 分层：HiCache 与 Mooncake 在 SGLang 里各管什么](inference-systems/hicache-mooncake/index.html) | inference-systems | KV Cache · HiCache · Mooncake · SGLang · 分层缓存 · prefix cache · PD 分离 | 用存储层级的容量/带宽阶梯解释 decode 为什么是纯访存瓶颈；SGLang HiCache 把 radix 树从 HBM 延伸到主机与集群，Mooncake 是其中一种 L3 实现（同时还是 PD 分离的默认传输后端），二者是策略层与数据面的正交关系。 | `sess_028051f7` |
| 2026-09-16 | [PP 适配 DFlash：伪造单例 PP 组 + 两条跨阶段数据流](inference-systems/dflash-pp-adaptation/index.html) ([notes.md](inference-systems/dflash-pp-adaptation/notes.md)) | inference-systems | SGLang · 流水并行 · DFlash · 投机解码 · 跨阶段通信 · 正确性 bug | 上游 SGLang 有三道护栏硬性禁止 DFlash 与 PP 共存；适配没有碰算法，而是用一个 world_size=1 的伪造 PP 组让草稿模型整体跑在 PP0，并设计 aux hidden 去程累积、投影 context 回程两条通道穿过 PP ring。 | `sess_0255fd71` |
| 2026-11-16 | [CP 边界的 FP8 通信：AllGather 先量化、ReduceScatter 把求和移出 wire](comm-parallel/cp-boundary-fp8-comm/index.html) | comm-parallel | Context Parallel · 量化通信 · AllGather · ReduceScatter · FP8 · PCIe · Prefill · SGLang | CP 边界每层搬两次残差流（MoE 前 AllGather、MoE 后 RS），BF16 大块、正落在 PCIe 的带宽区间。AG 把量化折进 gather 的读地址、RS 把求和从 wire 挪到本地 FP32 累加，线上字节降到 50.2%（scale 仅占 0.39%）；代价是数值路径改变、行数 <256 回退 BF16、block 必须整除 4096。实测端到端 +8.39%~+10.48%，且 RS 那两档是「在 AG 已开之上」的增量，不能相加。 | — |
| 2026-11-16 | [P 端 CP4 的量化通信三件套：MoE 先量化再 AllGather、quant-RS、CP KV 的 FP8 gather](comm-parallel/prefill-cp4-quantized-comm/index.html) | comm-parallel | PD 分离 · Context Parallel · 量化通信 · AllGather · ReduceScatter · FP4 · FP8 · Prefill | CP4 的 P 端一层有五处跨卡搬运，归四类机制：MoE 输入先压成 4 bit 再 gather、MoE 输出的 RS 改成 FP8 wire + 本地 FP32 累加（<1024 行必须回退，否则 2K/16K 并发掉档）、KV gather 复用已有 FP8 字节（零转换，四条守卫要同时对上）、attention 边界同款按行量化。组合后 16K/384 冷前缀 TPS +29.5%；attention 边界非无损（rel-L2 2–3%）。 | — |
| 2026-11-04 | [D 端 TP16 分层通信：把 16 卡集合通信拆成「跨机配对 + 机内 8 卡」](comm-parallel/decode-tp16-2level-comm/index.html) | comm-parallel | 集合通信 · AllGather · ReduceScatter · TP16 · 分层通信 · PD 分离 · 投机解码 · SLO | decode 端 TP16 必然横跨两台机，每层要跨机同步两次、每次只搬 12–48 KB。把全组 AllGather／ReduceScatter 拆成「同位置 rank 跨机配对 1 轮 + 机内 8 卡 7 轮」，串行轮数 15→8：每验证轮通信 37.79→29.44 ms 与 34.45→20.77 ms，TPOT 21.41→18.67 ms，达标并发 <5→8；代价是短输入档 TTFT P50 +65%／+20%，900 秒会话回放只剩 +2.26%。 | — |
| 2026-11-02 | [--tp 8 --ep 1/4/8 的切分与通信：SGLang v0.5.23 里的 Mixtral-8x7B](comm-parallel/sglang-tp8-ep1-4-8-mixtral/index.html) ([notes.md](comm-parallel/sglang-tp8-ep1-4-8-mixtral/notes.md)) | comm-parallel | SGLang · MoE · EP · TP8 · AllReduce · all2all · Mixtral · 并行组 · moe_tp · 源码走读 | 在 SGLang v0.5.23 的默认 backend（moe_a2a_backend=none）下，--tp 8 --ep 1/4/8 的 MoE 段通信完全相同：同一个 _TP(8) 组上的一次 all-reduce，载荷 [T,4096] BF16，约 0.94 MiB/token。ep 只改权重排布；引入 dispatch/combine all-to-all 的是 deepep 等后端，它们把 ep 强制改成 tp，三种写法收敛到 ep=8。 | — |
| 2026-10-21 | [三种集合通信的落点：all2all、allreduce、allgather 与 GLM-5.3 的一条 forward](comm-parallel/a2a-ar-ag-sglang/index.html) | comm-parallel | SGLang · 集合通信 · all2all · AllReduce · AllGather · p2p · GLM 系列 · 张量/专家/流水并行 · 源码走读 | 两个判据分开 all2all / allreduce / allgather：输出在各 rank 上是否相同、是否做算术 —— 字节数相同不等于可互换。再沿 SGLang 的五层调用链走到 PP 点对点（send_tensor_dict 只发 1/P 切片，每跨一条 stage 边界多一次 allgather）。GLM-5.3 的 H200 配方按调用点计是 91 次 all-reduce + 1 次 all-gather + 0 次 all2all。 | — |
| 2026-09-22 | [注意力 AllReduce 与 MoE 计算重叠：改的是一层内部的指令排布](comm-parallel/ar-compute-overlap-moe-chunk/index.html) ([notes.md](comm-parallel/ar-compute-overlap-moe-chunk/notes.md)) | comm-parallel | AllReduce · 通信计算重叠 · 张量并行 · MoE · 流水并行 · 预填充 · PCIe | 把 MoE 前的那次全量 AllReduce 切成两片扔到侧流，主流拿第一片就先算：藏住 1.77 ms、付出 0.64+0.35 ms，净赚 0.78 ms/层（实测整层 −0.79，逐位吻合）；同容器 A/B 端到端 +2.8%/6.8%/7.7%（C=8/16/32），但同一补丁在 TP2×PP4 上净 ≈ 0 —— 收益正比于「被暴露的通信时长」，不是模型规模。 | — |
| 2026-09-22 | [Prefill CP8 / Decode TP8：为什么 CP 能拿到收益](comm-parallel/prefill-cp8-decode-tp8/index.html) | comm-parallel | Context Parallel · Prefill · TP8 · Sparse MLA · mHC · 超连接 · MoE · AllGather · ReduceScatter · SGLang | CP 的收益不来自注意力 FLOPs（TP8 与 CP8 每卡 query×head 都是 65,536），而来自把每卡行数从 8192 降到 1024：TP 切不动的按 token 重复的算子随之掉 8 倍（mHC 136.77→13.06 ms），GPU 窗口 811.55→533.70 ms，输入吞吐 +27.1%、TTFT −34.5%、TPOT −15.8%；代价是 tp=1 的权重复制（静态显存 0.68→0.74）。 | `sess_30d19c4e` |
| 2026-09-20 | [CUDA IPC 直推：decode 小消息换的是「固定开销」，不是带宽](comm-parallel/cuda-ipc-direct-push/index.html) | comm-parallel | AllReduce · CUDA IPC · PCIe · 张量并行 · 通信优化 · 延迟vs带宽 | decode 小消息把 AllReduce 换成 CUDA IPC 直推，砍的是成本式里的固定开销 α（不是带宽）；1-stage ↔ 2-stage 是同一套 IPC 内核里的 Variant 选择（seed 交叉点 32 KiB），按载荷尺寸分流——两条改动的收益不能相乘。 | `sess_f72fea5d` |
| 2026-09-15 | [1-stage → 2-stage AllReduce：为什么省下的是 4 倍字节](comm-parallel/allreduce-1stage-vs-2stage/index.html) ([allreduce-infographic.png](comm-parallel/allreduce-1stage-vs-2stage/allreduce-infographic.png)) | comm-parallel | AllReduce · 张量并行 · 通信优化 · PCIe · 投机解码 · Amdahl | 把 AllReduce 从广播式改成 reduce-scatter + all-gather，单次归约 907 → 222.6 µs；收益全部来自「少搬 4 倍字节」，端到端 +60% 对 Amdahl 上限 +76% 的兑现率 91%。 | `sess_0255fd71` |
| 2026-11-16 | [show-me 知识库体检：改了什么，以及为什么这么改](tooling/show-me-knowledge-base-review/index.html) ([notes.md](tooling/show-me-knowledge-base-review/notes.md)) | tooling | 信息架构 · 全库搜索 · 发布检查 · 脱敏 · 标签治理 · 阅读路线 · 站点验收 | 对 18 篇条目做的实测审查与随后的落地：先修失效的分类锚点、模糊的 visibility/publish 语义与含本机路径的公开元数据，再把搜索改成别名归一 + 词边界 + 打分排序并按文章分组，最后收敛标签、区分路线与合集、让「已打开」在文章页也记账。每条都附了验收判据和实测数字。 | — |
| 2026-10-13 | [torchrun 启动流程与进程管理：从一条命令到 N 个训练进程](tooling/torchrun-launch-internals/index.html) ([notes.md](tooling/torchrun-launch-internals/notes.md)) | tooling | torchrun · torchelastic · rendezvous · TCPStore · DDP · 分布式训练 · 源码走读 · 容错重启 | 沿 torch.distributed.run → elastic_launch → LocalElasticAgent 拆开 torchrun：第 2 步就把单节点命令改为 c10d + localhost:0；RANK 不是参数算出来的，而是 rendezvous 分配的结果（RANK = GROUP_RANK × LOCAL_WORLD_SIZE + LOCAL_RANK）；注入给 worker 的是环境变量字典而非命令行参数。 | — |
<!-- ENTRIES:END -->

## 维护

```bash
open index.html                 # 打开门户（全库检索 ⌘K + 分类 + 标签筛选 + 阅读路线）
python3 build_index.py          # 新增/修改 meta.json 或文章页后重建（门户 + 外壳 + README 表格）
python3 build_index.py --dry-run    # 只看解析结果：每篇多少分钟、多少小节、路线覆盖几篇
python3 verify_site.py          # 验收：链接、六档宽度、门户交互、外壳幂等
python3 check_links.py          # 链接与锚点体检（sync.sh 里也会自动跑，失败不提交）
bash sync.sh "show-me: <主题>"   # 提交并推送（幂等；无改动时不产生空提交）
```

新增条目后要决定它进哪条路线：`build_index.py` 会对「既不在 `routes` 也没写进 `unlisted`」的条目告警，
漏收录不会再悄悄发生（这个清单是手工的，没有任何自动收录规则）。

**别手改根目录的 `index.html`** —— 它由 `templates/portal.html` 渲染生成，下次构建会被覆盖。
要改门户就去改模板。文章页里的站点外壳同样由 `shell.py` 生成，用
`<!-- show-me:shell:begin -->…<!-- show-me:shell:end -->` 标出范围，构建时整段重写；
标记之外的手写正文不受影响。

## 视觉约定

所有页面共用一套设计变量，来源是 `theme.py` 的 `TOKENS_CSS`（门户渲染时注入，文章页外壳注入时内联）。
每页仍然**自包含**——单个 `index.html` 可以直接发出去，不依赖外部 CSS/JS/字体。

- 底色 `#f5f7fb`、卡片白、边框 `#dde3ee`、文字 `#131c2e` / 次要 `#4a5871`
- 主色 `#1d4ed8`（链接、速览卡、导航 chip 悬停）
- 分类色：推理系统 `#1d4ed8` / 通信与并行 `#0b6b3f` / 训练与算法 `#6b2fb5` / 工具与方法 `#a35a06` / 其他 `#5b6b82`——门户的分类点、文章页的分类 pill、阅读路线统一取色
- 正文 15.5px / 行高 1.72，内容宽 980px
- 页面骨架：面包屑 → 标题 → 日期与标签 pill → 锚点导航 → 速览卡 → 正文 → 页脚（数据来源 + 许可）；
  外壳再叠一层：吸顶条（品牌 / 分类 / 阅读时长 / 搜索）+ 宽屏左侧同分类侧栏 + 页尾上一篇/下一篇
- 交互计算器：按「载荷 / 换算 / 结论」这类逻辑分组，每组 2–3 格，避免出现单独一张的孤儿卡
- 移动端：导航 chip 横向滚动，计算器的对照组自动折成一列；≥1280px 才出现左侧分类侧栏，正文宽度不随之变化
- 数据图表：不手写，由 [`lieflat-charts`](https://github.com/larashero3-dotcom/lieflat-charts) skill（模板驱动）生成，`catalog.md` 选型、`templates/` 取骨架。页壳仍用上面这套设计变量，图内排版与最小字号听 lieflat 的；整页只锁一套色系（默认 Mono 灰阶，有序单序列可用 porcelain；用 palm / wire 时把 `--accent` 一起换掉，不留两种主色）

## 许可

内容（`*.html` / `*.md` / 图片）以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 共享，脚本与代码以 MIT 共享。引用时请注明出处并链接回本仓库。

例外：页面里的数据图表由 [lieflat-charts](https://github.com/larashero3-dotcom/lieflat-charts) 的模板与 token 生成，该部分代码遵循 **PolyForm Noncommercial License 1.0.0**，不属于上面的 CC BY 4.0 / MIT——可非商业使用与再分发，但不得用于商业用途；转发带图表的页面时请保留本声明。页面文字的引用仍按 CC BY 4.0。