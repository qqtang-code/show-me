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

每条目的 `meta.json` 用 `visibility` 标记（`public` / `internal`）与 `redactions[]` 记录脱敏了哪些东西。

## 目录约定

```
show-me/
├── build_index.py             # 扫 meta.json → 重建 README 表格与门户 index.html
├── build_index.py             # 扫 meta.json → 重建 README 表格与门户 index.html
├── check_links.py             # 校验所有页面的相对链接是否可达（sync.sh 会自动跑）
├── sync.sh                    # commit + push（skill 每次写完自动调用）
├── <category>/<slug>/
│   ├── index.html             # 产物本体（固定文件名）
│   ├── notes.md               # 纯文字回答也要留档
│   ├── meta.json              # 元数据（见下）
│   └── *.png                  # 配图等附加文件
```

分类固定为：`inference-systems` 推理系统 · `comm-parallel` 通信与并行 · `training-algorithms` 训练与算法 · `tooling` 工具与方法 · `misc` 其他。

`meta.json` 字段：`title` `slug` `category` `date` `tags[]` `one_liner` `artifact` `extra_files[]` `session_ref` `sources[]` `visibility` `redactions[]` `enriched`。

## 条目总览

<!-- ENTRIES:BEGIN -->
| 日期 | 主题 | 分类 | 标签 | 一句话 | 出处 |
|---|---|---|---|---|---|
| 2026-09-19 | [KV 预取与预测：该不该提前搬，以及提前搬什么](inference-systems/kv-prefetch-prediction/index.html) | inference-systems | KV Cache · 预取 · 稀疏注意力 · DSA · 投机解码 · 评测方法 · HiCache | 把 KV 预取拆成两个问题——该不该提前搬（系统层四类信号）与提前搬什么（模型层预测），并给出判据「提前量 ≥ 传输时间」、五个必报指标与一套跨并发梯度的测试矩阵；模型层部分拆解 SparDA（层间 lookahead，提前量一层）与 DualDecoder（用投机 token 预测下一 decode step 的检索索引，KV 预测准确率 88%）。 | `sess_028051f7` |
| 2026-09-19 | [SGLang 推理全流程：一条请求从进到出的每一步](inference-systems/sglang-request-lifecycle/index.html) | inference-systems | SGLang · 调度 · DSA 稀疏注意力 · MoE · NVFP4 · 投机解码 · CUDA Graph · 量化 AllReduce · 连续批处理 | 以一台 8 卡 PCIe 单机的 GLM 系列 MoE 部署（TP8 · DSA · NVFP4 · EAGLE · 270K 上下文）为例，把一条请求从 HTTP 接入到流式输出的五段流程拆开讲：调度器每轮怎么组批、78 层里到底算了什么、每轮上百次 all-reduce 怎么被压下去、容量与延迟各由什么决定。 | `sess_028051f7` |
| 2026-09-18 | [L1–L4 缓存与 LLM 推理的 KV 分层：HiCache 与 Mooncake 在 SGLang 里各管什么](inference-systems/hicache-mooncake/index.html) | inference-systems | KV Cache · HiCache · Mooncake · SGLang · 分层缓存 · 前缀复用 · PD 分离 | 用存储层级的容量/带宽阶梯解释 decode 为什么是纯访存瓶颈；SGLang HiCache 把 radix 树从 HBM 延伸到主机与集群，Mooncake 是其中一种 L3 实现（同时还是 PD 分离的默认传输后端），二者是策略层与数据面的正交关系。 | `sess_028051f7` |
| 2026-09-16 | [PP 适配 DFlash：伪造单例 PP 组 + 两条跨阶段数据流](inference-systems/dflash-pp-adaptation/index.html) ([notes.md](inference-systems/dflash-pp-adaptation/notes.md)) | inference-systems | SGLang · 流水并行 · DFlash · 投机解码 · 跨阶段通信 · 正确性 bug | 上游 SGLang 有三道护栏硬性禁止 DFlash 与 PP 共存；适配没有碰算法，而是用一个 world_size=1 的伪造 PP 组让草稿模型整体跑在 PP0，并设计 aux hidden 去程累积、投影 context 回程两条通道穿过 PP ring。 | `sess_0255fd71` |
| 2026-09-20 | [CUDA IPC 直推：decode 小消息换的是「固定开销」，不是带宽](comm-parallel/cuda-ipc-direct-push/index.html) | comm-parallel | AllReduce · CUDA IPC · PCIe · 张量并行 · 通信优化 · 延迟vs带宽 | decode 小消息把 AllReduce 换成 CUDA IPC 直推，砍的是成本式里的固定开销 α（不是带宽）；1-stage ↔ 2-stage 是同一套 IPC 内核里的 Variant 选择（seed 交叉点 32 KiB），按载荷尺寸分流——两条改动的收益不能相乘。 | `sess_f72fea5d` |
| 2026-09-15 | [1-stage → 2-stage AllReduce：为什么省下的是 4 倍字节](comm-parallel/allreduce-1stage-vs-2stage/index.html) ([allreduce-infographic.png](comm-parallel/allreduce-1stage-vs-2stage/allreduce-infographic.png)) | comm-parallel | AllReduce · 张量并行 · 通信优化 · PCIe · 投机解码 · Amdahl | 把 AllReduce 从广播式改成 reduce-scatter + all-gather，单次归约 907 → 222.6 µs；收益全部来自「少搬 4 倍字节」，端到端 +60% 对 Amdahl 上限 +76% 的兑现率 91%。 | `sess_0255fd71` |
<!-- ENTRIES:END -->

## 维护

```bash
open index.html                 # 打开门户（搜索 + 标签筛选）
python3 build_index.py          # 新增/修改 meta.json 后重建索引
python3 check_links.py          # 链接体检（sync.sh 里也会自动跑）
bash sync.sh "show-me: <主题>"   # 提交并推送（幂等；无改动时不产生空提交）
```

## 视觉约定

三篇算一个站点，共用一套设计变量（每页 `<style>` 顶部那份 `:root` 保持一致，页面仍各自自包含）：

- 底色 `#f5f7fb`、卡片白、边框 `#dde3ee`、文字 `#131c2e` / 次要 `#4a5871`
- 主色 `#1d4ed8`（链接、速览卡、导航 chip 悬停）
- 正文 15.5px / 行高 1.72，内容宽 980px
- 页面骨架：面包屑 → 标题 → 日期与标签 pill → 锚点导航 → 速览卡 → 正文 → 页脚（数据来源 + 许可）
- 交互计算器：按「载荷 / 换算 / 结论」这类逻辑分组，每组 2–3 格，避免出现单独一张的孤儿卡
- 移动端：导航 chip 横向滚动，计算器的对照组自动折成一列

## 许可

内容（`*.html` / `*.md` / 图片）以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 共享，脚本与代码以 MIT 共享。引用时请注明出处并链接回本仓库。