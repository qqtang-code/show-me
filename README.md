# 🧠 show-me 知识库 · 图解与概念解释合集

每次用 [`show-me`](/Users/qqtang/.agents/skills/show-me/SKILL.md) 把概念讲清楚时产出的图解、示意图与解释，都归档在这里。一个主题一个目录，`:index.html` 是产物本体。

- **定位**：和 [Paper-Reading-Collection](../Paper-Reading-Collection/) 并列——那边是读别人的论文，这边是讲清楚一个概念/一次改动。
- **⚠️ 本仓库私有**：部分条目含未公开的工程细节与内部实验数据（GLM-5.3 性能数字、内部环境变量与代码路径），**不要公开或外发**。每条目的 `meta.json` 里有 `visibility` 字段：`internal` 表示不可公开，`public-ok` 表示可以。
- **门户**：本地直接打开 `index.html`（含搜索与标签筛选）。

## 目录约定

```
show-me/
├── build_index.py             # 扫 meta.json → 重建 README 表格与门户 index.html
├── sync.sh                    # commit + push（skill 每次写完自动调用）
├── <category>/<slug>/
│   ├── index.html             # 产物本体（固定文件名）
│   ├── notes.md               # 纯文字回答也要留档
│   ├── meta.json              # 元数据（见下）
│   └── *.png                  # 配图等附加文件
```

分类固定为：`inference-systems` 推理系统 · `comm-parallel` 通信与并行 · `training-algorithms` 训练与算法 · `tooling` 工具与方法 · `misc` 其他。

`meta.json` 字段：`title` `slug` `category` `date` `tags[]` `one_liner` `artifact` `extra_files[]` `session_id` `cwd` `sources[]` `visibility`。

## 条目总览

<!-- ENTRIES:BEGIN -->
| 日期 | 主题 | 分类 | 标签 | 一句话 | 出处 |
|---|---|---|---|---|---|
| 2026-09-16 | [PP 适配 DFlash：伪造单例 PP 组 + 两条跨阶段数据流](inference-systems/dflash-pp-adaptation/index.html) ([notes.md](inference-systems/dflash-pp-adaptation/notes.md)) | inference-systems | SGLang · 流水并行 · DFlash · 投机解码 · 跨阶段通信 · 正确性 bug | 上游 SGLang 有三道护栏硬性禁止 DFlash 与 PP 共存；适配没有碰算法，而是用一个 world_size=1 的伪造 PP 组让草稿模型整体跑在 PP0，并设计 aux hidden 去程累积、投影 context 回程两条通道穿过 PP ring。 | `sess_0255fd71-2740-4760-8543-294de23e9fc5` |
| 2026-09-18 | [L1–L4 缓存与 LLM 推理的 KV 分层：HiCache 与 Mooncake 在 SGLang 里各管什么](inference-systems/hicache-mooncake/index.html) | inference-systems | KV Cache · HiCache · Mooncake · SGLang · 分层缓存 · 前缀复用 · PD 分离 | 用存储层级的容量/带宽阶梯解释 decode 为什么是纯访存瓶颈；SGLang HiCache 把 radix 树从 HBM 延伸到主机与集群，Mooncake 是其中一种 L3 实现（同时还是 PD 分离的默认传输后端），二者是策略层与数据面的正交关系。 | `sess_028051f7-061f-4ed4-ae09-2e0f7bd6f263` |
| 2026-09-15 | [1-stage → 2-stage AllReduce：为什么省下的是 4 倍字节](comm-parallel/allreduce-1stage-vs-2stage/index.html) ([allreduce-infographic.png](comm-parallel/allreduce-1stage-vs-2stage/allreduce-infographic.png)) | comm-parallel | AllReduce · 张量并行 · 通信优化 · PCIe · 投机解码 · Amdahl | GLM-5.3 TP8 上把 AllReduce 从广播式改成 reduce-scatter + all-gather，单次归约 907 → 222.6 µs；收益全部来自「少搬 4 倍字节」，端到端 +60% 对 Amdahl 上限 +76% 的兑现率 91%。 | `sess_0255fd71-2740-4760-8543-294de23e9fc5` |
<!-- ENTRIES:END -->

## 维护

```bash
open index.html                 # 打开门户（搜索 + 标签筛选）
python3 build_index.py          # 新增/修改 meta.json 后重建索引
bash sync.sh "show-me: <主题>"   # 提交并推送（幂等；无改动时不产生空提交）
```