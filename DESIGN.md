# show-me 知识库 · 站点重构设计说明

目标：把 show-me 从一个「归档目录」做成一个**易用、易懂、清晰、易查找**的知识库。

参考对象是 [learn-deepinfer](https://learn-deepinfer.github.io/zh)：它把一组知识点组织成一条
可浏览的路径，每篇有统一的页头、编号小节、页内侧栏与页尾翻页。我们**只取它的组织方式与导航方式**，
不搬它的技术栈（React + Vite SPA）——本库首页与文章页必须仍是「单文件自包含、可以直接把一份
`index.html` 发出去」的产物。

---

## 一、取什么、不取什么

| 参考做法 | 本库决定 | 理由 |
|---|---|---|
| 门户有 hero：一句话说明这是什么、怎么用 | **取** | 现在门户只有标题＋统计，第一次来的人不知道这条目怎么读 |
| 分类/层次色编码（layer color） | **取** | 本库已有分类，但要变成一等公民：门户点、文章 pill、阅读路线统一取色 |
| 卡片统一解剖：编号徽标 / 时长 / 标题 / 摘要 / 标签 / 页脚 | **取** | 现在卡片里 `meta.json` 当成内容显示，读者会当成正文链接 |
| 阅读顺序（learning path） | **取，改写** | 本库是主题归档不是课程，所以叫「阅读路线」，按主题编组，且允许多条 |
| 进度条（localStorage） | **取** | 回流读者的刚需；只在门户显示，不上文章页 |
| ⌘K 全库搜索（带片段与键盘导航） | **取** | 当前搜索只看标题/标签/一句话，正文里的机制名搜不到 |
| 文章页 3 栏：侧栏目录 / 正文 / 页内 TOC | **取，改造** | 本库正文已有页内锚点条（`.nav`），再加一层 TOC 就冗余；侧栏改成**同分类条目**，且只放在宽屏空白边栏里，不动 980px 正文宽度 |
| 文章页 4 个 tab（讲/代码/演示/源码） | **不取** | 本库页面是图文交织的长文，拆 tab 反而割裂 |
| 双语 zh/en | **不取** | 内容全是中文，加一套英文壳是纯负担 |
| 深色主题 | **暂不取** | 现有 11 页都是浅色手写样式，加暗色要逐页核对配色；留作后续 |
| React + Vite + 预渲染 | **不取** | 破坏「单文件可单独发送」这一条，且 GitHub Pages 静态托管下收益为零 |

---

## 二、构建管线

```
<category>/<slug>/meta.json  +  index.html
                │
                ├── build_index.py parse  ──► entry dict（标题 / 小节 / 正文纯文本 / 阅读时长）
                │                              ├── templates/portal.html ──► 门户 index.html（内联全库索引）
                │                              └── shell.py ──────────────► 文章页注入站点外壳
                │
                └── verify_site.py  ──► 链接体检 / 390px 溢出探针 / 交互断言
```

一条命令：`python3 build_index.py`。它会重建门户、重建 README 表格、**并且重写每个文章页的 shell 段**。
shell 段用注释标记包裹，重复运行不会叠加：

```html
<!-- show-me:shell:begin -->…<!-- show-me:shell:end -->
```

`sync.sh` 不用改，它仍然先跑 `build_index.py` 再体检再提交。

---

## 三、数据契约

`build_index.py` 产出 `entries: list[dict]`，每个 entry 在 `meta.json` 原字段之外补这些下划线字段：

```python
entry = {
    # ── 来自 meta.json ──
    "title", "slug", "category", "date", "tags", "one_liner",
    "artifact", "extra_files", "session_ref", "sources", "visibility",
    # ── build 时补 ──
    "_dir": "inference-systems/sglang-stream-interval",  # 相对仓库根
    "_root": "../../",                                   # 回到仓库根的相对前缀
    "_depth": 2,
    "_cat_zh": "推理系统",
    "_color": "#1d4ed8",
    "_minutes": 12,                                      # 阅读时长（按正文中文字数/400 每分钟取整，下限 3）
    "_h1": "原文 h1 文本（纯文本）",
    "_lead": "导语段纯文本，≤200 字",                      # 门户卡片在缺 one_liner 时兜底
    "_sections": [{"id": "tldr", "n": "01", "title": "结论", "text": "该节正文纯文本（不截断，最长约 8000 字）"}],
    "_text": "整页纯文本，仅用于构建期检索，不落地",
}
```

### 门户模板占位符（`templates/portal.html`）

| 占位符 | 注入内容 |
|---|---|
| `__TOKENS__` | `theme.TOKENS_CSS` 原样 |
| `__SITE__` | 下面的 SITE JSON（`json.dumps(..., ensure_ascii=False)`） |
| `__PATHS__` | 阅读路线 JSON |
| `__STATS__` | 已渲染好的统计条 HTML |
| `__BUILD__` | 构建日期 `YYYY-MM-DD` |

```jsonc
// SITE: 门户与搜索的唯一数据源，整块内联进 index.html
{"entries": [{
  "dir": "inference-systems/sglang-stream-interval",
  "title": "…", "slug": "…", "date": "2026-09-24",
  "cat": "inference-systems", "catZh": "推理系统", "color": "#1d4ed8",
  "tags": ["SGLang", "…"],
  "one": "one_liner 原文",
  "minutes": 12,
  "files": [{"name": "notes.md", "href": "…/notes.md"}],
  "sections": [{"id": "tldr", "n": "01", "title": "结论", "text": "…"}]
}]}
```

`text` 是**完整**小节正文（不截断；最长一节约 8000 字），整块 SITE 约 135 KiB / gzip 96 KiB。
门户因此必须**在命中位置附近截窗口**做片段（约 90 字，命中词高亮），不要把整节 text 直接渲染出来。
`n`（页内编号）可能为空字符串——有些页面用小节标题排版成的小节本身不编号，原样尊重。

```jsonc
// PATHS: curation.json 原样，门户渲染成若干条阅读路线
[{"name": "推理链路主线", "desc": "一条请求从进到出，先把主链路上的机制看全",
  "items": ["inference-systems/sglang-request-lifecycle", "…"]}]
```

### 文章页 shell 契约（`shell.py`）

```python
def inject(html: str, ctx: dict) -> str:
    """返回注入了站点外壳的整页 HTML。必须幂等：先剥掉旧的 shell 段再插新的。"""

# ctx:
#   "entry":     上面那个 entry dict
#   "siblings":  同分类的其他 entry，日期新→旧，当前篇也在里面（用于高亮）
#   "prev","next": 全局阅读顺序上的上一篇/下一篇（entry 或 None）
#             全局顺序 = 分类顺序 → 组内日期新→旧，与门户一致
#   "all":       全部 entry（页尾「同类条目」用 siblings 即可，all 供兜底）
#   "built":     "YYYY-MM-DD"
```

注入的五块内容，**全部只增不改正文**：

1. `<head>` 里的 meta 段：`description` + `og:*` + `canonical`（页面已有 `description` 则原位替换）。
2. `<head>` 里的 `<style id="sm-shell">`（`theme.TOKENS_CSS` + 外壳样式，`@media` 阶梯到 390px）。
3. `<body>` 之后的**吸顶条**：`show-me` 品牌 → 门户；分隔；分类 pill（带分类色点）→ 门户分类锚点；
   右侧：阅读时长 · 日期 · 「搜索 ⌘K」按钮（点了跳 `../../index.html?q=`，由门户接管）。
   吸顶条下方 2px 滚动进度线。
4. `<body>` 之后、正文之前插入**分类侧栏**：`position:fixed` 放在左侧空白边栏，
   只在 `min-width:1280px` 显示；内容是同分类条目列表（当前篇高亮 + 阅读时长）。
   正文 `.wrap` 的 980px 宽度和位置一像素不动。
5. `</div>`（`.wrap` 收尾）之前插入**页尾导航**：上一篇 / 下一篇（带标题），
   以及「同分类的其他条目」卡片列表（≤640px 时两列变一列）。
不要动页脚 `footer` 里已有的来源/许可说明。

约束：

- 注入后 `<body>` 直接子元素仍是 `.wrap`（吸顶条与侧栏用 fixed，不参与流）；
  页尾导航放在 `.wrap` 内、原 `footer` 之前，这样它跟着 980px 量宽度走。
- shell 自身不许发起网络请求；不许依赖外部字体或 CDN。
- 正文里已有的 `.nav` 锚点条保持原样，不要重复生成页内 TOC。
- 所有新增 class 前缀 `sm-`，避免和 11 个页面各自的手写样式撞名。
- 390px 下页面 `documentElement.scrollWidth === clientWidth`。

---

## 四、门户版式

```
                     ┌ 吸顶站点头：品牌 · 条目 · 阅读路线 · 标签 ┐
hero（浅渐变）        │ kicker / h1 / 一句话 / 说明 / CTA 按钮组      │
                     │ stats：11 篇 · 3 分类 · 59 标签 · 最近更新    │
阅读路线             │ 2–3 条路线，每条 3–5 篇，显示 01→02→03 顺序   │
全部条目             │ 吸顶工具条：搜索 / 分类页签(带计数) / 标签抽屉 │
                     │ 分类标题：色点 + 中文名 + 一句话 + 篇数 · 最新 │
                     │ 卡片网格 2 列（移动端 1 列）                  │
footer               │ 生成方式 / meta.json 说明 / 许可              │
```

卡片解剖（自上而下，行内对齐靠预留高度，不靠 hope）：

```
[分类徽标·分类色]                          [12 分钟]
标题（2 行预留）
一句话摘要（4 行 clamp，预留 4 行高）
标签 3 个 + N                 
──────────────────────────────
2026-09-24 · notes.md · 阅读 →
```

搜索（⌘K）：输入即搜，命中粒度到**小节**，结果显示「条目标题 → 命中小节标题 → 片段（命中词高亮）」，
回车进 `/…/#section`。支持 ↑↓ 选择、Esc 关闭、点击遮罩关闭；URL `?q=` 可以预填（文章页的搜索按钮靠这个）。

---

## 五、验收

- `python3 build_index.py` 幂等：连跑两次，`git status` 除时间戳外无差异。
- `python3 check_links.py` 全绿。
- `python3 verify_site.py`：
  - 每个文章页 390 / 560 / 640 / 900 / 1280 / 1440 六档无横向溢出；
  - 门户在 390px 下 `scrollWidth === clientWidth`，卡片单列；
  - 门户交互断言：搜索输入后卡片数变化、⌘K 打开、Esc 关闭、分类页签过滤、标签抽屉开合。
- 桌面与 390px 各截图人工看一遍：卡片不空、标题不折成两行断词、数值行对齐。
