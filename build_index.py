#!/usr/bin/env python3
"""重建 show-me 知识库的 README 索引表与门户 index.html。

数据源：<category>/<slug>/meta.json
用法：python3 build_index.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 分类顺序与中文名；不在表里的分类排到最后
CATEGORIES = [
    ("inference-systems", "推理系统"),
    ("comm-parallel", "通信与并行"),
    ("training-algorithms", "训练与算法"),
    ("tooling", "工具与方法"),
    ("misc", "其他"),
]
CAT_ZH = dict(CATEGORIES)
CAT_ORDER = {name: i for i, (name, _) in enumerate(CATEGORIES)}

REQUIRED = ("title", "slug", "category", "date", "tags", "one_liner")


def load_entries() -> list[dict]:
    entries = []
    for meta_path in sorted(ROOT.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in meta]
        if missing:
            raise SystemExit(f"{meta_path}: 缺少字段 {missing}")
        entry = dict(meta)
        entry["_dir"] = meta_path.parent.relative_to(ROOT).as_posix()
        entry.setdefault("artifact", "index.html")
        entries.append(entry)
    entries.sort(key=lambda e: e["date"], reverse=True)          # 组内新的在前
    entries.sort(key=lambda e: CAT_ORDER.get(e["category"], 99))  # 稳定排序保住上面的日期序
    return entries


def render_readme(entries: list[dict]) -> str:
    lines = [
        "| 日期 | 主题 | 分类 | 标签 | 一句话 | 出处 |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        tags = " · ".join(e["tags"])
        link = f"[{e['title']}]({e['_dir']}/{e['artifact']})"
        extra = " ".join(f"[{f}]({e['_dir']}/{f})" for f in e.get("extra_files", []))
        if extra:
            link += f" ({extra})"
        date = e["date"]
        sid = e.get("session_ref") or ""
        src = f"`{sid}`" if sid else "—"
        lines.append(f"| {date} | {link} | {e['category']} | {tags} | {e['one_liner']} | {src} |")
    table = "\n".join(lines)

    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- ENTRIES:BEGIN -->).*?(<!-- ENTRIES:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit("README.md 里找不到 ENTRIES 标记")
    return pattern.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", text)


PORTAL = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<title>show-me 知识库 · 图解与概念解释合集</title>
<style>
  :root{
    --bg:#f5f7fb; --panel:#fff; --panel2:#f8fafc; --line:#dde3ee; --line2:#e9eef6;
    --fg:#131c2e; --ink:#0d1526; --dim:#4a5871; --dim2:#657389;
    --accent:#17509e; --accent-soft:#eef4fd; --accent-line:#d5e2f6;
    --chip:#eef2f9; --chipb:#dbe2ee;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    --shadow-1:0 1px 2px rgba(16,24,40,.04);
    --shadow-2:0 10px 28px -12px rgba(16,24,40,.22), 0 2px 6px rgba(16,24,40,.05);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15.5px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
       padding:0 20px 72px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1000px;margin:0 auto}
  code{font-family:var(--mono);font-size:.9em;background:var(--chip);color:#12489c;
       padding:1.5px 5px;border-radius:4px;border:1px solid var(--line2)}
  a{color:var(--accent)}

  /* ---------- 头部：一层极浅的渐变，把标题区与列表分开 ---------- */
  .hero{margin:0 -20px 26px;padding:52px 20px 30px;
        background:linear-gradient(180deg,#eaf0fa 0%,#f2f6fc 58%,rgba(245,247,251,0) 100%);
        border-bottom:1px solid transparent}
  .hero .inner{max-width:1000px;margin:0 auto}
  h1{font-size:31px;line-height:1.25;margin:0 0 10px;letter-spacing:-.022em;color:var(--ink);
     display:flex;align-items:center;gap:12px}
  h1 .logo{width:30px;height:30px;border-radius:9px;flex:none;
           background:linear-gradient(150deg,#2f6fd0,#17509e);color:#fff;
           display:grid;place-items:center;font:600 15px/1 var(--mono);
           box-shadow:0 4px 10px -3px rgba(23,80,158,.5)}
  .sub{color:var(--dim);font-size:14.5px;margin:0;max-width:74ch}
  .stats{display:flex;flex-wrap:wrap;gap:26px;margin:22px 0 0}
  .stat{display:flex;flex-direction:column;gap:1px}
  .stat b{font:600 20px/1.2 var(--mono);color:var(--ink);letter-spacing:-.01em}
  .stat span{font:11.5px var(--mono);color:var(--dim2);letter-spacing:.03em}

  /* ---------- 工具条：搜索 + 标签抽屉 ---------- */
  .tools{position:sticky;top:0;z-index:20;padding:12px 0;margin:0 0 4px;
         background:rgba(245,247,251,.92);backdrop-filter:blur(8px);
         border-bottom:1px solid var(--line2)}
  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
  .searchwrap{position:relative;flex:1 1 260px;min-width:0}
  .searchwrap svg{position:absolute;left:13px;top:50%;transform:translateY(-50%);
                  width:16px;height:16px;color:var(--dim2);pointer-events:none}
  #q{width:100%;padding:11px 14px 11px 38px;border:1px solid var(--line);border-radius:10px;
     font-size:14.5px;background:var(--panel);color:var(--fg);box-shadow:var(--shadow-1)}
  #q::placeholder{color:#93a0b4}
  #q:focus{outline:2px solid #c7d7f5;border-color:#a8c0e8}
  .tools .count{font:12px var(--mono);color:var(--dim2);white-space:nowrap}
  .toggle{display:inline-flex;align-items:center;gap:6px;font:12px var(--mono);color:#3f4c63;
          background:var(--chip);border:1px solid var(--chipb);border-radius:9px;
          padding:9px 12px;cursor:pointer;user-select:none;white-space:nowrap}
  .toggle:hover{background:#e6eefb;border-color:#bcd0f0;color:var(--accent)}
  .toggle .cv{transition:transform .18s ease;display:inline-block}
  .toggle[aria-expanded="true"] .cv{transform:rotate(180deg)}
  .tagbox{max-height:0;overflow:hidden;transition:max-height .24s ease}
  .tagbox.open{max-height:340px;overflow-y:auto}
  .chips{display:flex;flex-wrap:wrap;gap:6px;padding:12px 0 2px}
  .chip{font:11.5px var(--mono);background:var(--chip);border:1px solid var(--chipb);color:#3f4c63;
        border-radius:999px;padding:4px 10px;cursor:pointer;user-select:none;
        transition:background .14s ease,color .14s ease,border-color .14s ease}
  .chip:hover{background:#e6eefb;border-color:#bcd0f0;color:var(--accent)}
  .chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
  .chip[aria-pressed="true"]:hover{color:#fff}
  .chip .n{opacity:.62;margin-left:4px}
  .activebar{display:none;align-items:center;gap:8px;font:12px var(--mono);color:var(--dim2);
             padding:0 0 10px}
  .activebar.on{display:flex}
  .activebar button{font:inherit;background:none;border:0;color:var(--accent);cursor:pointer;padding:0}

  /* ---------- 分类 ---------- */
  .cathead{display:flex;align-items:baseline;gap:10px;margin:40px 0 14px;
           padding-bottom:9px;border-bottom:1px solid var(--line)}
  .cathead .cn{font-size:17px;font-weight:700;color:var(--ink);letter-spacing:-.01em}
  .cathead .ck{font:11.5px var(--mono);color:var(--dim2)}
  .cathead .cl{margin-left:auto;font:11.5px var(--mono);color:var(--dim2)}
  .cathead::before{content:"";width:8px;height:8px;border-radius:50%;flex:none;
                   transform:translateY(-2px);background:var(--accent)}
  .cathead[data-cat="comm-parallel"]::before{background:#0b6b3f}
  .cathead[data-cat="tooling"]::before{background:#a35a06}
  .cathead[data-cat="training-algorithms"]::before{background:#6b2fb5}
  .cathead[data-cat="misc"]::before{background:#657389}

  /* ---------- 卡片：等高 + 摘要定高，行内不再参差 ---------- */
  .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:stretch}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:13px;
        padding:17px 18px 14px;box-shadow:var(--shadow-1);
        display:flex;flex-direction:column;gap:10px;position:relative;overflow:hidden;
        transition:box-shadow .16s ease,border-color .16s ease,transform .16s ease}
  .card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);
                transform:scaleX(0);transform-origin:left;transition:transform .22s ease}
  .card:hover{box-shadow:var(--shadow-2);border-color:#c9d6ea;transform:translateY(-2px)}
  .card:hover::before,.card:focus-within::before{transform:scaleX(1)}
  .card h3{margin:0;font-size:16px;line-height:1.45;letter-spacing:-.005em}
  .card h3 a{color:var(--ink);text-decoration:none}
  .card h3 a::after{content:"→";display:inline-block;margin-left:6px;color:var(--accent);
                    opacity:0;transform:translateX(-4px);
                    transition:opacity .18s ease,transform .18s ease}
  .card:hover h3 a::after,.card:focus-within h3 a::after{opacity:1;transform:translateX(0)}
  .card .one{color:var(--dim);font-size:13.5px;line-height:1.68;margin:0;
             display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
  .card .tags{display:flex;flex-wrap:wrap;gap:6px}
  .card .tags span{font:11.5px var(--mono);background:var(--chip);border:1px solid var(--chipb);
                   border-radius:999px;padding:3px 8px;color:#3f4c63}
  .card .tags span.more{background:#fff;border-style:dashed;color:var(--dim2)}
  .card .foot{margin-top:auto;display:flex;flex-wrap:wrap;gap:12px;align-items:center;
              font:11.5px var(--mono);color:var(--dim2);border-top:1px solid var(--line2);
              padding-top:10px}
  .card .foot .date{color:#3f4c63}
  .card .foot a{color:var(--accent);text-decoration:none}
  .card .foot a:hover{text-decoration:underline}
  .grid > .card:last-child:nth-child(odd){grid-column:1 / -1}
  .grid > .card:last-child:nth-child(odd) .one{-webkit-line-clamp:3}
  .empty{color:var(--dim2);font:13px var(--mono);padding:46px 0;text-align:center;
         border:1px dashed var(--line);border-radius:13px;background:var(--panel)}

  footer{color:var(--dim2);font:11.5px/1.75 var(--mono);text-align:center;
         margin-top:52px;padding-top:20px;border-top:1px solid var(--line2)}

  @media(max-width:760px){
    body{padding:0 14px 56px}
    .hero{margin:0 -14px 20px;padding:34px 14px 22px}
    h1{font-size:24px}
    h1 .logo{width:26px;height:26px;border-radius:8px;font-size:13px}
    .stats{gap:20px}
    .stat b{font-size:17px}
    .grid{grid-template-columns:minmax(0,1fr)}
    .grid > .card:last-child:nth-child(odd){grid-column:auto}
    .card .one{-webkit-line-clamp:5}
    .tools{padding:10px 0}
    .cathead{margin:30px 0 12px}
    .cathead .cl{display:none}
  }
</style>
</head>
<body>
<div class="hero">
  <div class="inner">
    <h1><span class="logo">▤</span>show-me 知识库</h1>
    <p class="sub">每次 <code>/show-me</code> 产出的图解与概念解释归档于此。一篇一个目录：<code>index.html</code>
      是产物本体，<code>meta.json</code> 是元数据，纯文字回答另存 <code>notes.md</code>。</p>
    <div class="stats">__STATS__</div>
  </div>
</div>

<div class="wrap">
  <div class="tools">
    <div class="controls">
      <div class="searchwrap">
        <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <circle cx="9" cy="9" r="5.5" stroke="currentColor" stroke-width="1.6"/>
          <path d="M13.2 13.2 17 17" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
        <input id="q" type="search" placeholder="搜索标题、标签、一句话…" autocomplete="off"
               aria-label="搜索条目">
      </div>
      <button class="toggle" id="tagtoggle" aria-expanded="false" aria-controls="tagbox">
        标签 <span class="cv">⌄</span>
      </button>
      <span class="count" id="count" aria-live="polite"></span>
    </div>
    <div class="tagbox" id="tagbox"><div class="chips" id="chips"></div></div>
    <div class="activebar" id="activebar"></div>
  </div>

  <div id="list"></div>

  <footer>由 build_index.py 从各目录 meta.json 生成 · 公开学习资料，内部标识已脱敏</footer>
</div>
<script>
const ENTRIES = __ENTRIES__;
const CATS = __CATS__;
const state = { q: "", tag: null };

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function matches(e) {
  if (state.tag && !(e.tags || []).includes(state.tag)) return false;
  if (!state.q) return true;
  const hay = [e.title, e.one_liner, e.category, ...(e.tags || [])].join(" ").toLowerCase();
  return hay.includes(state.q.toLowerCase());
}

function render() {
  const list = document.getElementById("list");
  const shown = ENTRIES.filter(matches);
  document.getElementById("count").textContent = shown.length === ENTRIES.length
    ? ENTRIES.length + " 篇" : shown.length + " / " + ENTRIES.length + " 篇";

  const bar = document.getElementById("activebar");
  if (state.tag) {
    bar.className = "activebar on";
    bar.innerHTML = '已筛选标签 <b>' + esc(state.tag) + '</b> <button id="cleartag">清除</button>';
    bar.querySelector("#cleartag").addEventListener("click", () => setTag(null));
  } else {
    bar.className = "activebar";
    bar.innerHTML = "";
  }

  if (!shown.length) {
    list.innerHTML = '<div class="empty">没有匹配的条目 —— 换个关键词，或清除标签筛选</div>';
    return;
  }

  const byCat = new Map();
  for (const e of shown) {
    if (!byCat.has(e.category)) byCat.set(e.category, []);
    byCat.get(e.category).push(e);
  }
  let html = "";
  for (const [cat, items] of byCat) {
    html += '<div class="cathead" data-cat="' + esc(cat) + '">' +
              '<span class="cn">' + esc(CATS[cat] || cat) + '</span>' +
              '<span class="ck">' + esc(cat) + '</span>' +
              '<span class="cl">' + items.length + ' 篇 · 最新 ' + esc(items[0].date) + '</span>' +
            '</div><div class="grid">';
    for (const e of items) {
      const MAXT = 3;
      const tags = (e.tags || []).slice(0, MAXT);
      const rest = (e.tags || []).length - tags.length;
      const chips = tags.map(t => '<span>' + esc(t) + '</span>').join("") +
                    (rest > 0 ? '<span class="more">+' + rest + '</span>' : "");
      const extras = (e.extra_files || [])
        .map(f => '<a href="' + esc(e._dir) + '/' + esc(f) + '">' + esc(f) + '</a>').join("");
      html += '<div class="card">' +
        '<h3><a href="' + esc(e._dir) + '/' + esc(e.artifact) + '">' + esc(e.title) + '</a></h3>' +
        '<p class="one">' + esc(e.one_liner) + '</p>' +
        '<div class="tags">' + chips + '</div>' +
        '<div class="foot"><span class="date">' + esc(e.date) + '</span>' + extras +
          '<a href="' + esc(e._dir) + '/meta.json">meta.json</a></div>' +
      '</div>';
    }
    html += "</div>";
  }
  list.innerHTML = html;
}

function setTag(tag) {
  state.tag = tag;
  document.querySelectorAll(".chip").forEach(c =>
    c.setAttribute("aria-pressed", String(c.dataset.tag === tag)));
  render();
}

function renderChips() {
  const counts = new Map();
  for (const e of ENTRIES) for (const t of (e.tags || [])) counts.set(t, (counts.get(t) || 0) + 1);
  // 高频标签排在前面，低频的在后 —— 单次点击最可能命中的先看到
  const tags = [...counts.keys()].sort((a, b) =>
    counts.get(b) - counts.get(a) || a.localeCompare(b));
  document.getElementById("chips").innerHTML = tags.map(t =>
    '<button class="chip" data-tag="' + esc(t) + '" aria-pressed="false">' +
    esc(t) + '<span class="n">' + counts.get(t) + '</span></button>').join("");
  document.querySelectorAll(".chip").forEach(c =>
    c.addEventListener("click", () => setTag(state.tag === c.dataset.tag ? null : c.dataset.tag)));
}

const tagtoggle = document.getElementById("tagtoggle");
const tagbox = document.getElementById("tagbox");
tagtoggle.addEventListener("click", () => {
  const open = tagtoggle.getAttribute("aria-expanded") === "true";
  tagtoggle.setAttribute("aria-expanded", String(!open));
  tagbox.classList.toggle("open", !open);
  if (!open && state.tag) tagbox.scrollTop = 0;
});

document.getElementById("q").addEventListener("input", ev => { state.q = ev.target.value; render(); });
document.addEventListener("keydown", ev => {
  if (ev.key === "/" && document.activeElement !== document.getElementById("q")) {
    ev.preventDefault(); document.getElementById("q").focus();
  }
  if (ev.key === "Escape" && state.tag) setTag(null);
});

renderChips(); render();
</script>
</body>
</html>
"""


def render_portal(entries: list[dict]) -> str:
    payload = []
    for e in entries:
        payload.append({k: e[k] for k in
                        ("title", "slug", "category", "date", "tags", "one_liner",
                         "artifact", "session_id") if k in e}
                       | {"_dir": e["_dir"], "extra_files": e.get("extra_files", [])})
    cats = dict(CATEGORIES)
    # 只保留实际存在的分类，避免门户出现空分类
    used = {e["category"] for e in entries}
    all_tags = {t for e in entries for t in e.get("tags", [])}
    cats = {k: v for k, v in cats.items() if k in used}
    latest = max((e["date"] for e in entries), default="—")
    stats = (f'<div class="stat"><b>{len(entries)}</b><span>篇图解</span></div>'
             f'<div class="stat"><b>{len(used)}</b><span>个分类</span></div>'
             f'<div class="stat"><b>{len(all_tags)}</b><span>个标签</span></div>'
             f'<div class="stat"><b>{latest}</b><span>最近更新</span></div>')
    return (PORTAL
            .replace("__ENTRIES__", json.dumps(payload, ensure_ascii=False, indent=2))
            .replace("__CATS__", json.dumps(cats, ensure_ascii=False))
            .replace("__STATS__", stats))


def main() -> None:
    entries = load_entries()
    (ROOT / "README.md").write_text(render_readme(entries), encoding="utf-8")
    (ROOT / "index.html").write_text(render_portal(entries), encoding="utf-8")
    print(f"已重建索引：{len(entries)} 篇")
    for e in entries:
        print(f"  {e['date']}  {e['category']}/{e['slug']}  {e['title']}")


if __name__ == "__main__":
    main()