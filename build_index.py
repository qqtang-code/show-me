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
    entries.sort(key=lambda e: (CAT_ORDER.get(e["category"], 99), e["date"], e["slug"]))
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
  :root{--bg:#f5f7fb;--panel:#fff;--panel2:#f8fafc;--line:#dde3ee;--fg:#131c2e;--dim:#4a5871;
        --dim2:#657389;--accent:#17509e;--chip:#eef2f9;--chipb:#dbe2ee;
        --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15.5px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
       padding:44px 20px 80px}
  .wrap{max-width:1000px;margin:0 auto}
  h1{font-size:28px;margin:0 0 8px;letter-spacing:-.01em}
  .sub{color:var(--dim);font-size:14px;margin:0 0 24px}
  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:8px}
  #q{flex:1;min-width:220px;padding:11px 14px;border:1px solid var(--line);border-radius:10px;
     font-size:14.5px;background:var(--panel);color:var(--fg)}
  #q:focus{outline:2px solid #c7d7f5;border-color:#a8c0e8}
  .count{font:12px var(--mono);color:var(--dim2)}
  .chips{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 26px}
  .chip{font:12px var(--mono);background:var(--chip);border:1px solid var(--chipb);color:#3f4c63;
        border-radius:999px;padding:5px 11px;cursor:pointer;user-select:none}
  .chip[aria-pressed="true"]{background:#17509e;border-color:#17509e;color:#fff}
  h2.cat{font-size:16px;margin:30px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line);
         display:flex;align-items:baseline;gap:9px;color:#0d1526}
  h2.cat span{font:11.5px var(--mono);color:var(--dim2)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:760px){.grid{grid-template-columns:1fr}body{padding:26px 14px 60px}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
        box-shadow:0 1px 2px rgba(16,24,40,.04);display:flex;flex-direction:column;gap:9px}
  .card h3{margin:0;font-size:16px;line-height:1.45}
  .card h3 a{color:#0d1526;text-decoration:none}
  .card h3 a:hover{color:var(--accent);text-decoration:underline}
  .card .one{color:var(--dim);font-size:13.5px;margin:0}
  .card .tags{display:flex;flex-wrap:wrap;gap:6px}
  .card .tags span{font:11.5px var(--mono);background:var(--chip);border:1px solid var(--chipb);
                   border-radius:999px;padding:3px 8px;color:#3f4c63}
  .card .foot{margin-top:auto;display:flex;flex-wrap:wrap;gap:12px;align-items:center;
              font:11.5px var(--mono);color:var(--dim2);border-top:1px solid #eef2f8;padding-top:10px}
  .card .foot a{color:var(--accent);text-decoration:none}
  .card .foot a:hover{text-decoration:underline}
  .empty{color:var(--dim2);font:13px var(--mono);padding:30px 0;text-align:center}
  footer{color:var(--dim2);font:11.5px/1.75 var(--mono);text-align:center;margin-top:44px}
</style>
</head>
<body>
<div class="wrap">
  <h1>show-me 知识库</h1>
  <p class="sub">每次 <code>/show-me</code> 产出的图解与概念解释归档于此。一篇一个目录：<code>index.html</code> 是产物本体，<code>meta.json</code> 是元数据，纯文字回答存 <code>notes.md</code>。</p>

  <div class="controls">
    <input id="q" type="search" placeholder="搜索标题、标签、一句话…" autocomplete="off">
    <span class="count" id="count"></span>
  </div>
  <div class="chips" id="chips"></div>

  <div id="list"></div>

  <footer>由 build_index.py 从各目录 meta.json 生成 · 公开学习资料，内部标识已脱敏</footer>
</div>
<script>
const ENTRIES = __ENTRIES__;
const CATS = __CATS__;
const state = { q: "", tag: null };

function matches(e) {
  if (state.tag && !e.tags.includes(state.tag)) return false;
  if (!state.q) return true;
  const hay = [e.title, e.one_liner, e.category, ...(e.tags || [])].join(" ").toLowerCase();
  return hay.includes(state.q.toLowerCase());
}

function render() {
  const list = document.getElementById("list");
  const shown = ENTRIES.filter(matches);
  const total = ENTRIES.length;
  document.getElementById("count").textContent = `${shown.length} / ${total} 篇`;

  if (!shown.length) {
    list.innerHTML = '<div class="empty">没有匹配的条目</div>';
    return;
  }
  const byCat = new Map();
  for (const e of shown) {
    if (!byCat.has(e.category)) byCat.set(e.category, []);
    byCat.get(e.category).push(e);
  }
  let html = "";
  for (const [cat, items] of byCat) {
    html += `<h2 class="cat">${CATS[cat] || cat}<span>${cat} · ${items.length}</span></h2><div class="grid">`;
    for (const e of items) {
      const tags = (e.tags || []).map(t => `<span>${t}</span>`).join("");
      const extras = (e.extra_files || []).map(f => `<a href="${e._dir}/${f}">${f}</a>`).join("");
      html += `<div class="card">
        <h3><a href="${e._dir}/${e.artifact}">${e.title}</a></h3>
        <p class="one">${e.one_liner}</p>
        <div class="tags">${tags}</div>
        <div class="foot"><span>${e.date}</span>${extras}<a href="${e._dir}/meta.json">meta.json</a></div>
      </div>`;
    }
    html += "</div>";
  }
  list.innerHTML = html;
}

function renderChips() {
  const tags = [...new Set(ENTRIES.flatMap(e => e.tags || []))].sort();
  const box = document.getElementById("chips");
  box.innerHTML = tags.map(t => `<button class="chip" data-tag="${t}" aria-pressed="${state.tag === t}">${t}</button>`).join("");
  box.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => {
    state.tag = state.tag === c.dataset.tag ? null : c.dataset.tag;
    renderChips(); render();
  }));
}

document.getElementById("q").addEventListener("input", ev => { state.q = ev.target.value; render(); });
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
    cats = {k: v for k, v in cats.items() if k in used}
    return (PORTAL
            .replace("__ENTRIES__", json.dumps(payload, ensure_ascii=False, indent=2))
            .replace("__CATS__", json.dumps(cats, ensure_ascii=False)))


def main() -> None:
    entries = load_entries()
    (ROOT / "README.md").write_text(render_readme(entries), encoding="utf-8")
    (ROOT / "index.html").write_text(render_portal(entries), encoding="utf-8")
    print(f"已重建索引：{len(entries)} 篇")
    for e in entries:
        print(f"  {e['date']}  {e['category']}/{e['slug']}  {e['title']}")


if __name__ == "__main__":
    main()