#!/usr/bin/env python3
"""重建 show-me 知识库：门户 index.html、每个文章页的站点外壳、README 索引表。

一条命令搞定三件事：
  1. 读各目录 meta.json，并把文章页解析成「标题 / 导语 / 小节 / 纯文本」；
  2. 用 templates/portal.html 渲染带全库检索的门户；
  3. 用 shell.py 给每个文章页注入站点外壳（幂等，重复运行不叠加）。

用法：
    python3 build_index.py                # 全量重建
    python3 build_index.py --dry-run      # 只解析并打印，不写文件
    python3 build_index.py --only-shell   # 只重注外壳
    python3 build_index.py --only-portal  # 只重建门户与 README
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import theme

ROOT = Path(__file__).resolve().parent
PORTAL_TEMPLATE = ROOT / "templates" / "portal.html"
CURATION = ROOT / "curation.json"

CATEGORIES = theme.CATEGORIES
CAT_ZH = theme.CAT_ZH
CAT_ORDER = theme.CAT_ORDER

REQUIRED = ("title", "slug", "category", "date", "tags", "one_liner")

SHELL_BEGIN = "<!-- show-me:shell:begin -->"
SHELL_END = "<!-- show-me:shell:end -->"
SHELL_RE = re.compile(re.escape(SHELL_BEGIN) + r".*?" + re.escape(SHELL_END), re.S)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-.]*")


# ────────────────────────────────────────────────────────────── 页面解析

def strip_shell(page: str) -> str:
    """剥掉上一次注入的外壳段，让解析与再注入都基于原文。"""
    return SHELL_RE.sub("", page)


def ensure_h2_ids(page: str) -> str:
    """给没有 id 的正文 <h2> 补一个稳定 id，否则检索命中的小节无法深链。

    id 由该 h2 在页面内的序号决定（s01, s02 …），内容冻结的归档里已经足够稳定。
    图表卡片里的 <h2> 是图题而不是文档小节，跳过——给它一个 id 只会在复制链接时误导读者。
    """
    out: list[str] = []
    stack: list[str] = []          # 打开中的 div 的 class
    index = 0
    for line in page.split("\n"):
        for tok in re.finditer(r"<div\b[^>]*>|</div\s*>", line):
            text = tok.group(0)
            if text.startswith("</"):
                if stack:
                    stack.pop()
            else:
                cls = re.search(r'class="([^"]*)"', text)
                stack.append(cls.group(1) if cls else "")

        m = re.search(r"<h2(\s[^>]*)?>", line)
        if m:
            index += 1
            attrs = m.group(1) or ""
            if not re.search(r"\bid\s*=", attrs) and not any("card" in c for c in stack):
                line = f'{line[:m.start()]}<h2 id="s{index:02d}"{attrs}>{line[m.end():]}'
        out.append(line)
    return "\n".join(out)


class _Analyzer(HTMLParser):
    """抽取 h1、导语、h2 小节及其正文纯文本。跳过 script/style。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.h1 = ""
        self._in_h1 = False
        self.lead = ""
        self._lead_depth = 0
        self.sections: list[dict] = []
        self._cur: dict | None = None
        self._in_h2 = False
        self._in_n = 0            # 位于 <h2> 里的 <span class="n">
        self._h2_is_card = False  # 图表卡片里的 <h2> 是图题，不是文档小节
        self._divs: list[str] = []  # 打开中的 div 的 class，用来判断 h2 是否在卡片内
        self._h2_id = ""
        self._h2_n = ""
        self._h2_title: list[str] = []
        self.body_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
            return
        if self._skip:
            return

        if tag == "h1" and not self.h1:
            self._in_h1 = True
        elif tag == "div":
            self._divs.append(a.get("class") or "")
        elif tag == "p":
            cls = (a.get("class") or "").split()
            if "lead" in cls and not self.lead:
                self._lead_depth = 1
            elif self._lead_depth:
                self._lead_depth += 1
        elif tag == "h2":
            if any("card" in c for c in self._divs):
                # 图表卡片的标题：并入上一节正文，不当小节切分点
                self._h2_is_card = True
                return
            self._close_section()
            self._in_h2 = True
            self._h2_id = a.get("id") or ""
            self._h2_n = ""
            self._h2_title = []
        elif tag == "span" and self._in_h2 and (a.get("class") or "") == "n":
            self._in_n += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return

        if tag == "h1":
            self._in_h1 = False
        elif tag == "div":
            if self._divs:
                self._divs.pop()
        elif tag == "p" and self._lead_depth:
            self._lead_depth = 0
        elif tag == "h2":
            if self._h2_is_card:
                self._h2_is_card = False
                return
            if self._in_h2:
                self._in_h2 = False
                self._in_n = 0
                self._cur = {
                    "id": self._h2_id,
                    "n": _squash(self._h2_n),
                    "title": _squash("".join(self._h2_title)),
                    "text": [],
                }
        elif tag == "span" and self._in_n:
            self._in_n -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_h1:
            self.h1 += data
            return
        if self._lead_depth:
            self.lead += data
        if self._in_h2:
            if self._in_n:
                self._h2_n += data
            else:
                self._h2_title.append(data)
            return
        if self._cur is not None:
            self._cur["text"].append(data)
        self.body_text.append(data)

    def close(self) -> None:
        super().close()
        self._close_section()

    def _close_section(self) -> None:
        if self._cur is None:
            return
        self._cur["text"] = _squash("".join(self._cur["text"]))
        if self._cur["title"]:
            self.sections.append(self._cur)
        self._cur = None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut.rsplit(" ", 1)[0] if " " in cut[-30:] else cut


def reading_minutes(text: str) -> int:
    """中文按 400 字/分钟，英文按 220 词/分钟；下限 3 分钟。"""
    cjk = len(CJK_RE.findall(text))
    words = len(WORD_RE.findall(text))
    return max(3, round(cjk / 400 + words / 220))


def analyze(page: str) -> dict:
    p = _Analyzer()
    p.feed(page)
    p.close()
    body = _squash("".join(p.body_text))
    return {
        "h1": _squash(p.h1),
        "lead": _clip(_squash(p.lead), 200),
        "sections": p.sections,
        "minutes": reading_minutes(body),
        "text": body,
    }


# ────────────────────────────────────────────────────────────── 条目装载

def normalize_pages() -> list[str]:
    """把缺 id 的 <h2> 补齐并落盘。检索能深链到小节，前提是这些 id 真的在文件里。

    只加属性、不动原有内容；已有 id 的一律不动，所以重复运行是空操作。
    """
    touched: list[str] = []
    for page_path in sorted(ROOT.glob("*/*/index.html")):
        raw = page_path.read_text(encoding="utf-8")
        new = ensure_h2_ids(raw)
        if new != raw:
            page_path.write_text(new, encoding="utf-8")
            touched.append(page_path.relative_to(ROOT).as_posix())
    return touched


def load_entries() -> list[dict]:
    entries: list[dict] = []
    for meta_path in sorted(ROOT.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in meta]
        if missing:
            raise SystemExit(f"{meta_path}: 缺少字段 {missing}")

        page_path = meta_path.parent / meta.get("artifact", "index.html")
        if not page_path.exists():
            raise SystemExit(f"{meta_path}: artifact 不存在 —— {page_path.name}")

        entry = dict(meta)
        entry["_dir"] = meta_path.parent.relative_to(ROOT).as_posix()
        entry["_depth"] = 2
        entry["_root"] = "../" * 2
        entry["_cat_zh"] = CAT_ZH.get(entry["category"], entry["category"])
        entry["_color"] = theme.cat_color(entry["category"])
        entry["_page"] = page_path
        entry.setdefault("artifact", "index.html")
        entry.setdefault("extra_files", [])

        raw = ensure_h2_ids(strip_shell(page_path.read_text(encoding="utf-8")))
        info = analyze(raw)
        entry["_h1"] = info["h1"]
        entry["_lead"] = info["lead"]
        entry["_sections"] = info["sections"]
        entry["_minutes"] = info["minutes"]
        entries.append(entry)

    # 分类顺序 → 组内日期新→旧。这条顺序同时决定门户、上一篇/下一篇与分类侧栏。
    entries.sort(key=lambda e: e["date"], reverse=True)
    entries.sort(key=lambda e: CAT_ORDER.get(e["category"], 99))
    return entries


# ────────────────────────────────────────────────────────────── 载荷

def portal_payload(entries: list[dict]) -> dict:
    out = []
    for e in entries:
        out.append(
            {
                "dir": e["_dir"],
                "title": e["title"],
                "slug": e["slug"],
                "date": e["date"],
                "cat": e["category"],
                "catZh": e["_cat_zh"],
                "color": e["_color"],
                "tags": e.get("tags", []),
                "one": e["one_liner"],
                "minutes": e["_minutes"],
                "files": [
                    {"name": f, "href": f"{e['_dir']}/{f}"}
                    for f in e.get("extra_files", [])
                ],
                "sections": [
                    {
                        "id": s["id"],
                        "n": s["n"],
                        "title": s["title"],
                        "text": s["text"],
                    }
                    for s in e["_sections"]
                ],
            }
        )
    return {"entries": out}


def load_paths(entries: list[dict]) -> list[dict]:
    if not CURATION.exists():
        print(f"  ⚠ 没有 {CURATION.name}，门户不带阅读路线", file=sys.stderr)
        return []
    raw = json.loads(CURATION.read_text(encoding="utf-8"))
    known = {e["_dir"] for e in entries}
    paths = []
    for p in raw:
        items = [d for d in p.get("items", []) if d in known]
        dropped = [d for d in p.get("items", []) if d not in known]
        if dropped:
            print(f"  ⚠ 路线「{p.get('name')}」里有不存在的条目，已跳过：{dropped}",
                  file=sys.stderr)
        if items:
            paths.append({"name": p["name"], "desc": p.get("desc", ""), "items": items})
    return paths


def render_stats(entries: list[dict]) -> str:
    used = {e["category"] for e in entries}
    tags = {t for e in entries for t in e.get("tags", [])}
    latest = max((e["date"] for e in entries), default="—")
    return "".join(
        f'<div class="stat"><b>{v}</b><span>{label}</span></div>'
        for v, label in (
            (len(entries), "篇图解"),
            (len(used), "个分类"),
            (len(tags), "个标签"),
            (latest, "最近更新"),
        )
    )


# ────────────────────────────────────────────────────────────── 门户静态块
# 卡片与阅读路线在构建期就渲染成 HTML：没有 JS 也能把整库读完，
# 搜索引擎也看得见条目。JS 只做「过滤 / 搜索 / 已打开标记」这些增强。

def _esc(text) -> str:
    return _html.escape(str(text), quote=True)


def render_paths(paths: list[dict], by_dir: dict[str, dict]) -> str:
    if not paths:
        return '<p class="empty">还没有阅读路线，直接看下面的分类。</p>'
    blocks = []
    for path in paths:
        items = []
        for i, d in enumerate(path["items"]):
            e = by_dir.get(d)
            if not e:
                continue
            items.append(
                '<li><a href="{dir}/{art}">'
                '<span class="pn">{n:02d}</span>'
                '<span class="pt">{title}</span>'
                '<span class="pm"><i style="--c:{c}"></i>{m} 分钟</span>'
                "</a></li>".format(
                    dir=_esc(e["_dir"]),
                    art=_esc(e["artifact"]),
                    n=i + 1,
                    title=_esc(e["title"]),
                    c=_esc(e["_color"]),
                    m=e["_minutes"],
                )
            )
        blocks.append(
            '<div class="path">'
            '<div class="path-h">'
            f'<h3>{_esc(path["name"])}</h3>'
            f'<p>{_esc(path.get("desc", ""))}</p>'
            f'<span class="path-n">{len(items)} 篇</span>'
            "</div>"
            f'<ol class="path-items">{"".join(items)}</ol>'
            "</div>"
        )
    return "".join(blocks)


def _render_card(e: dict) -> str:
    tags = e.get("tags", [])
    shown = tags[:3]
    rest = len(tags) - len(shown)
    chips = "".join(f"<span>{_esc(t)}</span>" for t in shown)
    if rest > 0:
        chips += f'<span class="more">+{rest}</span>'
    notes = "".join(
        f'<a class="file" href="{_esc(e["_dir"])}/{_esc(f)}">{_esc(f)}</a>'
        for f in e.get("extra_files", [])
    )
    href = f'{_esc(e["_dir"])}/{_esc(e["artifact"])}'
    return (
        f'<article class="card" data-dir="{_esc(e["_dir"])}" data-cat="{_esc(e["category"])}">'
        '<div class="ctop">'
        f'<span class="badge" style="--c:{_esc(e["_color"])}">{_esc(e["_cat_zh"])}</span>'
        f'<span class="mins">{e["_minutes"]} 分钟</span>'
        "</div>"
        f'<h3><a href="{href}">{_esc(e["title"])}</a></h3>'
        f'<p class="one">{_esc(e["one_liner"])}</p>'
        f'<div class="tags">{chips}</div>'
        '<div class="cfoot">'
        f'<span class="date">{_esc(e["date"])}</span>{notes}'
        f'<a class="go" href="{href}">阅读 →</a>'
        "</div>"
        "</article>"
    )


def render_cards(entries: list[dict]) -> str:
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    blocks = []
    for cat, _zh in CATEGORIES:
        items = by_cat.get(cat)
        if not items:
            continue
        newest = max(e["date"] for e in items)
        blocks.append(
            f'<section class="cat" data-cat="{_esc(cat)}" id="cat-{_esc(cat)}">'
            '<div class="cat-h">'
            f'<span class="cd" style="--c:{_esc(items[0]["_color"])}" aria-hidden="true"></span>'
            f'<h3>{_esc(CAT_ZH.get(cat, cat))}</h3>'
            f'<code>{_esc(cat)}</code>'
            f'<span class="cat-n">{len(items)} 篇 · 最新 {_esc(newest)}</span>'
            f'<p class="blurb">{_esc(theme.CAT_BLURB.get(cat, ""))}</p>'
            "</div>"
            f'<div class="grid">{"".join(_render_card(e) for e in items)}</div>'
            "</section>"
        )
    return "".join(blocks)


def render_tabs(entries: list[dict]) -> str:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    tabs = [
        f'<button class="tab is-on" data-cat="all" aria-pressed="true">全部'
        f'<span class="tn">{len(entries)}</span></button>'
    ]
    for cat, zh in CATEGORIES:
        if cat not in counts:
            continue
        tabs.append(
            f'<button class="tab" data-cat="{_esc(cat)}" aria-pressed="false">'
            f'<i style="--c:{_esc(theme.cat_color(cat))}" aria-hidden="true"></i>'
            f'{_esc(zh)}<span class="tn">{counts[cat]}</span></button>'
        )
    return "".join(tabs)


def _json_literal(obj) -> str:
    """内联进 <script> 的 JSON：转义 </ 以免提前闭合脚本标签。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_portal(entries: list[dict], paths: list[dict], built: str) -> str:
    if not PORTAL_TEMPLATE.exists():
        raise SystemExit(
            f"找不到门户模板 {PORTAL_TEMPLATE.relative_to(ROOT)} —— "
            "门户由 build_index.py 生成，不要手改根目录 index.html"
        )
    template = PORTAL_TEMPLATE.read_text(encoding="utf-8")
    by_dir = {e["_dir"]: e for e in entries}
    start = paths[0]["items"][0] if paths and paths[0]["items"] else None
    start_entry = by_dir.get(start) if start else entries[0]
    mapping = {
        "TOKENS": theme.TOKENS_CSS,
        "SITE": _json_literal(portal_payload(entries)),
        "PATHS_HTML": render_paths(paths, by_dir),
        "CARDS_HTML": render_cards(entries),
        "TABS_HTML": render_tabs(entries),
        "STATS": render_stats(entries),
        "TOTAL": str(len(entries)),
        "START_HREF": f'{_esc(start_entry["_dir"])}/{_esc(start_entry["artifact"])}',
        "START_TITLE": _esc(start_entry["title"]),
        "BUILD": built,
    }
    # 单次扫描替换：避免 JSON 里恰好出现 __XXX__ 时被二次替换
    out = re.sub(
        r"__([A-Z][A-Z0-9_]*)__",
        lambda m: mapping.get(m.group(1), m.group(0)),
        template,
    )
    leftover = sorted(set(re.findall(r"__([A-Z][A-Z0-9_]*)__", out)))
    if leftover:
        raise SystemExit(f"门户模板里有未知占位符：{leftover}")
    return out


# ────────────────────────────────────────────────────────────── README

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
        sid = e.get("session_ref") or ""
        src = f"`{sid}`" if sid else "—"
        lines.append(
            f"| {e['date']} | {link} | {e['category']} | {tags} | {e['one_liner']} | {src} |"
        )
    table = "\n".join(lines)

    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- ENTRIES:BEGIN -->).*?(<!-- ENTRIES:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit("README.md 里找不到 ENTRIES 标记")
    return pattern.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", text)


# ────────────────────────────────────────────────────────────── 外壳

def inject_shells(entries: list[dict], built: str) -> int:
    try:
        import shell  # 可选模块：并行开发中，缺了就先跳过
    except ImportError:
        print("  ⚠ 没有 shell.py，跳过文章页外壳注入", file=sys.stderr)
        return 0

    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)

    changed = 0
    for i, e in enumerate(entries):
        before = e["_page"].read_text(encoding="utf-8")
        ctx = {
            "entry": e,
            "siblings": by_cat.get(e["category"], [e]),
            "prev": entries[i - 1] if i > 0 else None,
            "next": entries[i + 1] if i + 1 < len(entries) else None,
            "all": entries,
            "built": built,
        }
        after = shell.inject(before, ctx)
        if after != before:
            e["_page"].write_text(after, encoding="utf-8")
            changed += 1
    return changed


# ────────────────────────────────────────────────────────────── main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重建 show-me 门户、文章页外壳与 README")
    ap.add_argument("--dry-run", action="store_true", help="只解析并打印，不写文件")
    ap.add_argument("--only-shell", action="store_true", help="只重注文章页外壳")
    ap.add_argument("--only-portal", action="store_true", help="只重建门户与 README")
    args = ap.parse_args(argv)

    built = _dt.date.today().isoformat()

    if not args.dry_run:
        touched = normalize_pages()
        if touched:
            print(f"小节 id 已补齐：{len(touched)} 页")

    entries = load_entries()
    paths = load_paths(entries)

    if args.dry_run:
        for e in entries:
            print(
                f"  {e['date']}  {e['category']}/{e['slug']:<30} "
                f"{e['_minutes']:>3} 分钟  {len(e['_sections']):>2} 节  {e['title']}"
            )
        print(f"\n共 {len(entries)} 篇，{len(paths)} 条阅读路线（dry-run，未写文件）")
        return 0

    if not args.only_shell:
        portal = render_portal(entries, paths, built)
        (ROOT / "index.html").write_text(portal, encoding="utf-8")
        (ROOT / "README.md").write_text(render_readme(entries), encoding="utf-8")
        print(f"门户已重建：{len(entries)} 篇 / {len(paths)} 条路线 → index.html")

    if not args.only_portal:
        changed = inject_shells(entries, built)
        print(f"文章页外壳：{changed} 页有变化，{len(entries) - changed} 页已是最新")

    return 0


if __name__ == "__main__":
    sys.exit(main())
