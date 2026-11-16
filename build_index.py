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
TAXONOMY = ROOT / "taxonomy.json"

CATEGORIES = theme.CATEGORIES
CAT_ZH = theme.CAT_ZH
CAT_ORDER = theme.CAT_ORDER

REQUIRED = ("title", "slug", "category", "date", "tags", "one_liner")

# 公开仓库里不该出现的字面内容：本机绝对路径、完整 session id。
# 这些是「已经脱敏过」的声明最容易滑掉的地方 —— meta.json 本身也是公开文件。
ABS_PATH_RE = re.compile(
    r"(?:^|[\s(\[（:：=,])"
    r"(?:/(?:Users|home|mnt|opt|srv|data|workspace|root|private|var)/(?!\s)|[A-Za-z]:\\\\)"
)
SESSION_FULL_RE = re.compile(r"\bsess_[A-Za-z0-9]{13,}")

# visibility 描述的是「原始材料的敏感度」，不是「是否已批准公开」；
# 后者由 publish 决定。public-ok 是旧口径的混血值，构建时归一到 public。
VISIBILITY_VALUES = ("public", "internal")
VISIBILITY_ALIASES = {"public-ok": "public"}
DEFAULT_PUBLISH = True

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


def load_alias_map() -> dict[str, str]:
    """读 taxonomy.json，得到「任意写法 → 规范写法」的映射。

    标签是门户上唯一跨条目聚合的维度，放任自由写法会让 95 个标签里 75 个只出现一次。
    归并只发生在展示层，各篇 meta.json 保留作者写下的原词。
    """
    if not TAXONOMY.exists():
        return {}
    raw = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for canonical, variants in (raw.get("aliases") or {}).items():
        mapping[canonical] = canonical
        for variant in variants:
            mapping[str(variant)] = canonical
    return mapping


def normalize_tags(tags: list[str], aliases: dict[str, str]) -> list[str]:
    """按词表归并标签，保留首次出现的顺序并去重。"""
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        canonical = aliases.get(str(tag), str(tag))
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _walk_strings(obj, field: str = ""):
    """递归产出 (字段路径, 字符串)，用来扫 meta 里的字面泄漏。"""
    if isinstance(obj, str):
        yield field, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_strings(value, f"{field}.{key}" if field else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk_strings(value, f"{field}[{i}]")


def scrub_meta(meta_path: Path, meta: dict) -> list[str]:
    """把 meta.json 收敛到公开仓库口径，返回做过哪些修正（幂等）。

    - cwd 是本机绝对路径时改写为仓库相对路径（仓库外则为空串），保留可回溯性但不出本机；
    - visibility 的旧混血值 public-ok 归一到 public；
    - 其余字段里若仍有绝对路径或完整 session id，直接中止构建 —— 这类内容一旦推上去
      就无法从 git 历史里真正撤回，宁可卡在本地。
    """
    notes: list[str] = []

    cwd = meta.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        path = Path(cwd)
        if path.is_absolute():
            try:
                rel = path.resolve().relative_to(ROOT)
                meta["cwd"] = "" if rel == Path(".") else rel.as_posix()
            except ValueError:
                meta["cwd"] = ""
            notes.append("cwd 改写为仓库相对路径")

    visibility = meta.get("visibility")
    if isinstance(visibility, str) and visibility in VISIBILITY_ALIASES:
        meta["visibility"] = VISIBILITY_ALIASES[visibility]
        notes.append(f"visibility {visibility} → {meta['visibility']}")
    if meta.get("visibility") not in VISIBILITY_VALUES:
        raise SystemExit(
            f"{meta_path.relative_to(ROOT).as_posix()}: visibility 只能是 "
            f"{' / '.join(VISIBILITY_VALUES)}，当前 {meta.get('visibility')!r}"
        )

    leaks = []
    for field, value in _walk_strings(meta):
        if field == "cwd":
            continue
        if ABS_PATH_RE.search(value):
            leaks.append(f"{field}: 本机绝对路径")
        if SESSION_FULL_RE.search(value):
            leaks.append(f"{field}: 完整 session id")
    if leaks:
        detail = "\n  ".join(leaks)
        raise SystemExit(
            f"{meta_path.relative_to(ROOT).as_posix()}: 有未脱敏内容，已中止（不打印原值）：\n  {detail}"
        )

    if notes:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return notes


def load_entries() -> list[dict]:
    entries: list[dict] = []
    unpublished: list[str] = []
    notes: list[str] = []
    aliases = load_alias_map()
    merged_tags = 0
    for meta_path in sorted(ROOT.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in meta]
        if missing:
            raise SystemExit(f"{meta_path}: 缺少字段 {missing}")

        fixed = scrub_meta(meta_path, meta)
        if fixed:
            notes.append(f"{meta_path.parent.name}: {'；'.join(fixed)}")

        publish = meta.get("publish", DEFAULT_PUBLISH)
        if not isinstance(publish, bool):
            raise SystemExit(
                f"{meta_path.relative_to(ROOT).as_posix()}: publish 必须是 true / false"
            )
        if not publish:
            # 未批准公开：不进门户、不进 README、不参与上一篇/下一篇，避免半公开
            unpublished.append(meta_path.parent.relative_to(ROOT).as_posix())
            continue

        page_path = meta_path.parent / meta.get("artifact", "index.html")
        if not page_path.exists():
            raise SystemExit(f"{meta_path}: artifact 不存在 —— {page_path.name}")

        entry = dict(meta)
        raw_tags = list(entry.get("tags") or [])
        entry["tags"] = normalize_tags(raw_tags, aliases)
        merged_tags += len(raw_tags) - len(entry["tags"])
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

    for note in notes:
        print(f"  脱敏修正 {note}", file=sys.stderr)
    if merged_tags:
        print(f"  标签词表归并：{merged_tags} 处重复/变体写法已合并", file=sys.stderr)
    if unpublished:
        print(f"  未发布（publish=false）：{len(unpublished)} 篇，已排除", file=sys.stderr)

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


def load_paths(entries: list[dict]) -> tuple[list[dict], list[str], int]:
    """读 curation.json，返回 (渲染用的路线, 未收录且已登记的条目, 路线覆盖的条目数)。

    阅读路线是手工清单 —— 新文章不会自己进去。所以这里做三件事：
      1. 丢掉路线里已不存在的目录（重构/改名后的残留）；
      2. 承认 unlisted：横向对比、站点自身文档这类「确定不进路线」的条目显式登记，
         否则它们会一直被当成漏收录；
      3. 剩下的（既不在 routes 也不在 unlisted）交回调用方告警 —— 让漏收录自己冒出来。
    """
    if not CURATION.exists():
        print(f"  ⚠ 没有 {CURATION.name}，门户不带阅读路线", file=sys.stderr)
        return [], [], 0

    raw = json.loads(CURATION.read_text(encoding="utf-8"))
    # 兼容两种写法：老的纯数组，以及带 unlisted 的对象
    if isinstance(raw, list):
        raw_routes, raw_unlisted = raw, []
    elif isinstance(raw, dict):
        raw_routes = raw.get("routes") or []
        raw_unlisted = raw.get("unlisted") or []
    else:
        raise SystemExit(f"{CURATION.name}: 顶层必须是数组或带 routes/unlisted 的对象")

    known = {e["_dir"] for e in entries}
    paths: list[dict] = []
    covered: set[str] = set()
    for p in raw_routes:
        if not isinstance(p, dict):
            raise SystemExit(f"{CURATION.name}: routes 里每一项都必须是对象，遇到 {type(p).__name__}")
        wanted = list(p.get("items", []))
        items = [d for d in wanted if d in known]
        dropped = [d for d in wanted if d not in known]
        if dropped:
            print(f"  ⚠ 路线「{p.get('name')}」里有不存在的条目，已跳过：{dropped}",
                  file=sys.stderr)
        if items:
            paths.append(
                {
                    "name": p["name"],
                    "desc": p.get("desc", ""),
                    "kind": p.get("kind", "path"),
                    "items": items,
                }
            )
            covered.update(items)

    unlisted = [d for d in raw_unlisted if d in known]
    for d in raw_unlisted:
        if d not in known:
            print(f"  ⚠ unlisted 里的条目不存在：{d}", file=sys.stderr)
    both = sorted(covered & set(unlisted))
    if both:
        print(f"  ⚠ 这些条目既在路线里又在 unlisted 里，已按「进路线」处理：{both}",
              file=sys.stderr)

    uncovered = [e["_dir"] for e in entries
                 if e["_dir"] not in covered and e["_dir"] not in unlisted]
    for d in uncovered:
        print(f"  ⚠ 未收录进任何阅读路线：{d}（确认不该进就写进 {CURATION.name} 的 unlisted）",
              file=sys.stderr)

    return paths, unlisted, len(covered)


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


def render_recent(entries: list[dict], limit: int = 5) -> str:
    """最近更新条：路线只覆盖一部分条目，新文章需要一个固定的入口，别沉在分类里。"""
    latest = sorted(entries, key=lambda e: e["date"], reverse=True)[:limit]
    rows = []
    for e in latest:
        rows.append(
            '<a class="recent-a" href="{dir}/{art}" data-dir="{dir}">'
            '<span class="rd">{date}</span>'
            '<span class="rc" style="--c:{color}" aria-hidden="true"></span>'
            '<span class="rt">{title}</span>'
            '<span class="rm">{minutes} 分钟</span>'
            "</a>".format(
                dir=_esc(e["_dir"]),
                art=_esc(e["artifact"]),
                date=_esc(e["date"]),
                color=_esc(e["_color"]),
                title=_esc(e["title"]),
                minutes=e["_minutes"],
            )
        )
    return '<div class="recent">' + "".join(rows) + "</div>"


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
                '<li><a href="{dir}/{art}" data-dir="{dir}">'
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
        kind = path.get("kind", "path")
        kind_label = "路线" if kind == "path" else "合集"
        blocks.append(
            '<div class="path">'
            '<div class="path-h">'
            f'<h3>{_esc(path["name"])}</h3>'
            f'<p>{_esc(path.get("desc", ""))}</p>'
            f'<span class="path-n kind">{kind_label} · {len(items)} 篇</span>'
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


def render_portal(entries: list[dict], paths: list[dict], built: str,
                  path_cover: int = 0) -> str:
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
        "RECENT_HTML": render_recent(entries),
        "PATHS_HTML": render_paths(paths, by_dir),
        "CARDS_HTML": render_cards(entries),
        "TABS_HTML": render_tabs(entries),
        "STATS": render_stats(entries),
        "TOTAL": str(len(entries)),
        "PATH_COVER": f"{path_cover} / {len(entries)}",
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
    paths, _unlisted, path_cover = load_paths(entries)

    if args.dry_run:
        for e in entries:
            print(
                f"  {e['date']}  {e['category']}/{e['slug']:<30} "
                f"{e['_minutes']:>3} 分钟  {len(e['_sections']):>2} 节  {e['title']}"
            )
        print(f"\n共 {len(entries)} 篇，{len(paths)} 条阅读路线覆盖 {path_cover} 篇"
              f"（dry-run，未写文件）")
        return 0

    if not args.only_shell:
        portal = render_portal(entries, paths, built, path_cover)
        (ROOT / "index.html").write_text(portal, encoding="utf-8")
        (ROOT / "README.md").write_text(render_readme(entries), encoding="utf-8")
        print(f"门户已重建：{len(entries)} 篇 / {len(paths)} 条路线（覆盖 {path_cover} 篇）"
              f" → index.html")

    if not args.only_portal:
        changed = inject_shells(entries, built)
        print(f"文章页外壳：{changed} 页有变化，{len(entries) - changed} 页已是最新")

    return 0


if __name__ == "__main__":
    sys.exit(main())
