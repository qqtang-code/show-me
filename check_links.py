#!/usr/bin/env python3
"""校验所有页面里的相对链接、资源引用与页内锚点是否可达。

捕捉的典型错误：
  - 页面在 <category>/<slug>/ 下，跨分类引用时少退一级
    （如 ../comm-parallel/... 应为 ../../comm-parallel/...）；
  - 链接写成 index.html#inference-systems，但门户里的分类锚点其实是
    id="cat-inference-systems"，文件存在、锚点不存在 —— 读者点过去只是
    停在页首，看起来「能用」，所以静态检查必须真的解析 fragment。

用法：python3 check_links.py     # 退出码 1 表示有失效链接或锚点
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:", "data:", "javascript:")
LINK_RE = re.compile(r'(?:href|src)="([^"]+)"')
# 门户 index.html 的链接是 JS 在运行时按数据拼出来的，静态扫描只看得到模板片段，
# 会把 ' + esc(e._dir) + '/… 这类表达式误判成失效链接。这些由脚本内置的路径在
# 交互探针里断言（见 SKILL.md「the portal」一节），静态检查跳过。
RUNTIME_PATH_MARKERS = ("${", " + ", "`")
# 这些属性里的值由 JS 运行时读取，不是可点击的目标
NON_TARGET_ATTRS = ("data-q=",)


class _IdCollector(HTMLParser):
    """收集一份 HTML 里所有 id（锚点目标的唯一来源）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)

    handle_startendtag = handle_starttag


_ID_CACHE: dict[Path, set[str]] = {}


def ids_of(path: Path) -> set[str]:
    """读取并缓存一个页面的 id 集合。读不了就返回空集（不静默通过）。"""
    cached = _ID_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        collector = _IdCollector()
        collector.feed(path.read_text(encoding="utf-8", errors="ignore"))
        collector.close()
        ids = collector.ids
    except (OSError, ValueError):
        ids = set()
    _ID_CACHE[path] = ids
    return ids


def _is_anchor_target(page: Path) -> bool:
    return page.suffix.lower() in {".html", ".htm"}


def check() -> list[tuple[str, str]]:
    broken: list[tuple[str, str]] = []
    for page in sorted(ROOT.rglob("*.html")):
        if ".git" in page.parts:
            continue
        # templates/ 放的是未替换占位符的门户源模板，不是可访问页面，跳过自检
        if "templates" in page.relative_to(ROOT).parts[:-1]:
            continue
        html = page.read_text(encoding="utf-8", errors="ignore")
        for raw in LINK_RE.findall(html):
            if raw.startswith(SKIP_PREFIXES) or any(m in raw for m in RUNTIME_PATH_MARKERS):
                continue
            if any(attr in raw for attr in NON_TARGET_ATTRS):
                continue
            # 查询串不是路径的一部分：index.html?q=xxx 校验的是 index.html
            raw_path, _, fragment = raw.partition("#")
            target_part = raw_path.split("?", 1)[0]
            if target_part:
                target = (page.parent / target_part).resolve()
                if not target.exists():
                    broken.append((page.relative_to(ROOT).as_posix(), raw))
                    continue
            else:
                target = page
            # 只对同页/同站 HTML 校验锚点：.md 等非 HTML 目标没有 id 概念
            fragment = fragment.split("?", 1)[0]
            if not fragment or not target.is_file() or not _is_anchor_target(target):
                continue
            anchor = unquote(fragment)
            if anchor.lower() == "top":
                continue
            if anchor not in ids_of(target):
                broken.append((page.relative_to(ROOT).as_posix(), raw))
    return broken


def main() -> int:
    broken = check()
    if not broken:
        pages = sum(1 for p in ROOT.rglob("*.html")
                    if ".git" not in p.parts
                    and "templates" not in p.relative_to(ROOT).parts[:-1])
        print(f"✓ {pages} 个页面的相对链接与页内锚点全部可达")
        return 0
    print(f"✗ 发现 {len(broken)} 处失效链接或锚点：")
    for page, url in broken:
        print(f"   {page}\n     → {url}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
