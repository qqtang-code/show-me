#!/usr/bin/env python3
"""校验所有页面里的相对链接与资源引用是否可达。

捕捉的典型错误：页面在 <category>/<slug>/ 下，跨分类引用时少退一级
（如 ../comm-parallel/... 应为 ../../comm-parallel/...）。

用法：python3 check_links.py     # 退出码 1 表示有失效链接
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:", "data:", "javascript:")
LINK_RE = re.compile(r'(?:href|src)="([^"]+)"')


def check() -> list[tuple[str, str]]:
    broken: list[tuple[str, str]] = []
    for page in sorted(ROOT.rglob("*.html")):
        if ".git" in page.parts:
            continue
        html = page.read_text(encoding="utf-8", errors="ignore")
        for raw in LINK_RE.findall(html):
            if raw.startswith(SKIP_PREFIXES) or "${" in raw:
                continue
            target = raw.split("#", 1)[0]
            if not target:
                continue
            if not (page.parent / target).resolve().exists():
                broken.append((page.relative_to(ROOT).as_posix(), raw))
    return broken


def main() -> int:
    broken = check()
    if not broken:
        pages = sum(1 for p in ROOT.rglob("*.html") if ".git" not in p.parts)
        print(f"✓ {pages} 个页面的相对链接全部可达")
        return 0
    print(f"✗ 发现 {len(broken)} 处失效链接：")
    for page, url in broken:
        print(f"   {page}\n     → {url}")
    return 1


if __name__ == "__main__":
    sys.exit(main())