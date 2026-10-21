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
# 门户 index.html 的链接是 JS 在运行时按数据拼出来的，静态扫描只看得到模板片段，
# 会把 ' + esc(e._dir) + '/… 这类表达式误判成失效链接。这些由脚本内置的路径在
# 交互探针里断言（见 SKILL.md「the portal」一节），静态检查跳过。
RUNTIME_PATH_MARKERS = ("${", " + ", "`")


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
            # 查询串不是路径的一部分：index.html?q=xxx 校验的是 index.html
            target = raw.split("#", 1)[0].split("?", 1)[0]
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