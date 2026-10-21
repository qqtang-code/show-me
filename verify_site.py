#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show-me 站点验收探针 —— DESIGN.md §5 的可执行版本。

用法：
    python3 verify_site.py [--root DIR] [--quick]

退出码：0 = 全部通过；1 = 有 FAIL；2 = 探针自身无法运行（找不到 Chrome / root 不合法）。

五组检查：
  A. 结构   meta.json 契约 / index.html 标签闭合 / shell 段标记 / 重建幂等
  B. 链接   相对 href·src 全部可达（复用 check_links.py）
  C. 布局   每页 6 档宽度（--quick 只跑 390/1440）无横向溢出，390px 无文字被裁切
  D. 交互   门户搜索 / ⌘K / Esc / ?q= 预填 / 分类页签 / 标签抽屉 / 卡片行内对齐
  E. 回归   离线优先 / 原 footer 与 h1 未被破坏 / 页内 TOC 不重复

所有探针、临时 HTML、Chrome profile 都写在 tempfile.mkdtemp() 里，结束即删；
本脚本不向仓库写入任何文件。
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
REQUIRED_META_KEYS = ("title", "slug", "category", "date", "tags", "one_liner")
FULL_WIDTHS = (390, 560, 640, 900, 1280, 1440)
QUICK_WIDTHS = (390, 1440)
# DESIGN §3 约定 5 段：head-meta（description / og / canonical）/ style / topbar /
# rail / footnav。外壳段数再变时，这里与 DESIGN §3 一起更新；
# 配对、重复段、未闭合仍逐页检查。
EXPECTED_SHELL_REGIONS = 5

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# HTML 允许省略结束标签的元素：遇到不匹配时静默弹出，避免把合法页面判成不闭合
OPTIONAL_END = {
    "p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot",
    "option", "optgroup", "colgroup", "rt", "rp",
}

CLASS_RE = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', re.I)
LINK_TAG_RE = re.compile(r"<link\b[^>]*>", re.I)
SRC_REMOTE_RE = re.compile(r'\b(?:src|srcset|poster)\s*=\s*["\']\s*(?:https?:)?//', re.I)
CSS_IMPORT_REMOTE_RE = re.compile(r"@import\s+(?:url\(\s*)?[\"']?\s*(?:https?:)?//", re.I)
CSS_URL_REMOTE_RE = re.compile(r"url\(\s*[\"']?\s*(?:https?:)?//", re.I)
REMOTE_LINK_HREF_RE = re.compile(r'href\s*=\s*["\']\s*(?:https?:)?//', re.I)
# <link> 里只有这些 rel 会真的发起加载；canonical / alternate / license 不拉资源，
# 离线优先检查放过它们（否则每页一条 canonical 都会误报）。
LINK_LOADING_RELS = {
    "stylesheet", "icon", "shortcut", "apple-touch-icon", "preload",
    "modulepreload", "prefetch", "preconnect", "dns-prefetch", "manifest",
    "mask-icon", "apple-touch-startup-image",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHELL_COMMENT_RE = re.compile(r"<!--\s*show-me:shell(?P<rest>[^>]*?)-->")

ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{3,}")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
STOPWORDS = set(
    """the and for with from that this then than when where what which while into
    your you are was were have has had not but all any can could should would will
    shall may might must does did doing been being over under about after before
    between each other some such only also more most much many very just same here
    there their them they these those because during without within through across
    above below off out own also onto upon does doing done""".split()
)

CHROME_CANDIDATES = (
    os.environ.get("CHROME"),
    os.environ.get("GOOGLE_CHROME"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome"),
    shutil.which("google-chrome-stable"),
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
)


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
class HarnessError(RuntimeError):
    """探针本身无法继续（不是站点的问题）。"""


class ProbeError(RuntimeError):
    """单次 headless Chrome 探针失败。"""


class Reporter:
    def __init__(self) -> None:
        self.passes = 0
        self.fails = 0
        self.skips = 0

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def info(self, label: str, detail: str = "") -> None:
        print(f"INFO  {label}" + (f"  {detail}" if detail else ""))

    def ok(self, label: str, detail: str = "") -> None:
        self.passes += 1
        print(f"PASS  {label}" + (f"  {detail}" if detail else ""))

    def fail(self, label: str, detail: str = "") -> None:
        self.fails += 1
        print(f"FAIL  {label}" + (f"  {detail}" if detail else ""))

    def skip(self, label: str, detail: str = "") -> None:
        self.skips += 1
        print(f"SKIP  {label}" + (f"  {detail}" if detail else ""))


def find_chrome() -> str | None:
    for cand in CHROME_CANDIDATES:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def article_pages(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*/index.html") if ".git" not in p.parts)


def gather_metas(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*/meta.json") if ".git" not in p.parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def count_class(html: str, token: str) -> int:
    """统计 class 属性里出现某个 class token 的元素数（脚本字符串可能误计，够用）。"""
    n = 0
    for m in CLASS_RE.finditer(html):
        if token in m.group(1).split():
            n += 1
    return n


def _kill_probe_group(pid: int) -> None:
    """macOS 上 Chrome 的 launcher 会 re-exec，可能脱离原进程组；先按组杀一次。"""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_chrome_for_profile(profile: Path) -> None:
    """按本次运行独有的 --user-data-dir 路径兜底回收 Chrome（含已脱离进程组的浏览器进程）。"""
    needle = f"--user-data-dir={profile}"
    try:
        listing = subprocess.run(["ps", "-axo", "pid=,command="],
                                 capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for line in listing.splitlines():
        if needle not in line:
            continue
        try:
            pid = int(line.split(None, 1)[0])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def run_chrome_probe(chrome: str, tmpdir: Path, name: str, page_html: str) -> dict:
    """写一个临时 probe 页，跑 headless Chrome，回读 <pre id="out"> 的 JSON。"""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60]
    probe = tmpdir / f"probe_{safe}.html"
    probe.write_text(page_html, encoding="utf-8")
    profile = tmpdir / "chrome-profile"
    cmd = [
        chrome,
        "--headless=new",
        "--allow-file-access-from-files",  # 必需：否则 iframe 跨源，contentDocument 读不到
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
        "--disable-background-networking",
        f"--user-data-dir={profile}",
        # 全新的 user-data-dir 在 macOS 上会让 headless Chrome 卡在首次启动；
        # 加 --incognito 后约 1s 退出，且不碰用户正在用的浏览器 profile。
        "--incognito",
        "--window-size=1600,1400",
        "--virtual-time-budget=8000",
        "--dump-dom",
        probe.as_uri(),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
    except OSError as exc:
        raise ProbeError(f"无法启动 Chrome：{exc}")
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        _kill_probe_group(proc.pid)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise ProbeError("Chrome 探针超时（120s）")
    finally:
        _kill_probe_group(proc.pid)
        _kill_chrome_for_profile(profile)
    out = out or ""
    m = re.search(r'<pre id="out"[^>]*>(.*?)</pre>', out, re.S)
    if not m:
        tail = " / ".join((err or "").strip().splitlines()[-3:])[:300]
        raise ProbeError(f"探针无输出（chrome exit={proc.returncode}）stderr={tail or '(空)'}")
    raw = html_mod.unescape(m.group(1)).strip()
    if not raw or raw == "PENDING":
        raise ProbeError("探针未在 virtual-time-budget 内完成（iframe 未全部 onload）")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"探针输出不是 JSON：{exc} —— {raw[:200]}")


# ---------------------------------------------------------------------------
# A. 结构
# ---------------------------------------------------------------------------
class BalanceParser(HTMLParser):
    """栈式标签平衡检查；可选结束标签（p/li/td…）静默自动闭合。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.first_error: str | None = None
        self.start_count = 0

    def _error(self, message: str) -> None:
        if self.first_error is None:
            self.first_error = message

    def handle_starttag(self, tag: str, attrs) -> None:
        self.start_count += 1
        if tag in VOID_TAGS:
            return
        self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.start_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            return
        names = [t for t, _ in self.stack]
        if tag in names:
            idx = len(names) - 1 - names[::-1].index(tag)
            intervening = self.stack[idx + 1:]
            if all(t in OPTIONAL_END for t, _ in intervening):
                del self.stack[idx:]
                return
            open_tag, line = self.stack[idx + 1]
            self._error(f"<{open_tag}> 未闭合（第 {line} 行）就遇到 </{tag}>（第 {self.getpos()[0]} 行）")
            del self.stack[idx:]
            return
        self._error(f"多余的闭合标签 </{tag}>（第 {self.getpos()[0]} 行）")

    def first_unclosed(self) -> str | None:
        if not self.stack:
            return None
        # 优先报非可选标签；全是可选标签时也报第一个
        for tag, line in self.stack:
            if tag not in OPTIONAL_END:
                return f"<{tag}> 未闭合（第 {line} 行）"
        tag, line = self.stack[0]
        return f"<{tag}> 未闭合（第 {line} 行）"


def check_meta(root: Path, rep: Reporter) -> None:
    rep.section("A. 结构 STRUCTURE")
    metas = gather_metas(root)
    if not metas:
        rep.fail("[meta] <category>/<slug>/meta.json", "一个都没找到")
        return
    problems: list[str] = []
    for meta_path in metas:
        rp = rel(root, meta_path)
        try:
            data = json.loads(read_text(meta_path))
        except Exception as exc:  # noqa: BLE001 - 任何解析错误都要报出来
            problems.append(f"{rp}: JSON 解析失败：{exc}")
            continue
        if not isinstance(data, dict):
            problems.append(f"{rp}: 顶层不是 JSON 对象")
            continue
        missing = [k for k in REQUIRED_META_KEYS if k not in data]
        if missing:
            problems.append(f"{rp}: 缺字段 {', '.join(missing)}")
        for key in ("title", "slug", "category", "date", "one_liner"):
            if key in data and (not isinstance(data[key], str) or not data[key].strip()):
                problems.append(f"{rp}: {key} 为空或不是字符串")
        if "tags" in data and not isinstance(data["tags"], list):
            problems.append(f"{rp}: tags 不是数组")
        if isinstance(data.get("date"), str) and data["date"] and not DATE_RE.match(data["date"]):
            problems.append(f"{rp}: date={data['date']!r} 不是 YYYY-MM-DD")
        artifact = data.get("artifact") or "index.html"
        if not (meta_path.parent / artifact).is_file():
            problems.append(f"{rp}: artifact 文件不存在：{artifact}")
    if problems:
        rep.fail("[meta] meta.json 契约", f"{len(problems)} 处问题（共 {len(metas)} 个目录）")
        for p in problems[:12]:
            print(f"        └ {p}")
    else:
        rep.ok("[meta] meta.json 契约", f"{len(metas)}/{len(metas)} 个目录：必填键齐全，artifact 均存在")

    index = root / "index.html"
    if not index.is_file():
        rep.fail("[html] index.html", "文件不存在")
        return
    parser = BalanceParser()
    try:
        parser.feed(read_text(index))
        parser.close()
    except Exception as exc:  # noqa: BLE001
        rep.fail("[html] index.html 标签闭合", f"html.parser 抛错：{exc}")
        return
    err = parser.first_error or parser.first_unclosed()
    if err:
        rep.fail("[html] index.html 标签闭合", f"首个问题：{err}")
    else:
        rep.ok("[html] index.html 标签闭合", f"{parser.start_count} 个开始标签，栈为空")


def shell_region_kind(text: str) -> str:
    t = text.lower()
    if "<style" in t:
        return "style"
    if "sm-topbar" in t or 'class="sm-bar' in t:
        return "topbar"
    if "sm-rail" in t:
        return "rail"
    if "sm-foot" in t or "sm-more" in t:
        return "footnav"
    if "<meta" in t or "og:" in t or "canonical" in t:
        return "head-meta"
    return "other"


def parse_shell_markers(html: str) -> dict:
    begins: list[dict] = []
    ends: list[dict] = []
    for m in SHELL_COMMENT_RE.finditer(html):
        tokens = [t for t in re.split(r"[:\s]+", m.group("rest")) if t]
        kind = "begin" if "begin" in tokens else ("end" if "end" in tokens else None)
        if not kind:
            continue
        names = [t for t in tokens if t not in ("begin", "end")]
        item = {"pos": m.start(), "kind": kind, "part": names[0] if names else ""}
        (begins if kind == "begin" else ends).append(item)
    events = sorted(begins + ends, key=lambda e: e["pos"])
    depth = 0
    nested = False
    stray = False
    regions: list[dict] = []
    open_ev = None
    for ev in events:
        if ev["kind"] == "begin":
            depth += 1
            if depth > 1:
                nested = True
            open_ev = ev
        else:
            depth -= 1
            if depth < 0:
                stray = True
            if open_ev is not None:
                regions.append({
                    "part": open_ev["part"] or "",
                    "kind": shell_region_kind(html[open_ev["pos"]:ev["pos"]]),
                })
                open_ev = None
    parts: dict[str, dict[str, int]] = {}
    for ev in begins:
        parts.setdefault(ev["part"], {"begin": 0, "end": 0})["begin"] += 1
    for ev in ends:
        parts.setdefault(ev["part"], {"begin": 0, "end": 0})["end"] += 1
    return {
        "begins": len(begins),
        "ends": len(ends),
        "paired": depth == 0 and not nested and not stray,
        "parts": parts,
        "regions": regions,
    }


def check_shell_markers(root: Path, rep: Reporter) -> None:
    pages = article_pages(root)
    if not pages:
        rep.fail("[shell] shell 段标记", "没有文章页")
        return
    failures: list[str] = []
    for page in pages:
        stats = parse_shell_markers(read_text(page))
        rp = rel(root, page)
        problems = []
        if stats["begins"] == 0 and stats["ends"] == 0:
            problems.append("没有任何 show-me:shell 标记（shell 注入未运行）")
        else:
            if stats["begins"] != EXPECTED_SHELL_REGIONS:
                problems.append(f"begin={stats['begins']}（期望 {EXPECTED_SHELL_REGIONS}）")
            if stats["ends"] != EXPECTED_SHELL_REGIONS:
                problems.append(f"end={stats['ends']}（期望 {EXPECTED_SHELL_REGIONS}）")
            if not stats["paired"]:
                problems.append("标记未成对交替（嵌套或叠加）")
            kinds = [r["kind"] for r in stats["regions"]]
            dup = sorted({k for k in kinds if k != "other" and kinds.count(k) > 1})
            if dup:
                problems.append("同一外壳段重复出现：" + ", ".join(dup))
            if len(stats["parts"]) > 1 or any(k for k in stats["parts"] if k):
                bad = {k: v for k, v in stats["parts"].items()
                       if k and (v["begin"] != 1 or v["end"] != 1)}
                if bad:
                    problems.append("每个 shell 部件应恰好 1 对标记：" +
                                    ", ".join(f"{k}=b{v['begin']}/e{v['end']}" for k, v in bad.items()))
        if problems:
            detail = f"{rp}: " + "；".join(problems)
            if stats["regions"]:
                detail += "（实际段：" + ", ".join(r["kind"] for r in stats["regions"]) + "）"
            failures.append(detail)
    if failures:
        rep.fail(f"[shell] 每页 {EXPECTED_SHELL_REGIONS} 段 shell 标记且不叠加",
                 f"{len(pages) - len(failures)}/{len(pages)} 页合格")
        for item in failures[:11]:
            print(f"        └ {item}")
    else:
        rep.ok(f"[shell] 每页 {EXPECTED_SHELL_REGIONS} 段 shell 标记且不叠加",
               f"{len(pages)}/{len(pages)} 页 × {EXPECTED_SHELL_REGIONS} 段，成对交替")


def snapshot_articles(repo: Path) -> dict[str, str]:
    snap: dict[str, str] = {}
    for meta_path in gather_metas(repo):
        for f in sorted(meta_path.parent.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            snap[f.relative_to(repo).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return snap


def check_idempotent(root: Path, rep: Reporter, tmpdir: Path, quick: bool) -> None:
    build = root / "build_index.py"
    if not build.is_file():
        rep.skip("[idempotent] 重建幂等", "build_index.py 尚未落地（并行重写中），跳过")
        return
    if quick:
        rep.skip("[idempotent] 重建幂等", "--quick 跳过两次重建")
        return
    copy = tmpdir / "repo-copy"
    try:
        shutil.copytree(
            root, copy,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store",
                                          ".venv", "node_modules", "verify_site.py"),
        )
    except Exception as exc:  # noqa: BLE001
        rep.fail("[idempotent] 重建幂等", f"复制仓库失败：{exc}")
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for run_no in (1, 2):
        try:
            proc = subprocess.run([sys.executable, "build_index.py"], cwd=copy,
                                  capture_output=True, text=True, timeout=180, env=env)
        except subprocess.TimeoutExpired:
            rep.fail("[idempotent] 重建幂等", f"第 {run_no} 次 build_index.py 超时（180s）")
            return
        if proc.returncode != 0:
            tail = " / ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-4:])[:400]
            rep.fail("[idempotent] 重建幂等", f"第 {run_no} 次 build_index.py 退出码 {proc.returncode}：{tail}")
            return
        snap = snapshot_articles(copy)
        if run_no == 1:
            first = snap
    changed = sorted(k for k in set(first) | set(snap)
                     if first.get(k) != snap.get(k))
    if changed:
        rep.fail("[idempotent] 重建幂等",
                 f"第二次运行改动了 {len(changed)} 个文章目录文件（期望 0）")
        for name in changed[:8]:
            print(f"        └ {name}")
    else:
        rep.ok("[idempotent] 重建幂等",
               f"连续两次 build_index.py：{len(snap)} 个文件哈希不变")


# ---------------------------------------------------------------------------
# B. 链接
# ---------------------------------------------------------------------------
def check_links(root: Path, rep: Reporter) -> None:
    rep.section("B. 链接 LINKS")
    path = root / "check_links.py"
    if not path.is_file():
        rep.skip("[links] 相对链接", "check_links.py 不存在，跳过")
        return
    try:
        spec = importlib.util.spec_from_file_location("showme_check_links", path)
        if spec is None or spec.loader is None:
            raise HarnessError("无法构造 import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = root  # 让 --root 生效（原脚本按自身位置定 ROOT）
        broken = module.check()
    except Exception as exc:  # noqa: BLE001
        rep.skip("[links] 相对链接", f"导入 check_links.py 失败：{exc}")
        return
    pages = sum(1 for p in root.rglob("*.html")
                if ".git" not in p.parts
                and "templates" not in p.relative_to(root).parts[:-1])
    if not broken:
        rep.ok("[links] 相对链接", f"{pages} 个页面，全部可达")
        return
    rep.fail("[links] 相对链接", f"{len(broken)} 处失效")
    for page, url in broken[:8]:
        print(f"        └ {page} → {url}")
    if len(broken) > 8:
        print(f"        └ …另有 {len(broken) - 8} 处未列出")


# ---------------------------------------------------------------------------
# C. 布局
# ---------------------------------------------------------------------------
LAYOUT_PROBE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>layout probe</title></head>
<body style="margin:0">
<div id="frames"></div>
<pre id="out">PENDING</pre>
<script>
var TARGETS = __TARGETS__;
var results = {frames: {}, errors: []};
var pending = TARGETS.length;
var finished = false;

function desc(el, r) {
  var cls = "";
  if (typeof el.className === "string") {
    cls = el.className.trim().split(/\s+/).filter(Boolean).slice(0, 2).join(".");
  }
  return el.tagName.toLowerCase() + (cls ? "." + cls : "") + " right=" + Math.round(r.right);
}
function clipDesc(el) {
  var cls = "";
  if (typeof el.className === "string") {
    cls = el.className.trim().split(/\s+/).filter(Boolean).slice(0, 2).join(".");
  }
  return el.tagName.toLowerCase() + (cls ? "." + cls : "") +
    " sw=" + el.scrollWidth + " cw=" + el.clientWidth +
    " sh=" + el.scrollHeight + " ch=" + el.clientHeight;
}
function hiddenChain(el, d) {
  var p = el, depth = 0;
  while (p && p.nodeType === 1 && depth < 40) {
    var cs = d.defaultView.getComputedStyle(p);
    if (cs.display === "none" || cs.visibility === "hidden" ||
        parseFloat(cs.opacity || "1") < 0.05) return true;
    if (p === d.body) break;
    p = p.parentElement; depth++;
  }
  return false;
}
function inScroller(el, d) {
  var p = el.parentElement;
  while (p && p !== d.body) {
    var cs = d.defaultView.getComputedStyle(p);
    if (cs.overflowX === "auto" || cs.overflowX === "scroll") return true;
    p = p.parentElement;
  }
  return false;
}
function isLeafText(el) {
  if (el.children.length) return false;
  return (el.textContent || "").trim().length > 0;
}
function measure(d, W) {
  var de = d.documentElement;
  var cw = de.clientWidth;
  var out = {
    width: W, clientWidth: cw, scrollWidth: de.scrollWidth,
    overflow: [], overflowTotal: 0, clipped: [], clippedTotal: 0
  };
  var all = d.body ? d.body.querySelectorAll("*") : [];
  var i, el, r;
  for (i = 0; i < all.length; i++) {
    el = all[i];
    r = el.getBoundingClientRect();
    if (r.width <= 0) continue;
    if (r.right > cw + 1 && !inScroller(el, d) && !hiddenChain(el, d)) {
      out.overflowTotal++;
      if (out.overflow.length < 3) out.overflow.push(desc(el, r));
    }
  }
  if (W === 390) {
    for (i = 0; i < all.length; i++) {
      el = all[i];
      if (!isLeafText(el)) continue;
      r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      var cs = d.defaultView.getComputedStyle(el);
      if (cs.display === "inline" || cs.display === "contents") continue;
      var clamp = cs.webkitLineClamp || cs.lineClamp || "none";
      if (clamp !== "none") continue;  // line-clamp 是有意的截断，不算被切片
      if (cs.textOverflow === "ellipsis" && cs.whiteSpace === "nowrap") continue;
      if (hiddenChain(el, d)) continue;
      var ox = cs.overflowX, oy = cs.overflowY;
      var slicedX = (ox === "hidden" || ox === "clip") && el.scrollWidth > el.clientWidth + 1;
      var slicedY = (oy === "hidden" || oy === "clip") && el.scrollHeight > el.clientHeight + 1;
      if (slicedX || slicedY) {
        out.clippedTotal++;
        if (out.clipped.length < 3) out.clipped.push(clipDesc(el));
      }
    }
  }
  return out;
}
function onFrame(id, ok) {
  var target = null;
  for (var i = 0; i < TARGETS.length; i++) {
    if (TARGETS[i].id === id) target = TARGETS[i];
  }
  try {
    var f = document.getElementById(id);
    var d = f ? f.contentDocument : null;
    if (!ok || !d || !d.body) throw new Error("contentDocument 不可读（检查 --allow-file-access-from-files）");
    results.frames[id] = measure(d, target.width);
  } catch (e) {
    results.frames[id] = {error: String(e)};
  }
  pending--;
  if (pending === 0) setTimeout(finish, 40);
}
function finish() {
  if (finished) return;
  finished = true;
  document.getElementById("out").textContent = JSON.stringify(results);
}
TARGETS.forEach(function (t) {
  var f = document.createElement("iframe");
  f.id = t.id;
  f.style.width = t.width + "px";
  f.style.height = "900px";
  f.style.border = "0";
  f.style.display = "block";
  f.setAttribute("onload", "onFrame('" + t.id + "', true)");
  f.setAttribute("onerror", "onFrame('" + t.id + "', false)");
  f.src = t.src;
  document.getElementById("frames").appendChild(f);
});
setTimeout(function () {
  if (!finished) {
    results.errors.push("timeout: iframe 未在预算内完成加载");
    finish();
  }
}, 6000);
</script>
</body></html>
"""


def check_layout(root: Path, rep: Reporter, tmpdir: Path, chrome: str,
                 widths: tuple[int, ...]) -> None:
    rep.section("C. 布局 LAYOUT")
    pages = [root / "index.html"] + article_pages(root)
    pages = [p for p in pages if p.is_file()]
    width_label = "/".join(str(w) for w in widths)
    for page in pages:
        rp = rel(root, page)
        targets = [
            {"id": f"w{w}", "width": w, "src": page.resolve().as_uri()}
            for w in widths
        ]
        probe_html = LAYOUT_PROBE.replace("__TARGETS__", json.dumps(targets))
        try:
            data = run_chrome_probe(chrome, tmpdir, f"layout_{rp.replace('/', '_')}", probe_html)
        except ProbeError as exc:
            rep.fail(f"[layout] {rp} @{width_label}", f"探针失败：{exc}")
            continue
        frames = data.get("frames") or {}
        failed_widths: list[str] = []
        pass_widths: list[str] = []
        frame_390 = None
        for w in widths:
            frame = frames.get(f"w{w}") or {}
            if not frame or frame.get("error"):
                failed_widths.append(f"{w}px")
                reason = frame.get("error") if frame else "无测量结果"
                rep.fail(f"[layout] {rp} @{w}px", f"探针错误：{reason}")
                continue
            if w == 390:
                frame_390 = frame
            cw = frame.get("clientWidth")
            sw = frame.get("scrollWidth")
            overflow_total = frame.get("overflowTotal", 0)
            offenders = frame.get("overflow") or []
            problems = []
            if sw != cw:
                problems.append(f"scrollWidth={sw} > clientWidth={cw}")
            if overflow_total:
                shown = ", ".join(offenders)
                more = overflow_total - len(offenders)
                problems.append(f"溢出元素 {overflow_total} 个：{shown}" + (f"（另有 {more} 个）" if more else ""))
            if problems:
                failed_widths.append(f"{w}px")
                rep.fail(f"[layout] {rp} @{w}px", "；".join(problems))
            else:
                pass_widths.append(str(w))
        if len(pass_widths) == len(widths):
            rep.ok(f"[layout] {rp}", f"{width_label}: scrollWidth==clientWidth，溢出元素 0")
        # 390px 文字裁切
        if 390 not in widths:
            continue
        if frame_390 is None:
            rep.skip(f"[clip] {rp} @390", "390 探针失败，见上")
            continue
        clipped_total = frame_390.get("clippedTotal", 0)
        clipped = frame_390.get("clipped") or []
        if clipped_total:
            more = clipped_total - len(clipped)
            rep.fail(f"[clip] {rp} @390",
                     f"{clipped_total} 处文字被裁切：" + ", ".join(clipped) +
                     (f"（另有 {more} 处）" if more else ""))
        else:
            rep.ok(f"[clip] {rp} @390", "0 处文字被裁切（scroll* <= client*+1）")


# ---------------------------------------------------------------------------
# D. 门户交互
# ---------------------------------------------------------------------------
PORTAL_PROBE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>portal probe</title></head>
<body style="margin:0">
<div id="frames"></div>
<pre id="out">PENDING</pre>
<script>
var TERM = __TERM__;
var WIDTH = __WIDTH__;
var SRC_MAIN = __SRC_MAIN__;
var SRC_PRE = __SRC_PRE__;
var checks = [];
var pending = 2;
var finished = false;

var CARD_SEL = ".card";
var INPUT_SEL = "#sm-q, #q, input[type='search'], input[type='text']";
var DIALOG_SEL = "dialog, [role='dialog'], [aria-modal='true'], .sm-dialog, .sm-modal, .sm-k, .sm-palette, .sm-search-dialog, .sm-searchbox";
var TAB_SEL = "[role='tab'], [data-cat-tab], [data-cat], .sm-tab, .sm-cat, .cat-tab";
var TOGGLE_SEL = "[aria-expanded]";

function report(name, ok, detail) {
  checks.push({name: name, ok: !!ok, detail: String(detail || "")});
}
function sleep(ms) {
  return new Promise(function (resolve) { setTimeout(resolve, ms); });
}
function docOf(id) {
  var f = document.getElementById(id);
  try { return f ? f.contentDocument : null; } catch (e) { return null; }
}
function vis(d, el) {
  if (!el) return false;
  var cs = d.defaultView.getComputedStyle(el);
  if (cs.display === "none" || cs.visibility === "hidden") return false;
  if (parseFloat(cs.opacity || "1") < 0.05) return false;
  var r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}
function visibleCards(d) {
  var cards = d.querySelectorAll(CARD_SEL + ", [data-entry]");
  var n = 0;
  for (var i = 0; i < cards.length; i++) if (vis(d, cards[i])) n++;
  return n;
}
function visibleHits(d) {
  // 门户把「输入即搜」渲染成 #hits 的命中行（粒度到小节），同时隐藏卡片列表；
  // 老实现则直接隐藏卡片。两种都接受，但必须真的过滤到非空子集。
  var rows = d.querySelectorAll("#hits .hit, .hits .hit, .sm-hit, [data-hit]");
  var n = 0;
  for (var i = 0; i < rows.length; i++) if (vis(d, rows[i])) n++;
  return n;
}
function resultCount(d) {
  var hits = visibleHits(d);
  return hits > 0 ? hits : visibleCards(d);
}
function describe(el) {
  var cls = "";
  if (typeof el.className === "string") {
    cls = el.className.trim().split(/\s+/).filter(Boolean).slice(0, 2).join(".");
  }
  return el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (cls ? "." + cls : "");
}
function fireInput(d, input) {
  input.dispatchEvent(new d.defaultView.Event("input", {bubbles: true}));
  input.dispatchEvent(new d.defaultView.Event("change", {bubbles: true}));
}
function findInput(d) { return d.querySelector(INPUT_SEL); }

async function checkSearch(d) {
  var input = findInput(d);
  if (!input) { report("search-filter", false, "element not found: " + INPUT_SEL); return; }
  var cards = d.querySelectorAll(CARD_SEL + ", [data-entry]");
  if (!cards.length) { report("search-filter", false, "element not found: " + CARD_SEL); return; }
  var full = visibleCards(d);
  input.focus();
  input.value = TERM;
  fireInput(d, input);
  await sleep(350);
  var typedCards = visibleCards(d);
  var typedHits = visibleHits(d);
  var filtered = typedHits > 0 ? typedHits : typedCards;
  input.value = "";
  fireInput(d, input);
  await sleep(350);
  var restoredCards = visibleCards(d);
  var restoredHits = visibleHits(d);
  report("search-filter",
    full > 0 && filtered > 0 && filtered < full &&
    restoredCards === full && restoredHits === 0,
    "full=" + full + " cards=" + typedCards + " hits=" + typedHits +
    " restoredCards=" + restoredCards + " term=" + JSON.stringify(TERM));
}

function visibleDialog(d) {
  var els = d.querySelectorAll(DIALOG_SEL);
  var found = null;
  for (var i = 0; i < els.length; i++) if (vis(d, els[i])) found = els[i];
  return found;
}
function keyEvent(d, key, extra) {
  var opts = {
    key: key,
    code: key === "Escape" ? "Escape" : "KeyK",
    bubbles: true,
    cancelable: true
  };
  if (extra) for (var k in extra) opts[k] = extra[k];
  return new d.defaultView.KeyboardEvent("keydown", opts);
}
async function checkDialog(d) {
  d.body.dispatchEvent(keyEvent(d, "k", {metaKey: true, ctrlKey: true}));
  await sleep(300);
  var opened = visibleDialog(d);
  report("cmd-k", !!opened,
    opened ? "dialog visible: " + describe(opened)
           : "element not found: " + DIALOG_SEL + " after keydown k (metaKey+ctrlKey)");
  if (!opened) { report("esc-close", false, "dialog never opened — see cmd-k"); return; }
  // 真实用户按 Esc 时焦点在面板输入框里：keydown 先派给 activeElement，
  // 再依次退到面板内输入框 / 面板本身 / body / document。
  var targets = [];
  if (d.activeElement && d.activeElement !== d.body) targets.push(d.activeElement);
  var inner = opened.querySelector("input, textarea");
  if (inner && targets.indexOf(inner) < 0) targets.push(inner);
  targets.push(opened, d.body, d);
  for (var i = 0; i < targets.length; i++) {
    targets[i].dispatchEvent(keyEvent(d, "Escape"));
    await sleep(180);
    if (!visibleDialog(d)) break;
  }
  var still = visibleDialog(d);
  report("esc-close", !still, still ? "dialog still visible after Escape" : "dialog closed");
  if (still) {
    // 清理：后面还有页签/抽屉检查，别让遮罩挡着
    var closer = opened.querySelector("#kclose, .dlg-x, [data-close], button[aria-label*='close']");
    if (closer) { closer.click(); await sleep(150); }
  }
}

function checkPrefill(mainDoc, preDoc) {
  if (!preDoc || !preDoc.body) {
    report("prefill", false, "iframe failed to load index.html?q=<term>");
    return;
  }
  var input = findInput(preDoc);
  if (!input) { report("prefill", false, "element not found: " + INPUT_SEL + " (?q= page)"); return; }
  var full = visibleCards(mainDoc);
  var cnt = resultCount(preDoc);
  var got = input.value;
  report("prefill", got === TERM && cnt > 0 && cnt < full,
    "input=" + JSON.stringify(got) + " count=" + cnt +
    " cards=" + visibleCards(preDoc) + " hits=" + visibleHits(preDoc) +
    " full=" + full + " term=" + JSON.stringify(TERM));
}

function findTabs(d) {
  var out = [];
  var els = d.querySelectorAll(TAB_SEL);
  for (var i = 0; i < els.length; i++) {
    var el = els[i];
    if (!vis(d, el)) continue;
    var tag = el.tagName.toLowerCase();
    var looksTab = el.hasAttribute("role") || el.hasAttribute("data-cat-tab") ||
                   el.classList.contains("sm-tab") || el.classList.contains("sm-cat") ||
                   el.classList.contains("cat-tab");
    if (!looksTab && tag !== "button" && tag !== "a") continue;
    if (el.classList.contains("card")) continue;
    out.push(el);
  }
  return out;
}
async function checkTabs(d) {
  var tabs = findTabs(d);
  if (!tabs.length) { report("category-tab", false, "element not found: " + TAB_SEL); return; }
  var full = visibleCards(d);
  var tab = null;
  for (var i = 0; i < tabs.length; i++) {
    var txt = (tabs[i].textContent || "").trim();
    if (!/全部|所有|all/i.test(txt)) { tab = tabs[i]; break; }
  }
  if (!tab) tab = tabs[0];
  var badge = tab.querySelector(".sm-n, .sm-count, .sm-badge, .tn, .n, .count, [class*='badge'], sup");
  var numSrc = badge ? badge.textContent : tab.textContent;
  var nums = (numSrc || "").match(/\d+/g);
  if (!nums) {
    report("category-tab", false,
      "element not found: tab count badge in " + JSON.stringify((tab.textContent || "").trim().slice(0, 24)));
    return;
  }
  var expected = parseInt(nums[0], 10);
  tab.click();
  await sleep(350);
  var cnt = visibleCards(d);
  report("category-tab", cnt > 0 && cnt !== full && cnt === expected,
    "tab=" + JSON.stringify((tab.textContent || "").trim().replace(/\s+/g, " ").slice(0, 24)) +
    " badge=" + expected + " cards=" + cnt + " full=" + full);
  // 还原到「全部」，让后面的行内对齐检查看到整张网格
  for (var j = 0; j < tabs.length; j++) {
    if (/全部|所有|all/i.test((tabs[j].textContent || "").trim())) {
      tabs[j].click();
      await sleep(250);
      break;
    }
  }
}

function findToggle(d) {
  var els = d.querySelectorAll(TOGGLE_SEL);
  for (var i = 0; i < els.length; i++) {
    var el = els[i];
    var sig = ((el.id || "") + " " + (typeof el.className === "string" ? el.className : "") +
      " " + (el.getAttribute("aria-controls") || "") + " " +
      (el.getAttribute("aria-label") || "") + " " + (el.textContent || "")).toLowerCase();
    if (/tag|标签/.test(sig)) return el;
  }
  return els.length ? els[0] : null;
}
function drawerOf(d, btn) {
  var id = btn.getAttribute("aria-controls");
  if (id) { var t = d.getElementById(id); if (t) return t; }
  if (btn.nextElementSibling) return btn.nextElementSibling;
  return d.querySelector(".sm-tagbox, .tagbox, .sm-drawer, [class*='drawer']");
}
async function checkDrawer(d) {
  var btn = findToggle(d);
  if (!btn) { report("tag-drawer", false, "element not found: " + TOGGLE_SEL + " (标签 disclosure)"); return; }
  var drawer = drawerOf(d, btn);
  if (!drawer) {
    report("tag-drawer", false,
      "element not found: drawer（aria-controls / nextElementSibling / .tagbox）");
    return;
  }
  var before = btn.getAttribute("aria-expanded");
  btn.click();
  await sleep(450);
  var openedState = btn.getAttribute("aria-expanded");
  var openH = drawer.getBoundingClientRect().height;
  var openScroll = drawer.scrollHeight;
  var openOk = openedState !== before && openedState === "true" && openScroll > 0 && openH > 0;
  btn.click();
  await sleep(450);
  var closedState = btn.getAttribute("aria-expanded");
  var closedH = drawer.getBoundingClientRect().height;
  var closedOk = closedState === "false" && closedH < 2;
  report("tag-drawer", openOk && closedOk,
    "aria=" + before + "→" + openedState + "→" + closedState +
    " openH=" + Math.round(openH) + " scrollH=" + openScroll +
    " closedH=" + Math.round(closedH));
}

var COL_SEL = {
  h3: "h3",
  one: ".one",
  tags: ".tags, .tagrow, .sm-tags, [class*='tagrow']",
  foot: ".cfoot, .foot, footer, .sm-foot, [class*='card-foot']"
};
async function checkAlign(d) {
  var cards = [];
  var all = d.querySelectorAll(CARD_SEL + ", [data-entry]");
  for (var i = 0; i < all.length; i++) if (vis(d, all[i])) cards.push(all[i]);
  if (cards.length < 2) { report("align-row", false, "element not found: ≥2 张 .card"); return; }
  cards.sort(function (a, b) {
    return a.getBoundingClientRect().top - b.getBoundingClientRect().top;
  });
  var rows = [], cur = [], rowTop = null;
  for (var j = 0; j < cards.length; j++) {
    var top = cards[j].getBoundingClientRect().top;
    if (rowTop === null || Math.abs(top - rowTop) <= 2) {
      cur.push(cards[j]);
      if (rowTop === null) rowTop = top;
    } else {
      rows.push(cur); cur = [cards[j]]; rowTop = top;
    }
  }
  if (cur.length) rows.push(cur);
  var rowNo = 0;
  for (var r = 0; r < rows.length; r++) {
    if (rows[r].length < 2) continue;
    rowNo++;
    var keys = Object.keys(COL_SEL), parts = [], ok = true, missing = null;
    for (var c = 0; c < keys.length; c++) {
      var key = keys[c], vals = [];
      for (var k = 0; k < rows[r].length; k++) {
        var card = rows[r][k];
        var el = card.querySelector(COL_SEL[key]);
        if (!el) { missing = key + ": " + COL_SEL[key]; break; }
        vals.push(Math.round((el.getBoundingClientRect().top -
                              card.getBoundingClientRect().top) * 10) / 10);
      }
      if (missing) break;
      var distinct = [];
      for (var v = 0; v < vals.length; v++) if (distinct.indexOf(vals[v]) < 0) distinct.push(vals[v]);
      distinct.sort(function (a, b) { return a - b; });
      if (distinct[distinct.length - 1] - distinct[0] > 1) ok = false;
      parts.push(key + "={" + distinct.join(",") + "}");
    }
    if (missing) {
      report("align-row" + rowNo, false, "element not found: card " + missing);
      continue;
    }
    report("align-row" + rowNo, ok, "cards=" + rows[r].length + " " + parts.join(" "));
  }
  if (rowNo === 0) report("align-row", false, "no multi-card row at width " + WIDTH);
}

function injectProbeCss(d) {
  // 关掉过渡/动画：headless 的 virtual-time 不推进 CSS transition，
  // 抽屉会永远停在 max-height:0。关掉后读到的是动画结束后的最终布局。
  if (!d || !d.head || d.getElementById("__probe_css")) return;
  var st = d.createElement("style");
  st.id = "__probe_css";
  st.textContent = "*,*::before,*::after{transition-property:none!important;" +
    "transition-duration:0s!important;animation-duration:0s!important;animation-delay:0s!important}";
  d.head.appendChild(st);
}
function frameReady(id) {
  try {
    var f = document.getElementById(id);
    if (f && f.contentDocument) injectProbeCss(f.contentDocument);
  } catch (e) {}
  pending--;
  if (pending === 0) setTimeout(main, 80);
}
async function main() {
  var d = docOf("main"), p = docOf("pre");
  try {
    if (!d || !d.body) throw new Error("main iframe contentDocument 不可读");
    await checkSearch(d);
    await checkDialog(d);
    checkPrefill(d, p);
    await checkTabs(d);
    await checkDrawer(d);
    if (WIDTH >= 1000) await checkAlign(d);
  } catch (e) {
    report("harness", false, "JS error: " + (e && e.message ? e.message : e));
  }
  finish();
}
function finish() {
  if (finished) return;
  finished = true;
  document.getElementById("out").textContent =
    JSON.stringify({width: WIDTH, checks: checks});
}
function addFrame(id, src) {
  var f = document.createElement("iframe");
  f.id = id;
  f.style.width = WIDTH + "px";
  f.style.height = "900px";
  f.style.border = "0";
  f.style.display = "block";
  f.setAttribute("onload", "frameReady('" + id + "')");
  f.src = src;
  document.getElementById("frames").appendChild(f);
}
addFrame("main", SRC_MAIN);
addFrame("pre", SRC_PRE);
setTimeout(function () {
  if (!finished) {
    report("harness", false, "timeout: portal iframe 未在预算内加载完成");
    finish();
  }
}, 6000);
</script>
</body></html>
"""

PORTAL_EXPECTED = (
    ("search-filter", "搜索过滤"),
    ("cmd-k", "⌘K 打开面板"),
    ("esc-close", "Esc 关闭面板"),
    ("prefill", "?q= 预填"),
    ("category-tab", "分类页签"),
    ("tag-drawer", "标签抽屉"),
)


# ---------------------------------------------------------------------------
# 运行时挑选「只出现在正文里」的检索词
# ---------------------------------------------------------------------------
def balanced_json_at(text: str, start: int) -> str | None:
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_site_entries(portal_html: str) -> list[dict]:
    """从门户内联的 SITE/ENTRIES 数据里找带 sections 的 entry 列表。"""
    entries: list[dict] = []
    seen: set[int] = set()
    for m in re.finditer(r'"entries"\s*:', portal_html):
        candidates = []
        brace = portal_html.rfind("{", 0, m.start() + 1)
        candidates.append(brace)
        # 再往前找几个 {，兼容外层对象包着 SITE 的写法
        cursor = brace
        for _ in range(3):
            cursor = portal_html.rfind("{", 0, cursor)
            if cursor < 0:
                break
            candidates.append(cursor)
        for start in candidates:
            if start < 0 or start in seen:
                continue
            chunk = balanced_json_at(portal_html, start)
            if not chunk:
                continue
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                got = [e for e in data["entries"] if isinstance(e, dict)]
                if got and any(isinstance(e.get("sections"), list) and e["sections"] for e in got):
                    seen.add(start)
                    entries = got
    return entries


def entry_haystack(entry: dict) -> str:
    parts = [
        str(entry.get("title", "")),
        str(entry.get("one_liner", entry.get("one", ""))),
        str(entry.get("category", entry.get("cat", ""))),
        str(entry.get("catZh", "")),
    ]
    tags = entry.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    return " ".join(parts).lower()


def entry_body(entry: dict) -> str:
    chunks = []
    for sec in entry.get("sections") or []:
        if isinstance(sec, dict):
            chunks.append(str(sec.get("title", "")))
            chunks.append(str(sec.get("text", "")))
        else:
            chunks.append(str(sec))
    return " ".join(chunks)


def pick_term_from_site(entries: list[dict], portal_html: str) -> tuple[str, int] | None:
    head = " ".join(entry_haystack(e) for e in entries)
    bodies = [entry_body(e) for e in entries]
    page_hits: Counter[str] = Counter()
    ai_tokens = [set(ASCII_TOKEN_RE.findall(body)) for body in bodies]
    cjk_tokens = []
    for body in bodies:
        grams = set()
        for run in CJK_RUN_RE.findall(body):
            for i in range(len(run) - 2):
                grams.add(run[i:i + 3])
        cjk_tokens.append(grams)
    for toks in ai_tokens:
        page_hits.update(toks)
    all_text = " ".join(bodies).lower()
    # 优先选正文小节（而非「延伸阅读/自测」这类页尾）里的词，检索更有代表性
    tail_ids = {"read", "quiz", "faq"}
    tail_titles = {"延伸阅读", "自测", "自测题", "常见误解"}
    content_tokens: set[str] = set()
    for e in entries:
        for sec in e.get("sections") or []:
            if not isinstance(sec, dict):
                continue
            if str(sec.get("id", "")) in tail_ids or str(sec.get("title", "")) in tail_titles:
                continue
            content_tokens.update(ASCII_TOKEN_RE.findall(
                str(sec.get("title", "")) + " " + str(sec.get("text", ""))))

    def predicted(term: str) -> int:
        needle = term.lower()
        return sum(1 for idx, e in enumerate(entries)
                   if needle in entry_haystack(e) or needle in bodies[idx].lower())

    def viable(term: str) -> bool:
        needle = term.lower()
        # 注意：不能拿整份 portal_html 当禁用词表 —— SITE JSON 内联了 sections 正文，
        # 否则所有正文词都会被判成「出现在门户里」。这里只排除标题/标签/一句话。
        if needle in head or needle in STOPWORDS:
            return False
        if not (4 <= len(term) <= 28):
            return False
        hits = page_hits[term]
        if not (1 <= hits <= 3):
            return False
        if needle not in all_text:
            return False
        n = predicted(term)
        return 0 < n < len(entries)

    # 既要可读（4–28 字符），又要好输入：优先正文小节，其次 6–20 字符、接近 10 字符
    def rank(t: str) -> tuple:
        length = len(t)
        return (page_hits[t], 0 if t in content_tokens else 1,
                0 if 6 <= length <= 20 else 1, abs(length - 10), t)

    for tok in sorted(page_hits, key=rank):
        if viable(tok):
            return tok, predicted(tok)

    cjk_hits: Counter[str] = Counter()
    for toks in cjk_tokens:
        cjk_hits.update(toks)
    for tok in sorted(cjk_hits, key=lambda t: (cjk_hits[t], t)):
        if 1 <= cjk_hits[tok] <= 3 and tok not in head:
            n = predicted(tok)
            if 0 < n < len(entries):
                return tok, n
    return None


def strip_html_text(html: str) -> str:
    html = re.sub(r"<script\b.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style\b.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return html_mod.unescape(html)


def pick_term_fallback(root: Path, portal_html: str) -> tuple[str, None] | None:
    heads = []
    for meta_path in gather_metas(root):
        try:
            data = json.loads(read_text(meta_path))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            heads.append(entry_haystack(data))
            heads.append(str(data.get("title", "")))
            heads.append(str(data.get("one_liner", "")))
            heads.append(" ".join(str(t) for t in (data.get("tags") or [])))
    head = " ".join(heads).lower()
    # 门户自身 UI 文案（不含 <script> 里的内联数据）也不该被选中
    forbidden = strip_html_text(portal_html).lower()
    page_texts = []
    for page in article_pages(root):
        text = strip_html_text(read_text(page))
        page_texts.append(text)
    page_hits: Counter[str] = Counter()
    for text in page_texts:
        page_hits.update(set(ASCII_TOKEN_RE.findall(text)))
    ranked = [t for t in page_hits
              if 4 <= len(t) <= 28 and 1 <= page_hits[t] <= 3
              and t.lower() not in head and t.lower() not in forbidden
              and t.lower() not in STOPWORDS]
    for tok in sorted(ranked, key=lambda t: (page_hits[t], 0 if 6 <= len(t) <= 20 else 1,
                                             abs(len(t) - 10), t)):
        return tok, None
    cjk_hits: Counter[str] = Counter()
    for text in page_texts:
        grams = set()
        for run in CJK_RUN_RE.findall(text):
            for i in range(len(run) - 2):
                grams.add(run[i:i + 3])
        cjk_hits.update(grams)
    for tok in sorted(cjk_hits, key=lambda t: (cjk_hits[t], t)):
        if 1 <= cjk_hits[tok] <= 3 and tok not in head and tok not in forbidden:
            return tok, None
    return None


def pick_search_term(root: Path) -> tuple[str | None, int | None, str, str]:
    portal_path = root / "index.html"
    if not portal_path.is_file():
        return None, None, "none", "index.html 不存在"
    portal_html = read_text(portal_path)
    entries = extract_site_entries(portal_html)
    if entries:
        picked = pick_term_from_site(entries, portal_html)
        if picked:
            term, predicted = picked
            return term, predicted, "portal-SITE", f"仅出现在 {predicted} 篇正文的小节文本中"
        return None, None, "portal-SITE", "SITE entries 里找不到只在正文出现的词"
    picked = pick_term_fallback(root, portal_html)
    if picked:
        term, _ = picked
        return term, None, "article-body", "从文章正文提取（门户尚未内联 SITE，无法预测命中数）"
    return None, None, "none", "无法从任何页面提取正文限定词"


def check_portal(root: Path, rep: Reporter, tmpdir: Path, chrome: str) -> None:
    rep.section("D. 门户交互 PORTAL")
    portal = root / "index.html"
    if not portal.is_file():
        rep.fail("[portal] index.html", "文件不存在")
        return
    term, predicted, source, why = pick_search_term(root)
    if not term:
        rep.info("[term] 检索词", f"未取到（{source}）：{why}")
    else:
        extra = f"predicted={predicted}" if predicted is not None else "predicted=?"
        rep.info("[term] 检索词", f"{term!r} source={source} {extra}  {why}")

    widths = QUICK_WIDTHS
    for width in widths:
        src_main = portal.resolve().as_uri()
        src_pre = src_main + "?q=" + quote(term or "")
        probe = (PORTAL_PROBE
                 .replace("__TERM__", json.dumps(term or ""))
                 .replace("__WIDTH__", str(width))
                 .replace("__SRC_MAIN__", json.dumps(src_main))
                 .replace("__SRC_PRE__", json.dumps(src_pre)))
        try:
            data = run_chrome_probe(chrome, tmpdir, f"portal_{width}", probe)
            results = data.get("checks") or []
        except ProbeError as exc:
            for name, _label in PORTAL_EXPECTED:
                if name == "prefill" and not term:
                    rep.fail(f"[portal:{width}] {name}", f"无法运行：没有检索词（{exc}）")
                else:
                    rep.fail(f"[portal:{width}] {name}", f"探针失败：{exc}")
            if width >= 1000:
                rep.fail(f"[portal:{width}] align-row", f"探针失败：{exc}")
            continue

        by_name = {}
        align_rows = []
        for item in results:
            name = str(item.get("name", ""))
            if name.startswith("align-row"):
                align_rows.append(item)
            by_name[name] = item
        harness = by_name.get("harness")
        if harness and not harness.get("ok"):
            rep.fail(f"[portal:{width}] harness", str(harness.get("detail", "")))

        for name, _label in PORTAL_EXPECTED:
            item = by_name.get(name)
            label = f"[portal:{width}] {name}"
            if item is None:
                rep.fail(label, "probe 未返回该检查（见 harness）")
            elif item.get("ok"):
                rep.ok(label, str(item.get("detail", "")))
            else:
                detail = str(item.get("detail", ""))
                if not term and name in ("search-filter", "prefill"):
                    detail = "没有取到正文限定检索词，无法验证搜索（见 INFO [term]）"
                rep.fail(label, detail)

        if width >= 1000:
            if not align_rows:
                rep.fail(f"[portal:{width}] align-row",
                         "element not found: 同一行 ≥2 张卡片（或探针中止）")
            for item in align_rows:
                label = f"[portal:{width}] {item.get('name')}"
                if item.get("ok"):
                    rep.ok(label, str(item.get("detail", "")))
                else:
                    rep.fail(label, str(item.get("detail", "")))
        else:
            rep.skip(f"[portal:{width}] align-row", "desktop only（1440 测量）")

        # 正文限定词必须真的只在正文：把 term 是否命中标题/标签的预测打印出来
        if term and predicted is not None:
            rep.info(f"[term:{width}] 预测", f"{predicted} 张卡片应命中 {term!r}")


# ---------------------------------------------------------------------------
# E. 回归守卫
# ---------------------------------------------------------------------------
def remote_offenders(html: str) -> list[str]:
    out: list[str] = []
    for tag in LINK_TAG_RE.findall(html):
        if not REMOTE_LINK_HREF_RE.search(tag):
            continue
        rel_m = re.search(r'rel\s*=\s*["\']([^"\']*)["\']', tag, re.I)
        rels = rel_m.group(1).lower().split() if rel_m else []
        if rels and not any(r in LINK_LOADING_RELS for r in rels):
            continue  # canonical / alternate / author … 不加载远程资源
        out.append(" ".join(tag.split())[:100])
    if SRC_REMOTE_RE.search(html):
        m = SRC_REMOTE_RE.search(html)
        out.append(html[max(0, m.start() - 20):m.end() + 60].replace("\n", " ")[:110])
    if CSS_IMPORT_REMOTE_RE.search(html):
        out.append("@import 指向远程样式")
    if CSS_URL_REMOTE_RE.search(html):
        m = CSS_URL_REMOTE_RE.search(html)
        out.append(html[max(0, m.start() - 10):m.end() + 50].replace("\n", " ")[:110])
    return out


def check_regression(root: Path, rep: Reporter) -> None:
    rep.section("E. 回归守卫 REGRESSION")
    html_files = sorted(p for p in root.rglob("*.html") if ".git" not in p.parts)
    if not html_files:
        rep.fail("[offline] 离线优先", "没有任何 HTML 页面")
    else:
        offenders: list[str] = []
        for path in html_files:
            for item in remote_offenders(read_text(path)):
                offenders.append(f"{rel(root, path)}: {item}")
        if offenders:
            rep.fail("[offline] 离线优先", f"{len(offenders)} 处远程资源引用")
            for item in offenders[:8]:
                print(f"        └ {item}")
        else:
            rep.ok("[offline] 离线优先",
                   f"{len(html_files)} 个页面无远程 <link>/script/img/@import/url()")

    pages = article_pages(root)
    if not pages:
        rep.fail("[regress] 原 footer 与 h1", "没有文章页")
        return
    footer_fail: list[str] = []
    for page in pages:
        html = read_text(page)
        has_footer = re.search(r"<footer\b", html, re.I) is not None or "由" in html
        h1_count = len(re.findall(r"<h1\b", html, re.I))
        if not has_footer or h1_count != 1:
            reasons = []
            if not has_footer:
                reasons.append("footer/「由」缺失")
            if h1_count != 1:
                reasons.append(f"h1={h1_count}（期望 1）")
            footer_fail.append(f"{rel(root, page)}: " + "，".join(reasons))
    if footer_fail:
        rep.fail("[regress] 原 footer 与 h1 未被外壳破坏",
                 f"{len(pages) - len(footer_fail)}/{len(pages)} 页合格")
        for item in footer_fail[:11]:
            print(f"        └ {item}")
    else:
        rep.ok("[regress] 原 footer 与 h1 未被外壳破坏",
               f"{len(pages)}/{len(pages)} 页：footer 保留，h1 唯一")

    toc_fail: list[str] = []
    for page in pages:
        html = read_text(page)
        nav_count = count_class(html, "nav")
        toc_count = count_class(html, "sm-toc")
        if nav_count != 1 or toc_count != 0:
            toc_fail.append(f"{rel(root, page)}: .nav={nav_count}（期望 1），.sm-toc={toc_count}（期望 0）")
    if toc_fail:
        rep.fail("[regress] 页内 TOC 不重复", f"{len(pages) - len(toc_fail)}/{len(pages)} 页合格")
        for item in toc_fail[:11]:
            print(f"        └ {item}")
    else:
        rep.ok("[regress] 页内 TOC 不重复",
               f"{len(pages)}/{len(pages)} 页：各 1 个 .nav，无 .sm-toc")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="show-me 站点验收探针（DESIGN.md §5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=None,
                        help="仓库根目录（默认：本脚本所在目录）")
    parser.add_argument("--quick", action="store_true",
                        help="跳过 560/640/900/1280 布局档与两次重建幂等")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    if not root.is_dir():
        print(f"HARNESS ERROR: --root 不是目录：{root}", file=sys.stderr)
        return 2
    if not (root / "index.html").is_file():
        print(f"HARNESS ERROR: {root} 下没有 index.html", file=sys.stderr)
        return 2
    chrome = find_chrome()
    if not chrome:
        print("HARNESS ERROR: 找不到 Chrome/Chromium 可执行文件。", file=sys.stderr)
        print("  设置 CHROME=/path/to/chrome 后重试。", file=sys.stderr)
        return 2

    widths = QUICK_WIDTHS if args.quick else FULL_WIDTHS
    print(f"show-me verify_site — root={root}")
    print(f"chrome={chrome}")
    print(f"mode={'quick' if args.quick else 'full'}  widths={'/'.join(map(str, widths))}")

    rep = Reporter()
    tmpdir = Path(tempfile.mkdtemp(prefix="showme-verify-"))
    try:
        check_meta(root, rep)
        check_shell_markers(root, rep)
        check_idempotent(root, rep, tmpdir, args.quick)
        check_links(root, rep)
        check_layout(root, rep, tmpdir, chrome, widths)
        check_portal(root, rep, tmpdir, chrome)
        check_regression(root, rep)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if rep.skips:
        print(f"SKIP {rep.skips}")
    print(f"PASS {rep.passes} / FAIL {rep.fails}")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(2)
    except HarnessError as exc:
        print(f"HARNESS ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 - 探针自身崩溃必须与站点 FAIL 区分开
        import traceback
        traceback.print_exc()
        print(f"HARNESS ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
