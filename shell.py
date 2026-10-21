#!/usr/bin/env python3
"""文章页站点外壳（shell）注入器 —— build_index.py 的唯一入口是 inject(html, ctx)。

    import shell
    page = shell.inject(page, {
        "entry":    e,            # entry dict，见 DESIGN.md §3
        "siblings": [...],        # 同分类，日期新→旧，含当前篇
        "prev":     p or None,    # 全局阅读顺序上的上一篇
        "next":     n or None,
        "built":    "2026-10-21",
    })

四条硬约束：

1. **幂等**：注入前先 `_strip()` 清掉全部 5 个外壳段；每段两侧不写任何空白，
   所以 `_strip(inject(x)) == x`，连跑两次字节一致。
2. **只增不改**：Meta / 样式进 `<head>`，吸顶条与侧栏是 `<body>` 下 fixed 的直接子元素
   （不参与流），页尾导航插在 `.wrap` 内、原 `<footer>` 之前。
   `.wrap` 的 980px 宽度与位置一像素不动，`<footer>` 正文一个字节不动。
3. **颜色只来自 theme.py**（TOKENS_CSS / cat_color / cat_zh），本文件不新增色号。
4. **无网络**：无外链字体、无 CDN、无请求；内联 JS 全部包在 try/catch 里。

外壳的 5 个段（顺序即页面里的顺序）：

    meta 块 → <style id="sm-shell"> → 吸顶条 → 分类侧栏 → 页尾导航

每段形如 `<!-- show-me:shell:begin -->…<!-- show-me:shell:end -->`，互不嵌套。
"""
from __future__ import annotations

import html as _html
import re

import theme

BEGIN = "<!-- show-me:shell:begin -->"
END = "<!-- show-me:shell:end -->"

# 文章页的站点根（GitHub Pages 子路径）；canonical / og:url 用它
SITE_URL = "https://qqtang-code.github.io/show-me"

# 所有已注入的段（含其间的换行）；`.*?` 保证只吃成对的 begin…end
_REGION_RE = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
_DESCRIPTION_RE = re.compile(r'<meta\s+name=["\']description["\'][^>]*>', re.I)
_TITLE_END_RE = re.compile(r"</title\s*>", re.I)
_HEAD_END_RE = re.compile(r"</head\s*>", re.I)
_BODY_OPEN_RE = re.compile(r"<body[^>]*>", re.I)
_FOOTER_RE = re.compile(r"^[ \t]*<footer\b", re.M)
_BODY_END_RE = re.compile(r"</body\s*>", re.I)


# ────────────────────────────── 基础工具 ──────────────────────────────

def _strip(html: str) -> str:
    """清掉所有已注入的外壳段。这是幂等的地基：先剥后插，永不叠加。"""
    return _REGION_RE.sub("", html)


def _esc(value) -> str:
    """文本 / 属性统一转义（标题里有 `<`、`>`、`&`、`--prefix`）。"""
    return _html.escape(str(value if value is not None else ""), quote=True)


def _region(content: str) -> str:
    """把一个段包进成对标记；两端不留空白，保证 strip 后原样还原。"""
    return BEGIN + "\n" + content.strip("\n") + "\n" + END


def _region_or_nothing(content: str) -> str:
    return _region(content) if content.strip() else ""


def _root(entry: dict) -> str:
    root = entry.get("_root") or "../../"
    return root if root.endswith("/") else root + "/"


def _color(entry: dict) -> str:
    """分类色：优先 entry['_color']，否则 theme.cat_color()；非法值回落主题默认。"""
    raw = entry.get("_color") or theme.cat_color(str(entry.get("category") or ""))
    return raw if isinstance(raw, str) and re.fullmatch(r"#[0-9a-fA-F]{3,8}", raw) else theme.DEFAULT_COLOR


def _cat_zh(entry: dict) -> str:
    return str(entry.get("_cat_zh") or theme.cat_zh(str(entry.get("category") or "")))


def _minutes(entry: dict) -> int:
    try:
        return max(0, int(entry.get("_minutes") or 0))
    except (TypeError, ValueError):
        return 0


def _min_label(entry: dict) -> str:
    n = _minutes(entry)
    return f"{n} 分钟" if n else ""


def _date(entry: dict) -> str:
    return str(entry.get("date") or "")


def _page_href(entry: dict, link_entry: dict) -> str:
    """从当前页出发指向另一篇的地址：`../../<category>/<slug>/index.html`。"""
    folder = str(link_entry.get("_dir") or "").strip("/")
    artifact = str(link_entry.get("artifact") or "index.html")
    return _root(entry) + (folder + "/" if folder else "") + artifact


def _is_current(entry: dict, other: dict) -> bool:
    a, b = entry.get("_dir"), other.get("_dir")
    if a and b:
        return a == b
    return entry.get("slug") == other.get("slug") and entry.get("category") == other.get("category")


def _insert_at(html: str, index: int, chunk: str) -> str:
    return html[:index] + chunk + html[index:] if chunk else html


# ────────────────────────────── 内联图标 ──────────────────────────────

_ICON_SEARCH = (
    '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
    '<circle cx="7" cy="7" r="4.2"/><path d="M10.4 10.4 14 14"/></svg>'
)
_ICON_LEFT = (
    '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
    '<path d="M10 3 5 8l5 5"/></svg>'
)
_ICON_RIGHT = (
    '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
    '<path d="M6 3l5 5-5 5"/></svg>'
)


# ────────────────────────────── ① meta 块 ──────────────────────────────

def _meta_tags(entry: dict, *, with_description: bool) -> str:
    one = str(entry.get("one_liner") or entry.get("_lead") or "").strip()
    title = str(entry.get("title") or entry.get("_h1") or "").strip()
    url = f"{SITE_URL}/{str(entry.get('_dir') or '').strip('/')}/{str(entry.get('artifact') or 'index.html')}"
    tags = []
    if with_description and one:
        tags.append(f'<meta name="description" content="{_esc(one)}">')
    tags += [
        '<meta property="og:type" content="article">',
        f'<meta property="og:title" content="{_esc(title)}">',
        f'<meta property="og:description" content="{_esc(one)}">',
        f'<meta property="og:url" content="{_esc(url)}">',
        f'<link rel="canonical" href="{_esc(url)}">',
        '<meta name="twitter:card" content="summary">',
    ]
    return "\n".join(tags)


# ────────────────────────────── ② 样式块 ──────────────────────────────

SHELL_CSS = """  /* ════════ show-me 文章页外壳 · 由 shell.py 注入，勿手改 ════════ */

  /* 11 页的 body 都是 padding:40px 20px 80px。吸顶条高 var(--bar-h)，
     差多少补多少，否则它会压住第一行 .crumb；除此之外正文几何不动。 */
  body::before{content:"";display:block;height:max(0px,calc(var(--bar-h) - 40px))}
  /* 页内锚点（.nav 的 #s1 …）落到吸顶条下方，而不是被它裁掉 */
  [id]{scroll-margin-top:calc(20px + var(--bar-h))}

  /* ---------- 吸顶条 ---------- */
  .sm-bar{position:fixed;top:0;left:0;right:0;z-index:60;height:var(--bar-h);
    background:rgba(245,247,251,.84);
    -webkit-backdrop-filter:saturate(150%) blur(12px);backdrop-filter:saturate(150%) blur(12px);
    border-bottom:1px solid var(--line)}
  .sm-bar-in{height:100%;max-width:1020px;margin:0 auto;padding:0 20px;
    display:flex;align-items:center;gap:10px;min-width:0}
  .sm-brand{flex:none;font-size:14px;font-weight:650;letter-spacing:-.015em;
    color:var(--ink);text-decoration:none}
  .sm-brand:hover{color:var(--accent)}
  .sm-div{flex:none;width:1px;height:16px;background:var(--line)}
  .sm-cat{display:inline-flex;align-items:center;gap:6px;min-width:0;max-width:190px;
    font:11.5px var(--mono);color:var(--dim);text-decoration:none;background:var(--panel);
    border:1px solid var(--line);border-radius:999px;padding:3px 10px 3px 8px}
  .sm-cat:hover{color:var(--ink);border-color:var(--line2);background:var(--panel2)}
  .sm-cat>span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .sm-dot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--sm-cat,var(--accent))}
  .sm-right{margin-left:auto;display:flex;align-items:center;gap:10px;min-width:0}
  .sm-min,.sm-date{font:11.5px var(--mono);white-space:nowrap}
  .sm-min{color:var(--dim)}
  .sm-date{color:var(--dim2)}
  .sm-sep{font:10px var(--mono);color:#c3ccdb}
  .sm-search{flex:none;display:inline-flex;align-items:center;gap:6px;
    font:11.5px var(--mono);color:var(--dim);text-decoration:none;background:var(--panel);
    border:1px solid var(--line);border-radius:8px;padding:5px 9px}
  .sm-search:hover{color:var(--accent);border-color:var(--accent-line);background:var(--accent-soft)}
  .sm-search svg{width:13px;height:13px;fill:none;stroke:currentColor;stroke-width:1.5;
    stroke-linecap:round}
  .sm-kbd{font:10px var(--mono);line-height:15px;color:var(--dim2);background:var(--chip);
    border:1px solid var(--chipb);border-radius:4px;padding:0 4px}
  .sm-prog{position:absolute;left:0;bottom:0;width:100%;height:2px;background:var(--accent);
    opacity:.85;pointer-events:none;transform:scaleX(0);transform-origin:0 50%;
    transition:transform .12s linear}
  @media (prefers-reduced-motion:reduce){.sm-prog{transition:none}}

  /* ---------- 分类侧栏：只在 ≥1280px 的左侧空白边栏里 ---------- */
  :root{--sm-rail-w:clamp(120px,calc(50vw - 512px),208px)}
  .sm-rail{display:none}
  @media (min-width:1280px){
    .sm-rail{display:block;position:fixed;z-index:55;pointer-events:auto;
      top:calc(var(--bar-h) + 24px);left:max(6px,calc(50% - 490px - var(--sm-rail-w) - 12px));
      width:var(--sm-rail-w);
      max-height:calc(100vh - var(--bar-h) - 48px);overflow-y:auto;overscroll-behavior:contain;
      scrollbar-width:thin;padding:12px 9px;
      background:rgba(255,255,255,.7);border:1px solid var(--line2);border-radius:12px;
      box-shadow:var(--shadow-1)}
  }
  .sm-rail-h{font:10px var(--mono);letter-spacing:.07em;color:var(--dim2);
    padding:0 5px 8px;margin-bottom:7px;border-bottom:1px solid var(--line2);white-space:nowrap}
  .sm-rail-list{display:flex;flex-direction:column;gap:1px}
  .sm-rail-a{display:grid;grid-template-columns:7px minmax(0,1fr);column-gap:7px;
    align-items:start;min-width:0;padding:6px 5px;border-radius:7px;
    text-decoration:none;color:var(--dim)}
  .sm-rail-a:hover{background:var(--chip);color:var(--ink)}
  .sm-rail-a[aria-current]{background:var(--accent-soft);color:var(--ink)}
  .sm-rail-d{width:6px;height:6px;margin-top:5px;border-radius:50%;opacity:.55;
    background:var(--sm-cat,var(--accent))}
  .sm-rail-a[aria-current] .sm-rail-d{opacity:1}
  .sm-rail-t{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
    overflow:hidden;font-size:11.5px;line-height:1.45;overflow-wrap:anywhere}
  .sm-rail-a[aria-current] .sm-rail-t{font-weight:600}
  .sm-rail-m{grid-column:2;margin-top:2px;font:9.5px var(--mono);color:var(--dim2)}

  /* ---------- 页尾导航（在 .wrap 内，跟着 980px 量宽） ---------- */
  .sm-foot{margin:54px 0 0;padding-top:18px;border-top:1px solid var(--line)}
  .sm-pn{display:flex;gap:16px;justify-content:space-between;align-items:flex-start}
  .sm-pn-a{display:flex;align-items:flex-start;gap:9px;min-width:0;max-width:48%;
    text-decoration:none;color:var(--dim)}
  .sm-pn-a svg{flex:none;width:14px;height:14px;margin-top:3px;fill:none;stroke:#b0bacb;
    stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
  .sm-pn-a:hover{color:var(--accent)}
  .sm-pn-a:hover svg{stroke:var(--accent)}
  .sm-pn-b{display:block;min-width:0}
  .sm-pn-l{display:flex;align-items:center;gap:6px;margin-bottom:3px;
    font:10px var(--mono);letter-spacing:.06em;color:var(--dim2)}
  .sm-pn-t{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
    font-size:13px;line-height:1.5;color:var(--ink);overflow-wrap:anywhere}
  .sm-next{flex-direction:row-reverse;margin-left:auto;text-align:right}
  .sm-next .sm-pn-b{text-align:right}
  .sm-next .sm-pn-l{justify-content:flex-end}
  .sm-more{margin:26px 0 0}
  .sm-more-h{display:block;margin:0 0 10px;padding:0;border:0;
    font:600 11px var(--mono);letter-spacing:.06em;color:var(--dim2)}
  .sm-more-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
  /* 同类条目为奇数时，最后一张横跨两列，不留半个空格子 */
  .sm-more-grid>a:last-child:nth-child(odd){grid-column:1/-1}
  .sm-more-a{display:block;min-width:0;padding:10px 12px;text-decoration:none;
    background:var(--panel);border:1px solid var(--line);border-radius:9px}
  .sm-more-a:hover{background:var(--panel2);border-color:var(--accent-line);box-shadow:var(--shadow-1)}
  .sm-more-t{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
    font-size:12.5px;line-height:1.5;color:var(--ink);overflow-wrap:anywhere}
  .sm-more-m{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:6px;
    font:10px var(--mono);color:var(--dim2)}
  .sm-more-m i{width:3px;height:3px;border-radius:50%;background:#c3ccdb}

  /* ---------- 阶梯：<1280 没有边栏可用；<900 收窄；<640 单列 ---------- */
  @media (max-width:1279px){
    .sm-bar-in{gap:8px}
    .sm-cat{max-width:160px}
  }
  @media (max-width:899px){
    .sm-date,.sm-sep{display:none}
  }
  @media (max-width:760px){
    .sm-more-grid{grid-template-columns:minmax(0,1fr)}
    .sm-more-grid>a:last-child:nth-child(odd){grid-column:auto}
  }
  @media (max-width:640px){
    .sm-bar-in{gap:8px;padding:0 14px}
    .sm-brand{font-size:13px}
    .sm-cat{max-width:118px;padding:2px 8px 2px 7px;font-size:11px}
    .sm-min{display:none}
    .sm-search{padding:5px 7px}
    .sm-kbd,.sm-search-t{display:none}
    .sm-foot{margin-top:40px}
    .sm-pn{flex-direction:column;gap:12px}
    .sm-pn-a{max-width:100%}
    .sm-next{margin-left:0;justify-content:flex-end}
    .sm-next .sm-pn-b{text-align:left}
    .sm-next .sm-pn-l{justify-content:flex-start}
  }

  /* 表格在 601–820px 之间的空档：各页自己的「卡片模式」断点有 600 也有 640，
     中间这一带表格会按 min-content 撑开并把整页推出视口（实测 torchrun 的参数速查表
     自然宽 658px，640px 视口下整页横向平移）。让表格自己横向滚动，不动列宽、
     也不让 td.v / td.k 的标识符折行（试过 white-space:normal，`LocalElasticAgent`
     会被切成 `LocalEla sticAgent`）。 */
  @media (min-width:601px) and (max-width:820px){
    .wrap table{display:block;overflow-x:auto;max-width:100%}
  }"""

# 滚动进度 + 搜索按钮。短页面 m<=0 时进度恒为 0，不报错。
# 搜索跳 `index.html?q=`：查询串写在 data-q 里、由这里拼到地址栏，
# 因为 check_links.py 只 strip `#` 不 strip `?`，写成 href 会被它判成失效链接。
_PROGRESS_JS = """<script>
try{(function(){var b=document.getElementById('sm-progress'),
u=function(){var d=document.documentElement,m=d.scrollHeight-d.clientHeight,
p=m>0?Math.min(1,Math.max(0,d.scrollTop/m)):0;if(b)b.style.transform='scaleX('+p+')'};
addEventListener('scroll',u,{passive:true});addEventListener('resize',u);u();
var s=document.getElementById('sm-search'),
g=function(){if(s)location.href=s.getAttribute('data-q')||s.href};
if(s)s.addEventListener('click',function(e){if(e.button||e.metaKey||e.ctrlKey||e.shiftKey)return;
e.preventDefault();g()});
addEventListener('keydown',function(e){if((e.metaKey||e.ctrlKey)&&(e.key||'').toLowerCase()==='k'){
e.preventDefault();g()}})})()}catch(e){}
</script>"""


# ────────────────────────────── ③ 吸顶条 ──────────────────────────────

def _bar(entry: dict) -> str:
    root, cat = _root(entry), str(entry.get("category") or "")
    color, cat_zh = _color(entry), _cat_zh(entry)
    minutes, date = _min_label(entry), _date(entry)

    meta_bits = []
    if minutes:
        meta_bits.append(f'<span class="sm-min">{_esc(minutes)}</span>')
    if date:
        if meta_bits:
            meta_bits.append('<span class="sm-sep" aria-hidden="true">·</span>')
        meta_bits.append(f'<span class="sm-date">{_esc(date)}</span>')

    return (
        '<div class="sm-bar" id="sm-topbar">'
        '<div class="sm-bar-in">'
        f'<a class="sm-brand" href="{_esc(root)}index.html">show-me</a>'
        '<i class="sm-div" aria-hidden="true"></i>'
        f'<a class="sm-cat" href="{_esc(root)}index.html#{_esc(cat)}"'
        f' style="--sm-cat:{_esc(color)}" title="{_esc(cat_zh)}">'
        '<i class="sm-dot" aria-hidden="true"></i>'
        f"<span>{_esc(cat_zh)}</span></a>"
        '<div class="sm-right">'
        + "".join(meta_bits) +
        f'<a class="sm-search" id="sm-search" href="{_esc(root)}index.html"'
        f' data-q="{_esc(root)}index.html?q=" aria-label="在全库中搜索（⌘K）">'
        f'{_ICON_SEARCH}<span class="sm-search-t">搜索</span>'
        '<kbd class="sm-kbd">⌘K</kbd></a>'
        "</div></div>"
        '<div class="sm-prog" id="sm-progress" aria-hidden="true"></div>'
        "</div>"
        + _PROGRESS_JS
    )


# ────────────────────────────── ④ 分类侧栏 ──────────────────────────────

def _rail(entry: dict, siblings: list[dict]) -> str:
    if not siblings:
        return ""
    rows = []
    for sib in siblings:
        current = _is_current(entry, sib)
        title = str(sib.get("title") or sib.get("_h1") or sib.get("slug") or "")
        minutes = _min_label(sib)
        rows.append(
            f'<a class="sm-rail-a" href="{_esc(_page_href(entry, sib))}"'
            f' style="--sm-cat:{_esc(_color(sib))}"'
            + (' aria-current="page"' if current else "") + ">"
            '<i class="sm-rail-d" aria-hidden="true"></i>'
            f'<span class="sm-rail-t">{_esc(title)}</span>'
            + (f'<span class="sm-rail-m">{_esc(minutes)}</span>' if minutes else "")
            + "</a>"
        )
    return (
        '<nav class="sm-rail" aria-label="同分类条目">'
        f'<div class="sm-rail-h">同分类 · {len(siblings)} 篇</div>'
        '<div class="sm-rail-list">' + "".join(rows) + "</div></nav>"
    )


# ────────────────────────────── ⑤ 页尾导航 ──────────────────────────────

def _prev_next(entry: dict, prev: dict | None, nxt: dict | None) -> str:
    out = []
    if prev:
        out.append(
            f'<a class="sm-pn-a sm-prev" href="{_esc(_page_href(entry, prev))}">'
            f"{_ICON_LEFT}<span class=\"sm-pn-b\"><span class=\"sm-pn-l\">"
            f'<i class="sm-dot" style="--sm-cat:{_esc(_color(prev))}" aria-hidden="true"></i>'
            f'上一篇</span><b class="sm-pn-t">{_esc(prev.get("title") or prev.get("_h1") or "")}</b>'
            "</span></a>"
        )
    if nxt:
        out.append(
            f'<a class="sm-pn-a sm-next" href="{_esc(_page_href(entry, nxt))}">'
            f"{_ICON_RIGHT}<span class=\"sm-pn-b\"><span class=\"sm-pn-l\">"
            f'<i class="sm-dot" style="--sm-cat:{_esc(_color(nxt))}" aria-hidden="true"></i>'
            f'下一篇</span><b class="sm-pn-t">{_esc(nxt.get("title") or nxt.get("_h1") or "")}</b>'
            "</span></a>"
        )
    return f'<div class="sm-pn">{"".join(out)}</div>' if out else ""


def _same_category(entry: dict, siblings: list[dict], prev: dict | None, nxt: dict | None) -> str:
    others = [s for s in siblings if not _is_current(entry, s)]
    if not others:
        return ""  # 分类里只有这一篇：整块省略，不印空标题
    cards = []
    for sib in others:
        bits = [_esc(_min_label(sib))] if _min_label(sib) else []
        if _date(sib):
            bits.append(_esc(_date(sib)))
        meta = '<i aria-hidden="true"></i>'.join(bits)
        cards.append(
            f'<a class="sm-more-a" href="{_esc(_page_href(entry, sib))}">'
            f'<span class="sm-more-t">{_esc(sib.get("title") or sib.get("_h1") or "")}</span>'
            + (f'<span class="sm-more-m">{meta}</span>' if meta else "")
            + "</a>"
        )
    return (
        '<section class="sm-more" aria-labelledby="sm-more-h">'
        '<h2 class="sm-more-h" id="sm-more-h">同分类的其他条目</h2>'
        f'<div class="sm-more-grid">{"".join(cards)}</div></section>'
    )


def _bottom(entry: dict, siblings: list[dict], prev: dict | None, nxt: dict | None) -> str:
    pn = _prev_next(entry, prev, nxt)
    more = _same_category(entry, siblings, prev, nxt)
    if not (pn or more):
        return ""
    return f'<nav class="sm-foot" aria-label="页面导航">{pn}</nav>' + more


# ────────────────────────────── 插入点 ──────────────────────────────

def _insert_head(html: str, entry: dict, ctx: dict) -> str:
    # meta 块：接在 <title> 之后；页面若已有 description，就地改写，不重复添加
    has_description = bool(_DESCRIPTION_RE.search(html))
    if has_description:
        one = str(entry.get("one_liner") or entry.get("_lead") or "").strip()
        if one:
            tag = f'<meta name="description" content="{_esc(one)}">'
            html = _DESCRIPTION_RE.sub(lambda _m: tag, html, count=1)
        else:
            has_description = False  # 没有可写的内容：保留原有 description，不新增

    meta = _region_or_nothing(_meta_tags(entry, with_description=not has_description))
    m = _TITLE_END_RE.search(html)
    if m:
        html = _insert_at(html, m.end(), meta)
    else:
        h = _HEAD_END_RE.search(html)
        if h:
            html = _insert_at(html, h.start(), meta)

    # 样式块：theme 令牌在前，外壳样式在后，贴着 </head>
    style = _region(
        '<style id="sm-shell">\n' + theme.TOKENS_CSS + "\n\n" + SHELL_CSS + "\n</style>"
    )
    h = _HEAD_END_RE.search(html)
    if h:
        html = _insert_at(html, h.start(), style)
    return html


def _insert_body(html: str, entry: dict, ctx: dict) -> str:
    siblings = [s for s in (ctx.get("siblings") or []) if isinstance(s, dict)] or [entry]
    chunks = _region_or_nothing(_bar(entry)) + _region_or_nothing(_rail(entry, siblings))
    m = _BODY_OPEN_RE.search(html)
    if m:
        return _insert_at(html, m.end(), chunks)
    return _insert_at(html, 0, chunks) if chunks else html


def _insert_footer(html: str, entry: dict, ctx: dict) -> str:
    siblings = [s for s in (ctx.get("siblings") or []) if isinstance(s, dict)] or [entry]
    chunk = _region_or_nothing(_bottom(entry, siblings, ctx.get("prev"), ctx.get("next")))
    if not chunk:
        return html
    m = _FOOTER_RE.search(html)          # .wrap 内、原 <footer> 之前
    if m:
        return _insert_at(html, m.start(), chunk)
    m = _BODY_END_RE.search(html)        # 兜底：正文末尾
    if m:
        close = html.rfind("</div>", 0, m.start())
        return _insert_at(html, close if close != -1 else m.start(), chunk)
    return html


# ────────────────────────────── 公开入口 ──────────────────────────────

def inject(html: str, ctx: dict) -> str:
    """返回注入了站点外壳的整页 HTML。

    幂等：先剥掉旧的 shell 段再插新的，连跑两次字节一致；
    且 `_strip(inject(html)) == html`，正文一个字节不动。
    """
    entry = (ctx or {}).get("entry") or {}
    out = _strip(html)
    out = _insert_head(out, entry, ctx or {})
    out = _insert_body(out, entry, ctx or {})
    out = _insert_footer(out, entry, ctx or {})
    return out


if __name__ == "__main__":  # 手工冒烟：python3 shell.py <page>（只打印，不落盘）
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) == 3:
        page = Path(sys.argv[1]).read_text(encoding="utf-8")
        ctx = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        out = inject(page, ctx)
        assert inject(out, ctx) == out, "not idempotent"
        print(f"ok · {len(out)} bytes · strip-restore={_strip(out) == page}")
    else:
        print(__doc__)
