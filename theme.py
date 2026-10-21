#!/usr/bin/env python3
"""全站设计令牌 —— 门户与文章页共用的唯一色彩/尺寸来源。

门户是生成的，文章页的 shell 是注入的；两边各自内联 CSS，
所以颜色必须从这一个文件流出，否则两条渲染路径会漂移成两套系统。

用法：
    import theme
    theme.TOKENS_CSS          # 变量块，放进任何一页的 <style> 开头
    theme.cat_color("tooling")  # 分类色
"""
from __future__ import annotations

# 分类中文名与顺序；不在表里的分类排到最后
CATEGORIES: list[tuple[str, str]] = [
    ("inference-systems", "推理系统"),
    ("comm-parallel", "通信与并行"),
    ("training-algorithms", "训练与算法"),
    ("tooling", "工具与方法"),
    ("misc", "其他"),
]
CAT_ZH: dict[str, str] = dict(CATEGORIES)
CAT_ORDER: dict[str, int] = {name: i for i, (name, _) in enumerate(CATEGORIES)}

# 分类色：门户的分类点、文章页的分类 pill、阅读路线都取这里
CAT_COLOR: dict[str, str] = {
    "inference-systems": "#1d4ed8",
    "comm-parallel": "#0b6b3f",
    "training-algorithms": "#6b2fb5",
    "tooling": "#a35a06",
    "misc": "#5b6b82",
}
DEFAULT_COLOR = "#1d4ed8"

# 分类一句话说明：门户的分类标题下需要回答「这一类讲什么」
CAT_BLURB: dict[str, str] = {
    "inference-systems": "请求怎么被调度、KV 怎么被复用、权重怎么被搬动 —— 推理链路上的机制与取舍",
    "comm-parallel": "集合通信、并行切分与计算重叠 —— 多卡协同时先把哪一项开销拿掉",
    "training-algorithms": "训练目标、算子与算法侧的结构选择",
    "tooling": "启动器、诊断工具与方法论 —— 让上面那些机制可复现、可观测",
    "misc": "暂时不归类的方法、清单与实验记录",
}


def cat_color(category: str) -> str:
    return CAT_COLOR.get(category, DEFAULT_COLOR)


def cat_zh(category: str) -> str:
    return CAT_ZH.get(category, category)


# 变量块：门户和文章页开头都原样内联这一块。
TOKENS_CSS = """  :root{
    --bg:#f5f7fb; --panel:#fff; --panel2:#f8fafc; --lane:#f3f6fb; --line:#dde3ee; --line2:#e9eef6;
    --fg:#131c2e; --ink:#0d1526; --dim:#4a5871; --dim2:#657389;
    --accent:#1d4ed8; --accent-soft:#eef4fd; --accent-line:#d5e2f6;
    --chip:#eef2f9; --chipb:#dbe2ee;
    --good:#0b6b3f; --waste:#b23a10; --warn:#a35a06;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
    --shadow-1:0 1px 2px rgba(16,24,40,.04);
    --shadow-2:0 10px 28px -12px rgba(16,24,40,.22), 0 2px 6px rgba(16,24,40,.05);
    --bar-h:52px;
    --measure:980px;
  }"""
