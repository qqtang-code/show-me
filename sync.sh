#!/usr/bin/env bash
# 归档后一键提交并推送到 GitHub。幂等：无改动时不产生空提交。
# 用法： bash sync.sh "show-me: <主题>"
set -euo pipefail

cd "$(dirname "$0")"
msg="${1:-show-me: update}"

python3 build_index.py >/dev/null

# 链接体检：跨分类引用少退一级、或者锚点写错，读者都会点到 404 / 停在页首。
# 这一关是阻断的 —— 公开仓库的坏链接不该靠人眼发现。
if ! python3 check_links.py; then
  echo "sync: ✗ 上面的链接或锚点失效，已中止，未提交" >&2
  exit 1
fi

git add -A
if git diff --cached --quiet; then
  echo "sync: 无改动，跳过"
  exit 0
fi

git commit -q -m "$msg"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "sync: 已提交（无 origin，未推送）"
  exit 0
fi

if git push -q origin HEAD 2>/dev/null; then
  echo "sync: 已推送 — $msg"
else
  # 推送失败不是「做完了」：本地领先远端时必须让调用方看见，否则下次会以为已上线。
  echo "sync: ✗ 已提交但推送失败（网络或认证），本地领先 remote，需重试" >&2
  exit 1
fi