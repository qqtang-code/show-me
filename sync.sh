#!/usr/bin/env bash
# 归档后一键提交并推送到 GitHub。幂等：无改动时不产生空提交。
# 用法： bash sync.sh "show-me: <主题>"
set -euo pipefail

cd "$(dirname "$0")"
msg="${1:-show-me: update}"

python3 build_index.py >/dev/null

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
  echo "sync: 已提交但推送失败（网络或认证问题），下次 sync 会重试" >&2
  exit 0   # 不阻断 skill 流程
fi