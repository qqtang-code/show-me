#!/usr/bin/env bash
# 归档后一键提交并推送到 GitHub。幂等：无改动时不产生空提交。
# 用法： bash sync.sh "show-me: <主题>"
set -euo pipefail

cd "$(dirname "$0")"
msg="${1:-show-me: update}"

python3 build_index.py >/dev/null

# 链接体检：跨分类引用最容易少退一级，坏了要立刻看见（只告警，不阻断提交）
if ! python3 check_links.py; then
  echo "sync: ⚠ 上面这些链接是坏的，页面已提交但读者会点到 404 —— 建议先修再继续" >&2
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
  echo "sync: 已提交但推送失败（网络或认证问题），下次 sync 会重试" >&2
  exit 0   # 不阻断 skill 流程
fi