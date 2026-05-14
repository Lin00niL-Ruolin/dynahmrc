#!/bin/bash
# DynaHMRC Auto-Improvement & Push Script
# Runs periodically via cron

cd "$(dirname "$0")"

# Load API key
export DEEPSEEK_API_KEY="sk-1a24ae1b09de434d87ef49913b1dfe66"
[ -f server/.env ] && export $(grep -v '^#' server/.env | xargs)

LOG_FILE=".cron.log"
NOTIFY_FILE=".notify-queue"
GH_REPO="https://github.com/Lin00niL-Ruolin/dynahmrc.git"
BRANCH="openclaw"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

notify() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$NOTIFY_FILE"
  log "NOTIFY: $msg"
}

# === 1. Check disk space ===
DISK_USED=$(df /home/developer | tail -1 | awk '{print $5}' | sed 's/%//')
log "Disk usage: ${DISK_USED}%"

if [ "$DISK_USED" -gt 85 ]; then
  log "WARN: Disk space low. Cleaning up..."
  rm -rf frontend/node_modules/.vite 2>/dev/null
  rm -rf frontend/dist 2>/dev/null
  rm -rf server/dist 2>/dev/null
  npm cache clean --force 2>/dev/null || true
  notify "⚠️ 磁盘空间不足(${DISK_USED}%)，已执行自动清理"
fi

# === 2. Commit changes ===
if git diff --quiet && git diff --cached --quiet; then
  log "No changes to commit."
else
  git add -A
  git commit -m "auto: periodic update $(date '+%Y-%m-%d %H:%M')"
  log "Committed changes."
fi

# === 3. Push to GitHub ===
if ! git remote | grep -q origin; then
  git remote add origin "$GH_REPO"
fi

log "Pushing to GitHub..."
PUSH_OUTPUT=$(git push origin "$BRANCH" 2>&1) || true
log "Push: $PUSH_OUTPUT"

if echo "$PUSH_OUTPUT" | grep -qi "everything up-to-date"; then
  log "Already up to date."
elif echo "$PUSH_OUTPUT" | grep -qi "could not read\|auth\|401\|403\|fatal"; then
  notify "🔴 GitHub 推送失败：需要配置认证。\n请运行：git remote set-url origin https://<token>@github.com/Lin00niL-Ruolin/dynahmrc.git"
elif echo "$PUSH_OUTPUT" | grep -qi "->\|create\|new branch"; then
  notify "✅ 代码已推送到 GitHub (openclaw 分支)"
fi

# === 4. Check server health ===
if curl -sf http://localhost:3001/api/health > /dev/null 2>&1; then
  log "Server is healthy."
else
  notify "⚠️ DynaHMRC 服务未运行，尝试重启..."
  cd server && npx tsx src/index.ts &
  cd ..
fi

log "Cron cycle complete."
