#!/bin/bash
# DynaHMRC Auto-Improvement & Push Script
# Runs periodically via cron to:
# 1. Check disk space and clean up if needed
# 2. Stage improvements to the project
# 3. Push to GitHub openclaw branch

set -e

cd "$(dirname "$0")"

LOG_FILE=".cron.log"
GH_REPO="https://github.com/Lin00niL-Ruolin/dynahmrc.git"
BRANCH="openclaw"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# === 1. Check disk space ===
DISK_USED=$(df /home/developer | tail -1 | awk '{print $5}' | sed 's/%//')
log "Disk usage: ${DISK_USED}%"

if [ "$DISK_USED" -gt 85 ]; then
  log "WARN: Disk space low. Cleaning up..."
  # Clean npm cache
  rm -rf frontend/node_modules/.vite 2>/dev/null
  rm -rf server/dist 2>/dev/null
  rm -rf frontend/dist 2>/dev/null
  npm cache clean --force 2>/dev/null || true
  log "Cleanup done."
fi

# === 2. Check for improvements ===
# Check if server is running with mock mode (no API key)
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  log "No DeepSeek API key configured - running in mock mode"
fi

# === 3. Git auto-commit and push ===
if git diff --quiet && git diff --cached --quiet; then
  log "No changes to commit."
else
  git add -A
  git commit -m "auto: periodic improvement update $(date '+%Y-%m-%d %H:%M')"
  log "Committed changes."
fi

# Push to GitHub
if git remote | grep -q origin; then
  PUSH_RESULT=$(git push origin "$BRANCH" 2>&1) || true
  log "Push: $PUSH_RESULT"
else
  git remote add origin "$GH_REPO"
  log "Added remote origin"
  PUSH_RESULT=$(git push -u origin "$BRANCH" 2>&1) || true
  log "Push: $PUSH_RESULT"
fi

log "Cron cycle complete."
