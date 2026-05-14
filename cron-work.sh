#!/bin/bash
# DynaHMRC Auto-Improvement & Push Script
# Runs every 10 minutes via cron

cd "$(dirname "$0")"

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
  log "Cleaning up..."
  rm -rf frontend/node_modules/.vite 2>/dev/null; rm -rf frontend/dist 2>/dev/null
  rm -rf server/dist 2>/dev/null; npm cache clean --force 2>/dev/null || true
  notify "⚠️ 磁盘清理完成 (${DISK_USED}%)"
fi

# === 2. Run test: check if server works with real API ===
log "Testing server..."
SERVER_OK=false
if curl -sf http://localhost:3001/api/health > /dev/null 2>&1; then
  log "Server is running."
  SERVER_OK=true
else
  log "Starting server..."
  cd server && npx tsx src/index.ts &
  cd ..
  sleep 3
  if curl -sf http://localhost:3001/api/health > /dev/null 2>&1; then
    SERVER_OK=true
    log "Server started."
  else
    log "Server failed to start."
    notify "⚠️ 服务启动失败"
  fi
fi

# === 3. Run a quick test scenario ===
if [ "$SERVER_OK" = true ]; then
  log "Running test scenario..."
  TEST_RESULT=$(node -e "
    const http = require('http');
    const WebSocket = require('ws');
    
    // Create run
    http.post = (url, data) => new Promise((resolve, reject) => {
      const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
      const req = http.request(url, opts, res => {
        let body = '';
        res.on('data', c => body += c);
        res.on('end', () => resolve(JSON.parse(body)));
      });
      req.on('error', reject);
      req.write(JSON.stringify(data));
      req.end();
    });
    
    (async () => {
      try {
        const run = await http.post('http://localhost:3001/api/run', {
          taskType: 'pack_objects', layout: 'kitchen', maxSteps: 2
        });
        console.log('RunID:', run.runId);
        
        // Connect WebSocket and collect dialogues
        const result = await new Promise((resolve) => {
          const ws = new WebSocket('ws://localhost:3001/ws/' + run.runId);
          const dialogues = [];
          let completed = false;
          ws.on('message', (data) => {
            const msg = JSON.parse(data);
            if (msg.type === 'dialogue') {
              dialogues.push(msg.data);
            }
            if (msg.type === 'state' && msg.data.stage === 'completed') {
              completed = true;
            }
          });
          ws.on('open', () => ws.send(JSON.stringify({command: 'start'})));
          setTimeout(() => resolve({ dialogues, completed }), 12000);
        });
        
        // Summarize
        const stages = [...new Set(result.dialogues.map(d => d.stage))];
        const robots = [...new Set(result.dialogues.map(d => d.robotName))];
        console.log('Dialogues:', result.dialogues.length);
        console.log('Stages:', stages.join(', '));
        console.log('Robots:', robots.join(', '));
        console.log('Completed:', result.completed);
        
        // Show first 2 self-descriptions
        for (const d of result.dialogues) {
          if (d.stage === 'self_description') {
            const brief = d.content.replace(/\\n/g, ' ').slice(0, 120);
            console.log(d.robotName + ': ' + brief);
          }
        }
      } catch(e) {
        console.log('TEST_ERROR:', e.message);
      }
    })();
  " 2>&1)
  log "Test result: $TEST_RESULT"
  
  # Check test result
  if echo "$TEST_RESULT" | grep -qi "Dialogues:"; then
    DIALOGUE_COUNT=$(echo "$TEST_RESULT" | grep "Dialogues:" | awk '{print $2}')
    STAGES=$(echo "$TEST_RESULT" | grep "Stages:" | cut -d: -f2 | xargs)
    ROBOTS=$(echo "$TEST_RESULT" | grep "Robots:" | cut -d: -f2 | xargs)
    
    if [ "$DIALOGUE_COUNT" -gt 0 ] 2>/dev/null; then
      notify "✅ 运行正常 | 本轮对话: $DIALOGUE_COUNT 条 | 阶段: $STAGES | 机器人: $ROBOTS"
    else
      notify "⚠️ 测试运行无对话输出"
    fi
  else
    notify "⚠️ 测试运行异常: $(echo "$TEST_RESULT" | tail -3)"
  fi
fi

# === 4. Commit and push ===
if git diff --quiet && git diff --cached --quiet; then
  log "No changes to commit."
else
  git add -A
  git commit -m "auto: update $(date '+%Y-%m-%d %H:%M')"
  log "Committed."
fi

if ! git remote | grep -q origin; then
  git remote add origin "$GH_REPO"
fi

PUSH_OUTPUT=$(git push origin "$BRANCH" 2>&1) || true
log "Push: $PUSH_OUTPUT"

if echo "$PUSH_OUTPUT" | grep -qi "->\|create\|new branch"; then
  notify "📤 代码已推送到 GitHub (openclaw)"
elif echo "$PUSH_OUTPUT" | grep -qi "auth\|401\|403\|fatal"; then
  notify "🔴 GitHub 推送失败，认证需要修复"
fi

log "=== Cycle complete ==="
echo "=== Cycle complete ==="
