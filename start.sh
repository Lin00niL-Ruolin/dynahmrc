#!/bin/bash

# DynaHMRC 一键启动脚本
# 启动后端 (端口3001，含前端静态文件) + BestMan 3D (端口5001) + ngrok 公网访问

PROJECT_DIR="/home/developer/.openclaw/workspace/dynahmrc"

echo "========================================="
echo "  🚀 DynaHMRC 一键启动"
echo "========================================="

# 1. 清理旧进程
echo "[1/6] 清理旧进程..."
pkill -9 -f "tsx" 2>/dev/null
pkill -9 -f "vite" 2>/dev/null
pkill -9 -f "node.*index.ts" 2>/dev/null
pkill -9 -f "python service.py" 2>/dev/null
pkill -9 -f ngrok 2>/dev/null
sleep 2
echo "      ✅ 已清理"

# 2. 构建前端
echo "[2/6] 构建前端..."
cd "$PROJECT_DIR/frontend"
npx vite build > /tmp/dynahmrc_frontend_build.log 2>&1
if [ $? -eq 0 ]; then
  echo "      ✅ 前端构建成功 (dist/)"
else
  echo "      ⚠️ 前端构建失败，查看日志: tail -f /tmp/dynahmrc_frontend_build.log"
fi

# 3. 启动后端
echo "[3/6] 启动 DynaHMRC 后端 (端口 3001)..."
cd "$PROJECT_DIR/server"
# 读取 .env 环境变量
export $(grep -v '^\s*#' ../server/.env | grep -v '^\s*$' | xargs) 2>/dev/null
npx tsx src/index.ts > /tmp/dynahmrc_server.log 2>&1 &
SERVER_PID=$!
echo "      PID: $SERVER_PID"

# 4. 等待后端就绪
echo "[4/6] 等待后端就绪..."
for i in {1..30}; do
  if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    echo "      ✅ 后端已就绪 (${i}s)"
    break
  fi
  sleep 1
done

# 5. 启动 BestMan 3D 服务
echo "[5/6] 启动 BestMan 3D (端口 5001)..."
cd "$PROJECT_DIR/bestman-service"
source /home/developer/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
PYTHONPATH="/home/developer/.openclaw/workspace/BestMan:$PYTHONPATH" /home/developer/miniconda/bin/python service.py > /tmp/dynahmrc_bestman.log 2>&1 &
BESTMAN_PID=$!
echo "      PID: $BESTMAN_PID"
sleep 2

# 6. 启动 ngrok 公网访问
echo "[6/6] 启动 ngrok 公网访问..."
which ngrok > /dev/null 2>&1
if [ $? -eq 0 ]; then
  nohup ngrok http 3001 --log=stdout > /tmp/ngrok.log 2>&1 &
  sleep 4
  NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null)
  if [ -n "$NGROK_URL" ]; then
    echo "      ✅ ngrok 已启动: $NGROK_URL"
  else
    echo "      ⚠️  ngrok 启动中，稍后查看: curl -s http://127.0.0.1:4040/api/tunnels"
  fi
else
  echo "      ⚠️  未安装 ngrok，跳过"
fi

echo ""
echo "========================================="
echo "  ✅ DynaHMRC 已启动!"
echo ""
echo "  🌐 公网访问: ${NGROK_URL:-'(ngrok 未启动)'}"
echo "  🔧 后端API:  http://localhost:3001"
echo "  🎮 BestMan:  http://localhost:5001"
echo ""
echo "  日志文件:"
echo "    server:   tail -f /tmp/dynahmrc_server.log"
echo "    bestman:  tail -f /tmp/dynahmrc_bestman.log"
echo "    ngrok:    tail -f /tmp/ngrok.log"
echo ""
echo "  停止服务:"
echo "    pkill -f tsx; pkill -f python; pkill -f ngrok"
echo "========================================="
