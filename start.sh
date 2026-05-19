#!/bin/bash

# DynaHMRC 一键启动脚本
# 同时启动 TypeScript 后端 (端口3001) + React 前端 (端口5173)

PROJECT_DIR="/home/developer/.openclaw/workspace/dynahmrc"

echo "========================================="
echo "  🚀 DynaHMRC 一键启动"
echo "========================================="

# 1. 清理旧进程
echo "[1/4] 清理旧进程..."
pkill -9 -f "tsx" 2>/dev/null
pkill -9 -f "vite" 2>/dev/null
pkill -9 -f "node.*index.ts" 2>/dev/null
sleep 2
echo "      ✅ 已清理"

# 2. 启动后端
echo "[2/4] 启动后端 (端口 3001)..."
cd "$PROJECT_DIR/server"
npm run dev > /tmp/dynahmrc_server.log 2>&1 &
SERVER_PID=$!
echo "      PID: $SERVER_PID"

# 3. 等待后端就绪
echo "[3/4] 等待后端就绪..."
for i in {1..30}; do
  if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    echo "      ✅ 后端已就绪 (${i}s)"
    break
  fi
  sleep 1
done

# 4. 启动前端
echo "[4/4] 启动前端 (端口 5173)..."
cd "$PROJECT_DIR/frontend"
npm run dev > /tmp/dynahmrc_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "      PID: $FRONTEND_PID"
sleep 3

echo ""
echo "========================================="
echo "  ✅ DynaHMRC 已启动!"
echo ""
echo "  🌐 前端:  http://localhost:5173"
echo "  🔧 后端:  http://localhost:3001"
echo ""
echo "  日志文件:"
echo "    server:  tail -f /tmp/dynahmrc_server.log"
echo "    frontend: tail -f /tmp/dynahmrc_frontend.log"
echo ""
echo "  停止服务: kill $SERVER_PID $FRONTEND_PID"
echo "========================================="
