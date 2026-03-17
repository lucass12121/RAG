#!/bin/bash
# 法律 Agent 启动脚本
# 步骤1: 启动法律 RAG MCP Server (后台)
# 步骤2: 启动 DeerFlow 前后端

echo "=========================================="
echo "  法律 RAG Agent 系统启动"
echo "=========================================="

# 1. 启动 MCP Server
echo ""
echo "[1/3] 启动法律 RAG MCP Server..."
cd "$(dirname "$0")"
python src/legal_mcp_server.py &
MCP_PID=$!
echo "  MCP Server PID: $MCP_PID"
echo "  等待加载模型..."
sleep 10

# 2. 检查 MCP Server 是否正常
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "  [ERROR] MCP Server 启动失败！"
    exit 1
fi
echo "  MCP Server 已启动: http://localhost:8000"

# 3. 启动 DeerFlow
echo ""
echo "[2/3] 启动 DeerFlow 后端..."
cd /c/Users/MS/deer-flow/backend
# 根据 DeerFlow 文档启动
uv run uvicorn src.gateway.app:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!

echo ""
echo "[3/3] 启动 DeerFlow 前端..."
cd /c/Users/MS/deer-flow/frontend
pnpm dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "  所有服务已启动:"
echo "  - MCP Server:  http://localhost:8000 (PID: $MCP_PID)"
echo "  - Gateway API: http://localhost:8001 (PID: $GATEWAY_PID)"
echo "  - Frontend:    http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo "=========================================="

# 等待并清理
trap "kill $MCP_PID $GATEWAY_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
