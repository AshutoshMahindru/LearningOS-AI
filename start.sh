#!/bin/bash

# LearningOS V3 Unified Startup Script

echo "🚀 Starting LearningOS V3..."

# Setup cleanup trap to kill all background processes on exit
trap 'echo "Shutting down services..."; kill 0' SIGINT SIGTERM EXIT

# Start Frontend
echo "📦 Starting Frontend UI..."
cd platform/frontend
npm run dev &
cd ../..

# Start Backend API
echo "🔌 Starting Backend API Server..."
cd platform/backend
uv run uvicorn app.main:app --reload &
cd ../..

# Start Worker Daemon
echo "⚙️  Starting Code Execution Worker..."
cd platform/backend
uv run python worker_daemon.py &
cd ../..

echo ""
echo "✅ All services started successfully!"
echo "➡️  Frontend: http://localhost:5173"
echo "➡️  Backend:  http://127.0.0.1:8000"
echo "Press Ctrl+C to shut down."
echo ""

# Wait for all background processes
wait
