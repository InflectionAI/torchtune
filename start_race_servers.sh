#!/bin/bash

echo "🏁 Starting vLLM Race Servers..."
echo "=================================="

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Check ports before starting
echo "🔍 Checking port availability..."
if ! check_port 8000; then
    echo "Please stop any process using port 8000"
    exit 1
fi

if ! check_port 8001; then
    echo "Please stop any process using port 8001"
    exit 1
fi

echo ""
echo "🚀 Starting Base Model Server (Port 8001)..."
echo "   This will run without Medusa speculative decoding"
echo "   Press Ctrl+C to stop this server when done"
echo ""

# Start base model server (no Medusa)
python3 medusa_vllm_server_no_medusa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001
