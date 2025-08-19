#!/usr/bin/env python3
"""
vLLM Server with Custom Medusa Engine
"""
import sys
from vllm.entrypoints.api_server import main

if __name__ == "__main__":
    sys.argv.extend([
        "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "--dtype", "float16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--enforce-eager",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
    main()