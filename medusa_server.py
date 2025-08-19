#!/usr/bin/env python3
"""
vLLM Server using Custom Medusa Engine (fixed)
- Use an explicit GPU_MEM_UTIL variable instead of reading DeviceConfig fields
- Build vllm_config, instantiate AsyncMedusaLLMEngine, then call run_server(args, engine=engine)
"""

import os
import asyncio
import gc
import torch

from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import run_server, make_arg_parser
from vllm.engine.llm_engine import LLMEngine
from vllm.config import (
    VllmConfig, ModelConfig, DeviceConfig, ParallelConfig,
    ObservabilityConfig, CacheConfig
)

# Import your custom Medusa engine implementation (ensure medusa_engine.py is importable)
from medusa_engine import AsyncMedusaLLMEngine

# ---------------- env tweaks ----------------
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.cuda.empty_cache()
gc.collect()

# ---------------- checkpoint + model ----------------
MEDUSA_CHECKPOINT = "./vllm_medusa_model"
if not os.path.exists(MEDUSA_CHECKPOINT):
    raise FileNotFoundError(f"Medusa checkpoint not found at {MEDUSA_CHECKPOINT}")

# dtype for vLLM config (string)
dtype_str = "float16"

# ---------------- vLLM config ----------------
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # change if needed

# Pick an explicit GPU memory utilization value here and reuse it both for vllm_config and CLI args.
GPU_MEM_UTIL = 0.25

model_config = ModelConfig(
    model=model_name,
    dtype=dtype_str,
    max_model_len=256,          # small context to reduce KV usage for testing
    trust_remote_code=True,
    enforce_eager=True,
    load_format="auto",
)

# Create DeviceConfig with the same numeric value but don't assume attribute names later
device_config = DeviceConfig(
    # many DeviceConfig implementations accept different fields; we keep this minimal
    max_num_batched_tokens=256,
    max_num_seqs=4,
)

parallel_config = ParallelConfig(tensor_parallel_size=1)

observability_config = ObservabilityConfig(
    show_hidden_metrics=False,
    enable_metrics_server=False,
)

cache_config = CacheConfig(
    gpu_memory_utilization=GPU_MEM_UTIL,
    swap_space_bytes=0,
    cpu_offload_gb=0,
    num_gpu_blocks=None,
    num_cpu_blocks=0,
    enable_prefix_caching=False,
)

vllm_config = VllmConfig(
    model_config=model_config,
    device_config=device_config,
    parallel_config=parallel_config,
    observability_config=observability_config,
    cache_config=cache_config,
)

# ---------------- main ----------------
async def main():
    # Let vLLM pick the executor for this config
    executor_class = LLMEngine._get_executor_cls(vllm_config)
    print(f"Executor class: {executor_class}")

    # Instantiate your AsyncMedusaLLMEngine from vllm_config
    engine = AsyncMedusaLLMEngine.from_vllm_config(
        vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )
    print("✅ Medusa Async engine created")

    # Use vLLM's CLI parser to produce a full args object that run_server expects.
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    # Build argv — use the explicit GPU_MEM_UTIL variable
    argv = [
        "--model", model_name,
        "--dtype", dtype_str,
        "--max-model-len", str(model_config.max_model_len),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--tensor-parallel-size", str(parallel_config.tensor_parallel_size),
        "--trust-remote-code",
        "--enforce-eager",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--uvicorn-log-level", "info",
    ]

    args = parser.parse_args(argv)
    print("vLLM parsed args (non-defaults):", args)

    # Set the engine in the args so vLLM can use it
    args.engine = engine
    
    # Call run_server without the engine parameter to avoid the uvicorn.Config error
    await run_server(args)

if __name__ == "__main__":
    asyncio.run(main())
