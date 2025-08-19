#!/usr/bin/env python3
"""
vLLM OpenAI API Server with Medusa speculative decoding enabled.

Usage examples:
  python medusa_vllm_server.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --medusa-checkpoint ./vllm_medusa_model/vllm_medusa_heads_final.pt \
    --medusa-num-speculative-tokens 5

Notes:
 - This script relies on vLLM's built-in speculative decoding. No custom
   engine overrides are used; we only supply SpeculativeConfig(method="medusa").
 - You can also provide the Medusa checkpoint path via the MEDUSA_CHECKPOINT
   environment variable.
"""

import os
import sys
import uvloop
from argparse import Namespace

from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import make_arg_parser, run_server


def build_args(argv: list[str]) -> Namespace:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (Medusa-enabled)."
    )
    parser = make_arg_parser(parser)

    # Medusa-specific convenience flags
    parser.add_argument(
        "--medusa-checkpoint",
        type=str,
        default=os.environ.get(
            "MEDUSA_CHECKPOINT", "./vllm_medusa_model/vllm_medusa_heads_final.pt"
        ),
        help=(
            "Path to Medusa weights (.pt) or directory containing them. "
            "Defaults to $MEDUSA_CHECKPOINT or ./vllm_medusa_model/vllm_medusa_heads_final.pt"
        ),
    )
    parser.add_argument(
        "--medusa-num-speculative-tokens",
        type=int,
        default=5,
        help=(
            "Number of speculative tokens for Medusa (num_lookahead). "
            "This becomes SpeculativeConfig.num_speculative_tokens."
        ),
    )

    args = parser.parse_args(argv)

    # Attach SpeculativeConfig for Medusa to vLLM engine args. The API server
    # will pass this to the engine via AsyncEngineArgs.
    # See EngineArgs.create_speculative_config for how this dict is consumed.
    args.speculative_config = {
        "method": "medusa",
        "model": args.medusa_checkpoint,
        "num_speculative_tokens": args.medusa_num_speculative_tokens,
    }

    return args


def main() -> None:
    # If users prefer V0 explicitly, they can set VLLM_USE_V1=0 in env.
    # Both V0 and V1 support Medusa; we leave the oracle to choose by default.
    args = build_args(sys.argv[1:])
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()


