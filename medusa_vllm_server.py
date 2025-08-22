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
    --medusa-checkpoint ./vllm_medusa_model \
    --medusa-num-speculative-tokens 5

Notes:
 - This script relies on vLLM's built-in speculative decoding. No custom
   engine overrides are used; we only supply SpeculativeConfig(method="medusa").
 - You can also provide the Medusa checkpoint path via the MEDUSA_CHECKPOINT
   environment variable.
 - Automatically uses CUDA device 1 (GPU 1) to avoid conflicts with base model.
 - Runs on port 8000 (Medusa server).
 - Deterministic decoding is automatically enforced (top_k=1, temperature=0.0).
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
import uvloop
from argparse import Namespace

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import (
    make_arg_parser,
    setup_server,
    build_async_engine_client,
    build_app,
    init_app_state,
)
from vllm.entrypoints.launcher import serve_http
from vllm.v1.metrics.prometheus import get_prometheus_registry
import vllm.envs as envs


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
            "MEDUSA_CHECKPOINT", "./vllm_medusa_model"
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
    
    # Set deterministic decoding parameters
    args.top_k = 1
    args.temperature = 0.0

    return args


def main() -> None:
    # Set CUDA device to GPU 1 by default for Medusa server
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # If users prefer V0 explicitly, they can set VLLM_USE_V1=0 in env.
    # Both V0 and V1 support Medusa; we leave the oracle to choose by default.
    args = build_args(sys.argv[1:])

    async def run_with_metrics() -> None:
        listen_address, sock = setup_server(args)

        async with build_async_engine_client(args) as engine_client:
            app = build_app(args)

            # Initialize app state (models/handlers)
            vllm_config = await engine_client.get_vllm_config()
            await init_app_state(engine_client, vllm_config, app.state, args)

            # Attach previous snapshot store for metrics deltas
            app.state._spec_prev_snapshot = None  # type: ignore[attr-defined]

            def _sample_spec_metrics(app_state) -> Optional[Dict[str, Any]]:
                try:
                    registry = get_prometheus_registry()
                except Exception:
                    return None

                # Get totals across all engines
                totals = {
                    "accepted": 0.0,
                    "drafted_tokens": 0.0,
                    "num_drafts": 0.0,
                    "per_pos": {},  # pos -> total accepted
                }

                for metric in registry.collect():
                    name = metric.name
                    if name == "vllm:spec_decode_num_accepted_tokens":
                        for s in metric.samples:
                            totals["accepted"] += float(s.value or 0.0)
                    elif name == "vllm:spec_decode_num_draft_tokens":
                        for s in metric.samples:
                            totals["drafted_tokens"] += float(s.value or 0.0)
                    elif name == "vllm:spec_decode_num_drafts":
                        for s in metric.samples:
                            totals["num_drafts"] += float(s.value or 0.0)
                    elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                        for s in metric.samples:
                            pos = s.labels.get("position") if s.labels else None
                            if pos is not None:
                                totals["per_pos"][pos] = totals["per_pos"].get(pos, 0.0) + float(s.value or 0.0)

                # Use deltas since last snapshot to approximate interval metrics
                prev = getattr(app.state, "_spec_prev_snapshot", None)  # type: ignore[attr-defined]
                setattr(app.state, "_spec_prev_snapshot", totals)  # type: ignore[attr-defined]

                def diff(curr: float, old: Optional[float]) -> float:
                    if old is None:
                        return 0.0
                    return max(0.0, curr - old)

                # If prev missing, fall back to cumulative
                accepted = diff(totals["accepted"], None if prev is None else prev.get("accepted")) if prev is not None else totals["accepted"]
                drafted_tokens = diff(totals["drafted_tokens"], None if prev is None else prev.get("drafted_tokens")) if prev is not None else totals["drafted_tokens"]
                num_drafts = diff(totals["num_drafts"], None if prev is None else prev.get("num_drafts")) if prev is not None else totals["num_drafts"]

                # Per-position array ordered by position index if available
                per_pos_rates: List[float] = []
                if totals["per_pos"]:
                    # sort by integer pos label
                    items = sorted(((int(k), v) for k, v in totals["per_pos"].items()), key=lambda x: x[0])
                    for pos, val in items:
                        base = 0.0
                        if prev and "per_pos" in prev and str(pos) in prev["per_pos"]:
                            base = prev["per_pos"][str(pos)]
                        inc = max(0.0, val - base) if prev is not None else val
                        per_pos_rates.append((inc / num_drafts) if num_drafts > 0 else 0.0)

                acceptance_rate = (accepted / drafted_tokens * 100.0) if drafted_tokens > 0 else None
                mean_acceptance_length = (1.0 + (accepted / num_drafts)) if num_drafts > 0 else None

                return {
                    "draft_acceptance_rate_percent": acceptance_rate,
                    "mean_acceptance_length": mean_acceptance_length,
                    "accepted": int(accepted),
                    "drafted": int(drafted_tokens),
                    "per_position_acceptance_rate": per_pos_rates,
                }

            @app.middleware("http")
            async def append_spec_metrics(request: Request, call_next):  # type: ignore[override]
                response = await call_next(request)
                # Only modify non-streaming JSON responses for completion routes
                path = request.url.path
                content_type = response.headers.get("content-type", "")
                if (path in ("/v1/completions", "/v1/chat/completions") and
                        content_type.startswith("application/json")):
                    try:
                        body_bytes = b"".join([section async for section in response.body_iterator])
                        payload = json.loads(body_bytes.decode("utf-8"))
                        metrics = _sample_spec_metrics(app.state)
                        if metrics is not None:
                            payload["speculative_decoding_metrics"] = metrics
                        # Return new JSONResponse with same status code
                        return JSONResponse(content=payload, status_code=response.status_code)
                    except Exception:
                        # On any error, return original response
                        return response
                return response

            # Start HTTP server
            shutdown_task = await serve_http(
                app,
                sock=sock,
                enable_ssl_refresh=args.enable_ssl_refresh,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                access_log=not args.disable_uvicorn_access_log,
                timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs,
            )

        try:
            await shutdown_task
        finally:
            sock.close()

    uvloop.run(run_with_metrics())


if __name__ == "__main__":
    main()


