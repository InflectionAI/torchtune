#!/usr/bin/env python3
"""
Run justpi JSONL data through the vLLM OpenAI server by issuing curl requests.

Reads a JSONL dataset (one JSON object per line), extracts either a prompt
string or a chat messages array, and POSTs to /v1/completions or
/v1/chat/completions accordingly. Responses are written to stdout and optionally to an
output JSONL file, preserving the original record with the response.

Example:
  python justpi_curl_runner.py \
    --data /home/ubuntu/vanshaj/justpi.jsonl \
    --endpoint http://127.0.0.1:8001 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-tokens 128 --temperature 0.0 --limit 10 \
    --output /home/ubuntu/vanshaj/out_justpi_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pick_prompt(obj: dict, prompt_key: str) -> Optional[str]:
    # Primary key, if provided
    if prompt_key and prompt_key in obj and isinstance(obj[prompt_key], str):
        return obj[prompt_key]
    # Common fallbacks
    for key in ("prompt", "input", "instruction", "question", "text"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def make_payload(model: str, prompt: str, max_tokens: int, temperature: float,
                 top_p: float, logprobs: Optional[int]) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    return payload


def send_curl(endpoint: str, payload: dict, timeout: int) -> tuple[int, str]:
    # Use curl with JSON via stdin to avoid quoting issues.
    data = json.dumps(payload).encode("utf-8")
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(timeout),
        endpoint,
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description="Send justpi JSONL through vLLM server using curl")
    ap.add_argument("--data", required=True, help="Path to justpi JSONL file")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001",
                    help="OpenAI server base URL (no trailing slash)")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="Model name to send in requests")
    ap.add_argument("--prompt-key", default="prompt",
                    help="JSON key to read the prompt from (fallbacks applied if missing)")
    ap.add_argument("--force-chat", action="store_true",
                    help="Treat all records as chat and send to /v1/chat/completions; expects 'messages' key or builds from prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--logprobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all)")
    ap.add_argument("--start-percent", type=float, default=0.0,
                    help="Start at this percent of the dataset (e.g., 90.0 for last 10%)")
    ap.add_argument("--end-percent", type=float, default=100.0,
                    help="End at this percent of the dataset (100.0 = end)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Set temperature=0, top_p=0 for deterministic decoding")
    ap.add_argument("--output", default=None, help="Write prompts/responses/tokens as JSONL here")
    ap.add_argument("--timeout", type=int, default=120, help="curl max time in seconds")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    out_fp = None
    if args.output:
        out_fp = Path(args.output).open("w", encoding="utf-8")

    count = 0
    base = args.endpoint.rstrip("/")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    total_count = 0

    def extract_prompt_from_messages(messages: list) -> str:
        # Prefer the last user message content
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
        # Fallback: stringify
        try:
            return json.dumps(messages, ensure_ascii=False)
        except Exception:
            return ""

    # Compute start/end indices based on percentages.
    # We count total lines quickly.
    total_lines = 0
    with data_path.open("r", encoding="utf-8") as _f:
        for _ in _f:
            total_lines += 1
    start_idx = int(total_lines * (args.start_percent / 100.0))
    end_idx = int(total_lines * (args.end_percent / 100.0))
    if end_idx <= start_idx:
        end_idx = total_lines

    # Iterate and skip until start_idx
    idx = -1
    for obj in iter_jsonl(data_path):
        idx += 1
        if idx < start_idx:
            continue
        if idx >= end_idx:
            break
        # Prefer chat if messages present or --force-chat
        is_chat = args.force_chat or isinstance(obj.get("messages"), list)

        if is_chat:
            messages = obj.get("messages")
            if not isinstance(messages, list):
                # Build simple chat from prompt field
                prompt = pick_prompt(obj, args.prompt_key)
                if not prompt:
                    continue
                messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0.0 if args.deterministic else args.temperature,
                "top_p": 1.0 if args.deterministic else args.top_p,
            }
            endpoint = f"{base}/v1/chat/completions"
            prompt_for_save = extract_prompt_from_messages(messages)
        else:
            prompt = pick_prompt(obj, args.prompt_key)
            if not prompt:
                continue
            payload = make_payload(
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=0.0 if args.deterministic else args.temperature,
                top_p=1.0 if args.deterministic else args.top_p,
                logprobs=args.logprobs,
            )
            endpoint = f"{base}/v1/completions"
            prompt_for_save = prompt

        code, resp = send_curl(endpoint, payload, args.timeout)
        # Echo minimal progress to stdout
        print(resp)

        if out_fp is not None:
            # Try to parse response and store prompt/response/tokens
            response_obj = None
            response_text = None
            usage = None
            spec_metrics = None
            mean_accept_len = None
            est_speedup = None
            try:
                response_obj = json.loads(resp)
                # completions
                choices = response_obj.get("choices") or []
                if choices:
                    first = choices[0]
                    if "text" in first:
                        response_text = first.get("text")
                    elif isinstance(first.get("message"), dict):
                        response_text = first["message"].get("content")
                usage = response_obj.get("usage")
                if isinstance(usage, dict):
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    tt = int(usage.get("total_tokens") or pt + ct)
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    total_total_tokens += tt
                    total_count += 1
                spec_metrics = response_obj.get("speculative_decoding_metrics")
                if isinstance(spec_metrics, dict):
                    mal = spec_metrics.get("mean_acceptance_length")
                    if isinstance(mal, (int, float)):
                        mean_accept_len = float(mal)
                        est_speedup = mean_accept_len
            except Exception:
                pass

            out_record = {
                "prompt": prompt_for_save,
                "response": response_text if response_text is not None else resp,
                "usage": usage,
                "speculative_decoding_metrics": spec_metrics,
                "estimated_speedup": est_speedup,
            }
            out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        count += 1
        if args.limit and count >= args.limit:
            break

    if out_fp is not None:
        # Append a final summary line with averages if any
        if total_count > 0:
            summary = {
                "summary": {
                    "count": total_count,
                    "avg_prompt_tokens": total_prompt_tokens / total_count,
                    "avg_completion_tokens": total_completion_tokens / total_count,
                    "avg_total_tokens": total_total_tokens / total_count,
                }
            }
            out_fp.write(json.dumps(summary, ensure_ascii=False) + "\n")
        out_fp.close()


if __name__ == "__main__":
    main()


