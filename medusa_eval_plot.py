#!/usr/bin/env python3
"""
Read a JSONL of per-sample results (from justpi_curl_runner.py), compute
aggregate metrics, and plot Mean Acceptance Length and Estimated Speedup over
the dataset index.

Usage:
  python medusa_eval_plot.py \
    --input /home/ubuntu/vanshaj/out_justpi_eval.jsonl \
    --output-dir /home/ubuntu/vanshaj
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(path: Path) -> Tuple[List[float], List[float]]:
    mals: List[float] = []
    speedups: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "summary" in obj:
                continue
            mal = None
            su = None
            spec = obj.get("speculative_decoding_metrics")
            if isinstance(spec, dict) and isinstance(spec.get("mean_acceptance_length"), (int, float)):
                mal = float(spec["mean_acceptance_length"])
            if isinstance(obj.get("estimated_speedup"), (int, float)):
                su = float(obj["estimated_speedup"])
            if mal is not None:
                mals.append(mal)
            if su is not None:
                speedups.append(su)
    return mals, speedups


def plot_series(values: List[float], title: str, ylabel: str, out_path: Path) -> None:
    if not values:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(values)), values, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL with eval outputs")
    ap.add_argument("--output-dir", required=True, help="Directory to save plots")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mals, speedups = read_metrics(input_path)

    # Compute macro averages
    avg_mal = sum(mals) / len(mals) if mals else 0.0
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    # Write a small summary JSON next to plots
    summary = {
        "count_with_metrics": len(mals),
        "avg_mean_acceptance_length": avg_mal,
        "avg_estimated_speedup": avg_speedup,
    }
    (out_dir / "medusa_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_series(mals, "Mean Acceptance Length per sample", "Mean acceptance length",
                out_dir / "mean_acceptance_length.png")
    plot_series(speedups, "Estimated Speedup per sample", "Estimated speedup",
                out_dir / "estimated_speedup.png")


if __name__ == "__main__":
    main()


