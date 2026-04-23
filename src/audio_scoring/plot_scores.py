#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Plot TTS model scores (naturalness, noisiness, loudness) with std error bars.

Reads minicpm baseline and finetuned scores from lily-data evaluation JSONs,
groups by TTS model, and produces a 2x3 grid.

Usage:
    python audio_quality_scoring/plot_scores.py
    python audio_quality_scoring/plot_scores.py --eval-dir path/to/evaluation
"""

import argparse
import json
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#4A90D9", "#D94F8A", "#E8735A", "#3DAA8F", "#9B8EC1",
          "#E08A2B", "#5CB85C", "#2BA5B5", "#6A5ACD", "#D95FAD"]
METRICS = ["naturalness", "noisiness", "loudness"]
DEFAULT_EVAL_DIR = "lily-data/evaluation_output/evaluation"

# Known TTS suffixes in combo folder names (longest first for greedy match)
TTS_NAMES = sorted([
    "aws_polly", "cartesia", "deepgram_aura", "groq", "nvidia_magpie",
], key=len, reverse=True)


def extract_tts(combo_name: str) -> str:
    """Extract TTS model name from a combo folder like 'assemblyai_cartesia'."""
    for tts in TTS_NAMES:
        if combo_name.endswith(tts):
            return tts
    # Fallback: last token after splitting known STT prefixes
    return combo_name.rsplit("_", 1)[-1]


def load_scores_from_evals(eval_dir: str) -> dict[str, dict[str, list[dict]]]:
    """Load minicpm scores from evaluation JSONs, grouped by TTS model.

    Returns: {mode: {tts_model: [score_dicts]}}
    """
    result = {"baseline": defaultdict(list), "finetuned": defaultdict(list)}

    for jf in sorted(glob.glob(os.path.join(eval_dir, "*/*_evaluation.json"))):
        combo = os.path.basename(os.path.dirname(jf))
        tts = extract_tts(combo)

        with open(jf) as f:
            data = json.load(f)

        for entry in data.get("evaluations", []):
            vq = entry.get("voice_quality") or {}
            for mode, key in [("baseline", "minicpm"), ("finetuned", "minicpm_finetuned")]:
                scores = vq.get(key) or {}
                if scores and "error" not in scores:
                    result[mode][tts].append(scores)

    return result


def plot_metric(ax, data, metric, colors, mode):
    models = sorted(data.keys())
    means = [np.mean([s[metric] for s in data[m]]) for m in models]
    stds = [np.std([s[metric] for s in data[m]]) for m in models]

    y_pos = range(len(models))
    ax.barh(y_pos, means, color=colors[:len(models)], edgecolor="none", height=0.7)
    ax.errorbar(means, y_pos, xerr=stds, fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=14)
    ax.set_xlabel("Score (1-10)", fontsize=14)

    max_val = max(m + s for m, s in zip(means, stds))
    min_val = min(m - s for m, s in zip(means, stds))
    pad = (max_val - min_val) * 0.15 or 0.3
    ax.set_xlim(max(0, min_val - pad), max_val + pad)

    title = "Baseline" if mode == "baseline" else "Fine-tuned"
    ax.set_title(f"{title} — {metric.capitalize()}", fontsize=16, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot MiniCPM audio quality scores")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                        help=f"Evaluation JSON directory (default: {DEFAULT_EVAL_DIR})")
    args = parser.parse_args()

    all_data = load_scores_from_evals(args.eval_dir)

    for mode in ["baseline", "finetuned"]:
        if not all_data[mode]:
            print(f"No {mode} scores found in {args.eval_dir}")
            return
        n = sum(len(v) for v in all_data[mode].values())
        print(f"{mode}: {n} scores across {len(all_data[mode])} TTS models")

    output_dir = os.path.join("output/scoring_output", "plots")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for row, mode in enumerate(["baseline", "finetuned"]):
        for col, metric in enumerate(METRICS):
            plot_metric(axes[row][col], all_data[mode], metric, COLORS, mode)

    fig.suptitle("Voice Evaluation LLM Scores", fontsize=20, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "tts_combined.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
