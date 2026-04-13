#!/usr/bin/env python3
"""Plot TTS model scores (naturalness, noisiness, loudness) with std error bars.

Combines baseline and finetuned into a single 2x3 grid.

Usage:
    python audio_quality_scoring/plot_scores.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

COLORS = ["#A8D8EA", "#F2C6DE", "#F0A8A8", "#B5EAD7", "#C3B1E1"]
METRICS = ["naturalness", "noisiness", "loudness"]


def load_scores(scores_dir):
    data = {}
    for fname in sorted(os.listdir(scores_dir)):
        if fname.startswith("_") or not fname.endswith(".json"):
            continue
        model = fname.replace(".json", "")
        with open(os.path.join(scores_dir, fname)) as f:
            scores = json.load(f)
        valid = [s for s in scores if "error" not in s]
        data[model] = valid
    return data


def plot_metric(ax, data, metric, colors, mode):
    models = list(data.keys())
    means = [np.mean([s[metric] for s in data[m]]) for m in models]
    stds = [np.std([s[metric] for s in data[m]]) for m in models]

    y_pos = range(len(models))
    ax.barh(y_pos, means, color=colors[:len(models)], edgecolor="none", height=0.7, alpha=0.7)
    ax.errorbar(means, y_pos, xerr=stds, fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Score (1-10)", fontsize=10)

    max_val = max(m + s for m, s in zip(means, stds))
    min_val = min(m - s for m, s in zip(means, stds))
    pad = (max_val - min_val) * 0.15 or 0.3
    ax.set_xlim(max(0, min_val - pad), max_val + pad)

    title = "Baseline" if mode == "baseline" else "Fine-tuned"
    ax.set_title(f"{title} — {metric.capitalize()}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def main():
    modes = ["baseline", "finetuned"]
    all_data = {}
    for mode in modes:
        scores_dir = os.path.join("scoring_output", mode, "tts_models")
        if not os.path.exists(scores_dir):
            print(f"No scores found at {scores_dir}. Run score_audio_minicpm.py --batch --mode {mode} first.")
            return
        all_data[mode] = load_scores(scores_dir)

    output_dir = os.path.join("scoring_output", "plots")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for row, mode in enumerate(modes):
        for col, metric in enumerate(METRICS):
            plot_metric(axes[row][col], all_data[mode], metric, COLORS, mode)

    fig.suptitle("Voice Evaluation LLM Scores", fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "tts_combined.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
