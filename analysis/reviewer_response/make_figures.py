"""
Figures for the reviewer-response IAA analysis.

Panel A: per-dimension annotator x annotator Pearson agreement heatmaps
         (lower triangle), with the mean pairwise r (human ceiling) annotated.
Panel B: grouped bars comparing the human agreement ceiling to the fine-tuned
         scorer's correlation with each annotator and with the 3-rater consensus,
         with 95% bootstrap CI error bars.

Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python make_figures.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import agreement_stats as ag
import data_io as io

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def full_corr_matrix(mat: np.ndarray) -> np.ndarray:
    """Symmetric Pearson matrix across raters (diagonal = 1)."""
    k = mat.shape[1]
    C = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = mat[:, i], mat[:, j]
            m = ~(np.isnan(a) | np.isnan(b))
            r = ag.pearson_stat(a[m], b[m])
            C[i, j] = C[j, i] = r
    return C


def main():
    test_ids = io.load_test_ids()
    preds, _ = io.load_model_predictions()
    raters = io.RATERS
    disp = io.display_names()
    dims = io.DIMENSIONS

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.45, wspace=0.35)

    # ---------- Panel A: agreement heatmaps ----------
    for col, dim in enumerate(dims):
        ax = fig.add_subplot(gs[0, col])
        mat = io.build_rater_matrix(dim, test_ids)
        C = full_corr_matrix(mat)
        im = ax.imshow(C, vmin=0, vmax=1, cmap="YlGnBu")
        # annotate upper triangle incl diagonal
        for i in range(len(raters)):
            for j in range(len(raters)):
                if j >= i:
                    ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                            color="white" if C[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(len(raters)))
        ax.set_yticks(range(len(raters)))
        ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(disp, fontsize=8)
        mean_r = np.mean([C[i, j] for i in range(len(raters))
                          for j in range(i + 1, len(raters))])
        ax.set_title(f"{dim.capitalize()}\nmean pairwise r = {mean_r:.2f}", fontsize=10)
    cax = fig.add_axes([0.92, 0.57, 0.015, 0.30])
    fig.colorbar(im, cax=cax, label="Pearson r")

    # ---------- Panel B: model vs ceiling ----------
    for col, dim in enumerate(dims):
        ax = fig.add_subplot(gs[1, col])
        mat = io.build_rater_matrix(dim, test_ids)
        consensus = np.nanmean(mat, axis=1)
        pred = np.array([preds[s][f"pred_{dim}"] for s in test_ids])

        labels, vals, los, his, colors = [], [], [], [], []
        # human ceiling (mean pairwise r) — drawn as a reference line, not a bar
        ceil = np.mean([ag.pearson_stat(mat[:, i][~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])],
                                        mat[:, j][~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])])
                        for i in range(len(raters)) for j in range(i + 1, len(raters))])

        for name in raters:
            idx = raters.index(name)
            ci = ag.bootstrap_ci(pred, mat[:, idx], ag.pearson_stat)
            labels.append(f"vs {io.RATER_DISPLAY.get(name, name)}")
            vals.append(ci["point"]); los.append(ci["point"] - ci["ci_low"])
            his.append(ci["ci_high"] - ci["point"])
            colors.append("#4C72B0" if name != "original" else "#8EA9D0")
        ci = ag.bootstrap_ci(pred, consensus, ag.pearson_stat)
        labels.append("vs consensus")
        vals.append(ci["point"]); los.append(ci["point"] - ci["ci_low"])
        his.append(ci["ci_high"] - ci["point"]); colors.append("#55A868")

        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=[los, his], capsize=3, color=colors, width=0.65)
        ax.axhline(ceil, color="crimson", ls="--", lw=1.5,
                   label=f"human ceiling ({ceil:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Pearson r" if col == 0 else "")
        ax.set_title(f"{dim.capitalize()}: scorer vs humans", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Inter-annotator agreement and fine-tuned scorer vs. human ceiling "
                 f"(held-out test set, n={len(test_ids)}, raters: {', '.join(disp)})",
                 fontsize=12, y=0.98)

    png = os.path.join(OUT_DIR, "iaa_figure.png")
    pdf = os.path.join(OUT_DIR, "iaa_figure.pdf")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}\nWrote {pdf}")


if __name__ == "__main__":
    main()
