# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Cross-method TTS quality radar.

Renders one radar (spider) chart with six evaluation-method axes, one polygon
per TTS provider. Each axis is drawn on its OWN fixed native scale (no
cross-provider normalization), so a provider's radius reflects its raw score.
Metrics that saturate (e.g. NISQA near its ceiling) therefore render as nearly
overlapping polygons rather than being stretched apart.

The six methods and their sources in each *_evaluation.json entry:

  1. NISQA MOS            voice_quality.nisqa.nisqa_mos                (1-5,  higher=better)
  2. MOSNet               voice_quality.speechmetrics.mosnet_score     (1-5,  higher=better)
  3. Librosa Naturalness  computed from voice_quality.librosa via the
                          rubric in the paper (Table: TTS scoring rubric),
                          raw range -6..8, normalized to 0-5              (higher=better)
  4. LLM Holistic         voice_quality.llm_judge.llm_overall          (1-5,  higher=better)
  5. MiniCPM Baseline     voice_quality.minicpm.naturalness            (1-10, generative)
  6. MiniCPM Fine-tuned   voice_quality.minicpm_finetuned.naturalness  (1-10, regression head)

All six are "higher is better" naturalness/quality proxies, so the per-axis
min-max normalization keeps polygon shape interpretable (larger = better).

Usage:
    python visualize/plot_cross_method_radar.py \
        --eval-dir output/eval-results/evaluation_output/evaluation \
        --out output/plots/cross-method-radar.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_EVAL_DIR = "output/eval-results/evaluation_output/evaluation"
DEFAULT_OUT = "output/plots/cross-method-radar.png"

KNOWN_TTS = ["aws_polly", "deepgram_aura", "cartesia", "groq", "nvidia_magpie"]
# Legend/plot draw order, matched to the paper figure.
TTS_ORDER = ["aws_polly", "cartesia", "deepgram_aura", "groq", "nvidia_magpie"]
TTS_LABEL = {
    "aws_polly": "AWS Polly",
    "deepgram_aura": "Deepgram Aura",
    "cartesia": "Cartesia",
    "groq": "Groq",
    "nvidia_magpie": "NVIDIA Magpie",
}
# Okabe-Ito colorblind-safe qualitative palette (a standard for scientific
# figures). Distinct, perceptually balanced hues, no near-black under alpha.
TTS_COLOR = {
    "aws_polly": "#0072B2",      # blue
    "cartesia": "#D55E00",       # vermillion
    "deepgram_aura": "#009E73",  # bluish green
    "groq": "#CC79A7",           # reddish purple
    "nvidia_magpie": "#E69F00",  # orange
}

METHODS = [
    "NISQA MOS",
    "MOSNet",
    "Librosa Naturalness",
    "LLM Holistic",
    "MiniCPM Baseline",
    "MiniCPM Fine-tuned",
]

# Native (fixed) value range of each method's scale. Used only to place points
# on each axis's own scale so multi-scale metrics stay readable; NO data-driven
# rescaling is applied, so a provider's position reflects its raw score, not its
# rank relative to the other providers.
AXIS_RANGE = {
    "NISQA MOS": (1.0, 5.0),
    "MOSNet": (1.0, 5.0),
    "Librosa Naturalness": (0.0, 5.0),
    "LLM Holistic": (1.0, 5.0),
    "MiniCPM Baseline": (1.0, 10.0),
    "MiniCPM Fine-tuned": (1.0, 10.0),
}


def tts_of(combo: str):
    for t in KNOWN_TTS:
        if combo.endswith(t):
            return t
    return None


def librosa_naturalness(lib: dict):
    """Naturalness score from raw librosa features (0-5).

    Reproduces the source implementation (VoiceQualityCalculator._calculate_
    naturalness in the original nab-eba codebase). Six sub-scores are summed to
    an integer point total, then mapped to 0-5 via:

        final = clamp(2.5 + points * 0.5, 0, 5)

    (Note: this is the actual normalization used; it centers at 2.5 with +/-0.5
    per point. It differs from the "-6..8 -> 0..5" wording in the paper table,
    but matches the code and the published per-provider values.)

    Requires pitch_std, harmonic_noise_ratio, dynamic_range,
    energy_pitch_correlation, energy_consistency; prosodic_pvi is optional.
    Returns None if a required feature is absent.
    """
    if not lib:
        return None
    pitch_std = lib.get("pitch_std")
    hnr = lib.get("harmonic_noise_ratio")
    dyn = lib.get("dynamic_range")
    pvi = lib.get("prosodic_pvi")
    epc = lib.get("energy_pitch_correlation")
    econs = lib.get("energy_consistency")
    if None in (pitch_std, hnr, dyn, epc, econs):
        return None

    pts = 0

    # Pitch std (Hz): >40 -> +1, <25 -> -1, else 0
    if pitch_std > 40:
        pts += 1
    elif pitch_std < 25:
        pts -= 1

    # HNR (dB): >15 -> +2, >=10 -> +1, else 0
    if hnr > 15:
        pts += 2
    elif hnr >= 10:
        pts += 1

    # Dynamic range (dB): 45-55 -> +1, 30-45 -> 0, 20-30 -> -2, <20 -> -3
    if 45 <= dyn <= 55:
        pts += 1
    elif 30 <= dyn < 45:
        pts += 0
    elif 20 <= dyn < 30:
        pts -= 2
    else:
        pts -= 3

    # Prosodic PVI: 12-30 -> +1 (optional feature)
    if pvi is not None and 12 <= pvi <= 30:
        pts += 1

    # Energy-pitch correlation: 0.3-0.7 -> +1
    if 0.3 <= epc <= 0.7:
        pts += 1

    # Energy consistency: 0.2-0.5 -> +1, <0.15 or >0.6 -> -1, else 0
    if 0.2 <= econs <= 0.5:
        pts += 1
    elif econs < 0.15 or econs > 0.6:
        pts -= 1

    return max(0.0, min(5.0, 2.5 + pts * 0.5))


def collect(eval_dir: str):
    """Return {tts: {method: mean_value}} averaged over all utterances."""
    acc = {t: defaultdict(list) for t in KNOWN_TTS}

    for combo_path in sorted(glob.glob(os.path.join(eval_dir, "*"))):
        if not os.path.isdir(combo_path):
            continue
        tts = tts_of(os.path.basename(combo_path))
        if tts is None:
            continue
        for jf in glob.glob(os.path.join(combo_path, "*_evaluation.json")):
            try:
                data = json.load(open(jf))
            except (json.JSONDecodeError, OSError):
                continue
            for e in data.get("evaluations", []):
                vq = e.get("voice_quality") or {}

                nisqa = (vq.get("nisqa") or {}).get("nisqa_mos")
                if nisqa is not None:
                    acc[tts]["NISQA MOS"].append(nisqa)

                mosnet = (vq.get("speechmetrics") or {}).get("mosnet_score")
                if mosnet is not None:
                    acc[tts]["MOSNet"].append(mosnet)

                ln = librosa_naturalness(vq.get("librosa") or {})
                if ln is not None:
                    acc[tts]["Librosa Naturalness"].append(ln)

                llm = (vq.get("llm_judge") or {}).get("llm_overall")
                if llm is not None:
                    acc[tts]["LLM Holistic"].append(llm)

                base = (vq.get("minicpm") or {}).get("naturalness")
                if base is not None:
                    acc[tts]["MiniCPM Baseline"].append(base)

                ft = (vq.get("minicpm_finetuned") or {}).get("naturalness")
                if ft is not None:
                    acc[tts]["MiniCPM Fine-tuned"].append(ft)

    means = {}
    for tts, md in acc.items():
        means[tts] = {m: (float(np.mean(v)) if v else None) for m, v in md.items()}
    return means


def place_on_axes(means: dict):
    """Map each raw value onto its axis's own fixed native range -> [0,1].

    This is a geometry-only transform: (v - lo) / (hi - lo) using the FIXED
    native scale of each method (AXIS_RANGE), not a data-driven min/max. So the
    plotted radius reflects the raw score on its own scale; providers are not
    rescaled relative to each other, and flat metrics (e.g. NISQA near its
    ceiling) render as near-identical polygons rather than being stretched.
    """
    providers = [t for t in KNOWN_TTS if t in means]
    placed = {t: {} for t in providers}
    for m in METHODS:
        lo, hi = AXIS_RANGE[m]
        span = hi - lo
        for t in providers:
            v = means[t].get(m)
            if v is None:
                placed[t][m] = 0.0
            else:
                placed[t][m] = max(0.0, min((v - lo) / span, 1.0))
    return providers, placed


def plot(providers, placed, out_path: str):
    n = len(METHODS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))

    # Draw providers in the paper's legend order, with matched colors.
    draw = [t for t in TTS_ORDER if t in providers]
    for tts in draw:
        vals = [placed[tts][m] for m in METHODS]
        vals += vals[:1]
        color = TTS_COLOR[tts]
        ax.plot(angles, vals, color=color, linewidth=2.2, label=TTS_LABEL[tts],
                zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.08, zorder=2)

    # Match paper orientation: NISQA at 0 deg (East), axes going counter-clockwise.
    ax.set_theta_offset(0.0)
    ax.set_theta_direction(1)

    # Two-line axis labels matching the paper layout.
    axis_labels = [
        "NISQA\nMOS",
        "MOSNet",
        "Librosa\nNaturalness",
        "LLM\nHolistic",
        "MiniCPM\nBaseline",
        "MiniCPM\nFine-tuned",
    ]
    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels, fontsize=15)
    # Push the angular tick labels outward so they clear the outer ring.
    ax.tick_params(axis="x", pad=18)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=11, color="gray")
    ax.set_rlabel_position(22)
    ax.grid(True, color="0.8", linewidth=0.8)

    ax.set_title("TTS Quality: Multi-Method Comparison\n(all metrics normalised to 0-1)",
                 fontsize=18, fontweight="bold", pad=44)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=13,
              frameon=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.82, top=0.82, bottom=0.14)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Cross-method TTS quality radar")
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    means = collect(args.eval_dir)

    # Report raw per-provider means for transparency before normalization.
    print("Raw per-provider means (pre-normalization):")
    hdr = "  {:14s}".format("TTS") + "".join(f"{m[:10]:>12s}" for m in METHODS)
    print(hdr)
    for t in [t for t in KNOWN_TTS if t in means]:
        row = "  {:14s}".format(TTS_LABEL[t])
        for m in METHODS:
            v = means[t].get(m)
            row += f"{('%.2f' % v) if v is not None else 'n/a':>12s}"
        print(row)

    providers, placed = place_on_axes(means)
    if not providers:
        print("No provider data found; nothing to plot.")
        return
    plot(providers, placed, args.out)


if __name__ == "__main__":
    main()
