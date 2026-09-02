# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Weight-sensitivity analysis for the composite STT x TTS ranking.

Re-computes the composite ranking under several weighting profiles to test
whether the top combinations are robust to the (necessarily subjective) choice
of weights. Reuses the exact scoring logic in rank_combinations.py so results
match the main ranking.

Weight vector order (7 values, summing to 100), matching rank_combinations.py:
    WER, LLM Judge, Latency, Voice LLM, MiniCPM naturalness, noisiness, loudness

Profiles reported in the paper (Appendix "Weight Sensitivity"):
    default              30/10/30/10/7/7/6   general-purpose interactive assistant
    accuracy-first       50/10/20/8/4/4/4    transcription-critical
    latency-first        20/10/50/8/4/4/4    real-time / telephony
    voice-quality-first  20/20/20/15/9/8/8   long-form narration
    equal-thirds         33/6/33/10/6/6/6    WER / latency / quality balanced

Usage:
    python visualize/weight_sensitivity.py \
        --eval-dir output/eval-results/evaluation_output/evaluation
    python visualize/weight_sensitivity.py --top 5 --latex
"""

import argparse
import os
import sys

# Import the shared ranking logic from rank_combinations.py in this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_combinations as rc  # noqa: E402

DEFAULT_EVAL_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "output/eval-results/evaluation_output/evaluation",
)

# name -> weight vector (percentages summing to 100), in rank_combinations order.
PROFILES = {
    "default": [30, 10, 30, 10, 7, 7, 6],
    "accuracy-first": [50, 10, 20, 8, 4, 4, 4],
    "latency-first": [20, 10, 50, 8, 4, 4, 4],
    "voice-quality-first": [20, 20, 20, 15, 9, 8, 8],
    "equal-thirds": [33, 6, 33, 10, 6, 6, 6],
}


def rank_with_weights(combos, weight_pcts):
    """Return combos ranked (desc) under the given percentage weight vector.

    combos is re-scored in place for each call; we deep-copy the composite by
    reading it out immediately after compute_composite.
    """
    w = [x / 100.0 for x in weight_pcts]
    assert abs(sum(w) - 1.0) < 0.02, f"weights must sum to 100, got {sum(weight_pcts)}"
    weights = {
        name: {"weight": w[i], "direction": rc.METRIC_DIRS[i]}
        for i, name in enumerate(rc.METRIC_NAMES)
    }
    rc.compute_composite(combos, weights)
    ranked = sorted(combos.items(), key=lambda kv: kv[1]["composite"], reverse=True)
    return [(c["stt"], c["tts"], c["composite"]) for _, c in ranked]


def main():
    p = argparse.ArgumentParser(description="Composite-ranking weight sensitivity")
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--top", type=int, default=5, help="rows per profile")
    p.add_argument("--latex", action="store_true", help="emit a LaTeX table")
    args = p.parse_args()

    combos = rc.collect_metrics(args.eval_dir)
    if not combos:
        print(f"No combinations found in {args.eval_dir}")
        return

    results = {name: rank_with_weights(combos, w)[: args.top]
               for name, w in PROFILES.items()}

    # Plain-text report.
    for name, w in PROFILES.items():
        print(f"\n=== {name}  (weights {'/'.join(map(str, w))}) ===")
        for i, (stt, tts, score) in enumerate(results[name], 1):
            print(f"  {i}. {stt} x {tts}  ({score:.3f})")

    # Robustness summary.
    firsts = {results[n][0][:2] for n in PROFILES}
    print("\nRank-1 combination across all profiles:",
          "STABLE" if len(firsts) == 1 else "VARIES", firsts)

    if args.latex:
        profiles = list(PROFILES.keys())
        print("\n% --- LaTeX table ---")
        print("\\begin{tabular}{r" + "l" * len(profiles) + "}")
        print("\\toprule")
        print("Rk & " + " & ".join(n.replace("-", "-") for n in profiles) + " \\\\")
        print("\\midrule")
        for i in range(args.top):
            cells = []
            for n in profiles:
                stt, tts, score = results[n][i]
                cells.append(f"{stt}$\\times${tts} ({score:.3f})")
            print(f"{i + 1} & " + " & ".join(cells) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
