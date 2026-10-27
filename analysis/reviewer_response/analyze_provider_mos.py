"""
Score the blinded provider MOS listening test (run after annotators fill the
playlist_<annotator>.csv files produced by build_provider_mos_sample.py).

Produces (addresses R2-W2):
  * Per-provider human MOS (mean +/- 95% CI) from the 3-annotator consensus.
  * Inter-rater reliability on the provider clips (Krippendorff alpha, ICC).
  * Human provider ranking vs. the fine-tuned scorer's ranking (Spearman rho).
  * Whether humans place AWS Polly lowest and reproduce the metric ordering.

Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python analyze_provider_mos.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import agreement_stats as ag

MOS_DIR = os.path.join(os.path.dirname(__file__), "results", "provider_mos")
PROV_ANALYSIS = os.path.join(os.path.dirname(__file__), "results",
                             "provider_analysis.json")


def load_ratings():
    key = json.load(open(os.path.join(MOS_DIR, "hidden_key.json")))
    ratings = defaultdict(dict)  # annotator -> {clip_id: score}
    for csv_path in glob.glob(os.path.join(MOS_DIR, "playlist_*.csv")):
        ann = os.path.basename(csv_path)[len("playlist_"):-len(".csv")]
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                v = (row.get("naturalness_1to5") or "").strip()
                if v:
                    ratings[ann][row["clip_id"]] = float(v)
    return key, ratings


def main():
    if not os.path.isdir(MOS_DIR):
        raise SystemExit("No provider_mos/ dir. Run build_provider_mos_sample.py "
                         "and collect ratings first.")
    key, ratings = load_ratings()
    annotators = sorted(ratings)
    filled = {a: len(ratings[a]) for a in annotators}
    total = len(key)
    if not annotators or all(v == 0 for v in filled.values()):
        print("Ratings not collected yet. Fill playlist_<annotator>.csv "
              "(naturalness_1to5 column).")
        print(f"Clips to rate per annotator: {total}")
        return
    print(f"Annotators: {filled}  (of {total} clips each)")

    clip_ids = sorted(key)
    providers = sorted({key[c]["provider"] for c in clip_ids})

    # rating matrix (clips x annotators)
    mat = np.full((len(clip_ids), len(annotators)), np.nan)
    for i, c in enumerate(clip_ids):
        for j, a in enumerate(annotators):
            if c in ratings[a]:
                mat[i, j] = ratings[a][c]

    # ---- inter-rater reliability on provider clips ----
    alpha = ag.krippendorff_alpha_interval(mat)
    icc = ag.icc_2(mat)
    pw = ag.pairwise_correlations(mat, annotators)
    print(f"\nInter-rater reliability (provider clips): "
          f"Krippendorff alpha={alpha:.3f}, ICC(2,k)={icc['icc_2_k']:.3f}, "
          f"mean pairwise r={ag.mean_pairwise(pw):.3f}")

    # ---- per-provider human consensus MOS + bootstrap CI ----
    consensus = np.nanmean(mat, axis=1)  # per-clip mean over annotators
    by_prov = defaultdict(list)
    for i, c in enumerate(clip_ids):
        if not np.isnan(consensus[i]):
            by_prov[key[c]["provider"]].append(consensus[i])

    rng = np.random.default_rng(42)
    print("\nHuman consensus MOS by provider (5-pt ACR, higher=more natural):")
    human_rank = {}
    for p in providers:
        vals = np.array(by_prov[p])
        boots = [rng.choice(vals, len(vals)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        human_rank[p] = vals.mean()
        print(f"  {key[[c for c in clip_ids if key[c]['provider']==p][0]]['provider_pretty']:<16}"
              f" MOS={vals.mean():.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (n={len(vals)})")

    # ---- compare to fine-tuned scorer ranking ----
    order_human = sorted(providers, key=lambda p: -human_rank[p])
    print("\nHuman provider ranking (best -> worst):",
          " > ".join(key[[c for c in clip_ids if key[c]['provider']==p][0]]['provider_pretty']
                     for p in order_human))

    if os.path.exists(PROV_ANALYSIS):
        pa = json.load(open(PROV_ANALYSIS))
        scorer = pa["finetuned_bootstrap"]["point"]  # provider -> mean
        common = [p for p in providers if p in scorer]
        h = np.array([human_rank[p] for p in common])
        s = np.array([scorer[p] for p in common])
        rho, pval = stats.spearmanr(h, s)
        print(f"\nHuman vs fine-tuned scorer provider ranking: "
              f"Spearman rho={rho:.3f} (p={pval:.3g}, n={len(common)})")
        worst_h = min(common, key=lambda p: human_rank[p])
        worst_s = min(common, key=lambda p: scorer[p])
        print(f"  Humans rank {worst_h} lowest; scorer ranks {worst_s} lowest "
              f"-> {'AGREE' if worst_h==worst_s else 'DISAGREE'}")

    out = {
        "annotators": annotators, "n_clips": total,
        "krippendorff_alpha": alpha, "icc_2_k": icc["icc_2_k"],
        "human_mos_by_provider": {p: float(np.mean(by_prov[p])) for p in providers},
        "human_ranking": order_human,
    }
    with open(os.path.join(MOS_DIR, "provider_mos_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {MOS_DIR}/provider_mos_results.json")


if __name__ == "__main__":
    main()
