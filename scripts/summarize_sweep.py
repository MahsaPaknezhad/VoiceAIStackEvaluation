#!/usr/bin/env python3
"""Aggregate sweep test_correlation.json reports into a comparison table.

Includes the previous single run (minicpm_audio_scorer_lora_rerun) as a baseline
reference if present. Sorts by mean |Pearson r| across the three metrics.
"""
import glob
import json
import os

ROOTS = [
    "data/trained_models/sweep/*",
    "data/trained_models/sweep_reg/*",
    "data/trained_models/cmp/*",
    "data/trained_models/minicpm_audio_scorer_lora_rerun",
]
METRICS = ["naturalness", "noisiness", "loudness"]


def load_reports():
    paths = []
    for r in ROOTS:
        paths.extend(glob.glob(r))
    rows = []
    for p in sorted(set(paths)):
        rep = os.path.join(p, "test_correlation.json")
        if not os.path.isfile(rep):
            continue
        with open(rep) as f:
            d = json.load(f)
        # Qualify the run name with its parent dir so same-named configs in
        # different sweep trees (e.g. sweep/baseline vs sweep_reg/baseline) don't
        # collide. The rerun single-run dir has no meaningful parent, so keep it
        # as just its basename.
        parent = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)
        name = f"{parent}/{base}" if parent.startswith("sweep") or parent in ("cmp",) else base
        row = {"name": name, "n": d.get("n_test_scored")}
        rs = []
        for m in METRICS:
            mm = d["metrics"][m]
            row[f"{m}_r"] = mm["pearson_r"]
            row[f"{m}_rho"] = mm["spearman_rho"]
            row[f"{m}_mae"] = mm["mae"]
            if mm["pearson_r"] is not None:
                rs.append(abs(mm["pearson_r"]))
        row["mean_abs_r"] = sum(rs) / len(rs) if rs else 0.0
        rows.append(row)
    return sorted(rows, key=lambda x: x["mean_abs_r"], reverse=True)


def fmt(x):
    return "  nan" if x is None else f"{x:+.3f}"


def main():
    rows = load_reports()
    if not rows:
        print("No reports found yet.")
        return
    hdr = f"{'run':<34} {'n':>3} " + " ".join(f"{m[:4]}_r  {m[:4]}_rho {m[:4]}_mae" for m in METRICS) + "  mean|r|"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['name']:<34} {r['n']:>3} "
        for m in METRICS:
            line += f"{fmt(r[f'{m}_r'])} {fmt(r[f'{m}_rho'])} {r[f'{m}_mae']:>6.2f} "
        line += f"  {r['mean_abs_r']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
