#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Analyse the blinded human study (standard library only).

Usage:
    python src/audio_scoring/analyze_human_study.py [--boot 5000]

Reports:
  * per-provider mean naturalness with a 95% prompt-level bootstrap CI
  * pairwise provider differences (paired by prompt) with 95% CIs
  * inter-rater agreement (mean pairwise Spearman) and per-rater test-retest
    on the repeated clips. These are quick proxies for the agreement ceiling;
    compute ICC / Krippendorff's alpha for the paper.

Repeated clips are excluded from the provider statistics.
"""

import argparse
import glob
import itertools
import json
import os
import random
import statistics as st

STUDY_DIR = os.environ.get("HUMAN_STUDY_DIR", "data/human_study")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


def load_ratings():
    out = {}
    for f in sorted(glob.glob(os.path.join(STUDY_DIR, "ratings", "*.jsonl"))):
        r = {}
        with open(f) as fh:
            for line in fh:
                e = json.loads(line)
                if e["score"] is None:
                    r.pop(e["clip_id"], None)
                else:
                    r[e["clip_id"]] = e["score"]
        out[os.path.basename(f)[:-6]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(STUDY_DIR, "manifest.json")) as f:
        clips = json.load(f)["clips"]
    by_id = {c["clip_id"]: c for c in clips}
    ratings = load_ratings()
    if not ratings:
        raise SystemExit("No ratings found in data/human_study/ratings/")
    print(f"Raters: {', '.join(f'{r} ({len(v)} clips)' for r, v in ratings.items())}\n")

    originals = [c for c in clips if not c["repeat_of"]]
    providers = sorted({c["tts"] for c in originals})

    # Mean rating per (prompt, provider) across raters.
    cell = {}
    for c in originals:
        vals = [r[c["clip_id"]] for r in ratings.values() if c["clip_id"] in r]
        if vals:
            cell[(c["prompt_id"], c["tts"])] = st.mean(vals)
    prompts = sorted({p for p, t in cell if all((p, x) in cell for x in providers)})
    print(f"Prompts rated for all providers: {len(prompts)}")

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.boot):
        sample = [rng.choice(prompts) for _ in prompts]
        boots.append({t: st.mean(cell[(p, t)] for p in sample) for t in providers})

    print("\nProvider mean naturalness (1-10), 95% prompt-bootstrap CI")
    for t in sorted(providers, key=lambda t: -st.mean(cell[(p, t)] for p in prompts)):
        m = st.mean(cell[(p, t)] for p in prompts)
        ci = (pct([b[t] for b in boots], 0.025), pct([b[t] for b in boots], 0.975))
        print(f"  {t:15s} {m:5.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\nPairwise differences (A - B), paired by prompt; '*' = CI excludes 0")
    for a, b in itertools.combinations(providers, 2):
        d = st.mean(cell[(p, a)] - cell[(p, b)] for p in prompts)
        dist = [x[a] - x[b] for x in boots]
        lo, hi = pct(dist, 0.025), pct(dist, 0.975)
        print(f"  {a:14s} - {b:14s} {d:+5.2f}  [{lo:+.2f}, {hi:+.2f}] {'*' if lo > 0 or hi < 0 else ''}")

    if len(ratings) > 1:
        print("\nInter-rater Spearman (originals)")
        rs = []
        for (n1, r1), (n2, r2) in itertools.combinations(ratings.items(), 2):
            ids = [c["clip_id"] for c in originals if c["clip_id"] in r1 and c["clip_id"] in r2]
            rho = spearman([r1[i] for i in ids], [r2[i] for i in ids])
            rs.append(rho)
            print(f"  {n1} vs {n2}: {rho:.3f} (n={len(ids)})")
        print(f"  mean: {st.mean(rs):.3f}  <- rough ceiling for any automatic metric")

    print("\nTest-retest on repeated clips")
    for name, r in ratings.items():
        pairs = [(r[c["repeat_of"]], r[c["clip_id"]]) for c in clips
                 if c["repeat_of"] and c["clip_id"] in r and c["repeat_of"] in r]
        if len(pairs) >= 3:
            a, b = zip(*pairs)
            print(f"  {name}: Spearman {spearman(list(a), list(b)):.3f}, "
                  f"mean |diff| {st.mean(abs(x - y) for x, y in pairs):.2f} (n={len(pairs)})")

    out = os.path.join(STUDY_DIR, "clip_scores.json")
    with open(out, "w") as f:
        json.dump([{"clip_id": c["clip_id"], "prompt_id": c["prompt_id"], "tts": c["tts"],
                    "human_mean": cell.get((c["prompt_id"], c["tts"]))} for c in originals], f, indent=1)
    print(f"\nWrote per-clip human means to {out} (join on prompt_id + tts to correlate with NISQA etc.)")


if __name__ == "__main__":
    main()
