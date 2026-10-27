"""
Build a blinded, stratified listening-test sample to human-validate the
provider-level naturalness ranking (addresses reviewer R2-W2).

Design (ITU-T P.800 ACR / MOS conventions):
  * Stimuli: TTS provider outputs under output/eval-results/tts_audio/.
  * 5 providers x N prompts = 5N clips. The SAME N prompt IDs are drawn for
    every provider (content-balanced at the set level). Because the upstream
    LLM text is not identical across providers, this is an UNPAIRED ACR test
    rather than a paired MUSHRA/CMOS test; we therefore average over enough
    prompts and rely on the common prompt-ID set to balance content.
  * One fixed STT front-end per provider is used so provider is the only
    deliberately varied factor in the TTS stage.
  * Blinding: every clip is copied to an opaque ID (clip_0001.wav ...). The
    provider mapping lives only in hidden_key.json, not in any annotator file.
  * Randomisation: each annotator gets an independently shuffled playlist with
    a no-two-consecutive-same-provider constraint to avoid order/assimilation
    bias.
  * Rating scale: 5-point ACR naturalness (1=Bad ... 5=Excellent), the P.800
    standard. Annotators rate one clip at a time.
  * Reliability: the same blinded set is rated by all annotators, enabling
    inter-rater agreement (ICC/Krippendorff) and a consensus provider ranking.

Outputs (under results/provider_mos/):
  clips/clip_XXXX.wav            blinded audio
  hidden_key.json                clip_id -> {provider, stt, question_id, src}
  playlist_<annotator>.csv       randomised rating sheet (clip_id, rating col)
  manifest.json                  full design record (seed, n, providers, order)

Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python build_provider_mos_sample.py \
      --n-per-provider 20 --annotators labeler1 labeler2 labeler3 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import shutil

EVAL_JSON = ("/home/ubuntu/VoiceAIStackEvaluation/output/eval-results/"
             "evaluation_output/evaluation")
AUDIO_ROOT = "/home/ubuntu/VoiceAIStackEvaluation/output/eval-results/tts_audio"
OUT_ROOT = os.path.join(os.path.dirname(__file__), "results", "provider_mos")

PROVIDERS = ["aws_polly", "cartesia", "deepgram_aura", "groq", "nvidia_magpie"]
PRETTY = {"aws_polly": "AWS Polly", "cartesia": "Cartesia",
          "deepgram_aura": "Deepgram Aura", "groq": "Groq",
          "nvidia_magpie": "NVIDIA Magpie"}
# One fixed STT front-end per provider (all exist locally).
FIXED_STT = {p: "nvidia_parakeet" for p in PROVIDERS}


def common_prompt_ids() -> list[str]:
    """question_ids present for every provider (under its fixed STT folder)."""
    sets = []
    for p in PROVIDERS:
        combo = f"{FIXED_STT[p]}_{p}"
        ids = set()
        for f in glob.glob(os.path.join(EVAL_JSON, combo, "*_evaluation.json")):
            d = json.load(open(f))
            for ev in d.get("evaluations", []):
                ids.add(ev["question_id"])
        sets.append(ids)
    return sorted(set.intersection(*sets), key=_qid_key)


def _qid_key(q):
    m = re.search(r"(\d+)$", q)
    return (q[:m.start()], int(m.group(1))) if m else (q, 0)


def audio_path(provider: str, qid: str) -> str | None:
    combo = f"{FIXED_STT[provider]}_{provider}"
    cand = os.path.join(AUDIO_ROOT, combo, f"{qid}_response.wav")
    return cand if os.path.exists(cand) else None


def constrained_shuffle(items, key_fn, rng, max_tries=200):
    """
    Order items so adjacent items avoid sharing the same key, using greedy
    max-spacing: at each step pick (randomly) among the keys with the most
    remaining items, excluding the key just placed. This reliably produces a
    valid de-correlated order whenever one exists (balanced design here).
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    for k in buckets:
        rng.shuffle(buckets[k])

    order = []
    last_key = None
    total = len(items)
    while len(order) < total:
        # candidate keys with items left, excluding the last placed key
        cands = [(k, len(v)) for k, v in buckets.items() if v and k != last_key]
        if not cands:
            # only the last_key remains; unavoidable adjacency
            cands = [(k, len(v)) for k, v in buckets.items() if v]
        maxlen = max(c[1] for c in cands)
        top = [k for k, n in cands if n == maxlen]
        choice = rng.choice(top)
        order.append(buckets[choice].pop())
        last_key = choice
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-provider", type=int, default=20)
    ap.add_argument("--annotators", nargs="+",
                    default=["labeler1", "labeler2", "labeler3"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(os.path.join(OUT_ROOT, "clips"), exist_ok=True)

    pool = common_prompt_ids()
    if len(pool) < args.n_per_provider:
        raise SystemExit(f"Only {len(pool)} common prompts; "
                         f"requested {args.n_per_provider}.")
    chosen_qids = rng.sample(pool, args.n_per_provider)

    # Build blinded clip set: same prompt IDs for every provider.
    clips = []          # list of dicts
    key = {}            # clip_id -> provenance (hidden)
    missing = []
    cid = 1
    for provider in PROVIDERS:
        for qid in chosen_qids:
            src = audio_path(provider, qid)
            if src is None:
                missing.append((provider, qid))
                continue
            clip_id = f"clip_{cid:04d}"
            dst = os.path.join(OUT_ROOT, "clips", f"{clip_id}.wav")
            shutil.copyfile(src, dst)
            clips.append({"clip_id": clip_id, "provider": provider,
                          "question_id": qid})
            key[clip_id] = {"provider": provider,
                            "provider_pretty": PRETTY[provider],
                            "stt": FIXED_STT[provider],
                            "question_id": qid, "source_path": src}
            cid += 1

    # Hidden key (NOT given to annotators)
    with open(os.path.join(OUT_ROOT, "hidden_key.json"), "w") as f:
        json.dump(key, f, indent=2)

    # Per-annotator randomised, de-correlated playlists
    playlists = {}
    for ann in args.annotators:
        ann_rng = random.Random(args.seed + hash(ann) % 10_000)
        order = constrained_shuffle([dict(c) for c in clips],
                                    key_fn=lambda c: c["provider"], rng=ann_rng)
        playlists[ann] = [c["clip_id"] for c in order]
        csv_path = os.path.join(OUT_ROOT, f"playlist_{ann}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["position", "clip_id", "audio_file",
                        "naturalness_1to5"])
            for pos, c in enumerate(order, 1):
                w.writerow([pos, c["clip_id"], f"clips/{c['clip_id']}.wav", ""])

    manifest = {
        "design": "ITU-T P.800 ACR (Absolute Category Rating), unpaired",
        "rating_scale": {"1": "Bad", "2": "Poor", "3": "Fair",
                         "4": "Good", "5": "Excellent"},
        "attribute": "naturalness",
        "seed": args.seed,
        "n_per_provider": args.n_per_provider,
        "providers": PROVIDERS,
        "fixed_stt_per_provider": FIXED_STT,
        "chosen_question_ids": chosen_qids,
        "n_clips_total": len(clips),
        "annotators": args.annotators,
        "blinding": "clips renamed to opaque IDs; provider only in hidden_key.json",
        "order_constraint": "no two consecutive clips from same provider (best effort)",
        "missing_clips": missing,
        "analysis": "score with analyze_provider_mos.py after ratings are filled",
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built blinded ACR MOS sample at {OUT_ROOT}")
    print(f"  providers        : {len(PROVIDERS)}")
    print(f"  prompts/provider : {args.n_per_provider}")
    print(f"  total clips       : {len(clips)}"
          + (f"  (missing {len(missing)})" if missing else ""))
    print(f"  annotators       : {', '.join(args.annotators)}")
    print(f"  scale            : 5-point ACR naturalness (1 Bad .. 5 Excellent)")
    print(f"  rating sheets    : playlist_<annotator>.csv "
          f"(fill the naturalness_1to5 column)")


if __name__ == "__main__":
    main()
