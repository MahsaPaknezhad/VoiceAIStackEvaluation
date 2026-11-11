#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Build a SOMOS labels JSON for MiniCPM MOS (1-5) fine-tuning.

Reads the official SOMOS split MOS lists (train/valid/test_mos_list.txt, each a
CSV of `utteranceId,mean`) and the extracted `audios/` folder, and emits a flat
labels JSON compatible with finetune_somos_mos.py:

    [
      {"audio_path": "/abs/path/booksent_2012_0005_001.wav",
       "sample_id": "booksent_2012_0005_001",
       "split": "train" | "valid" | "test",
       "mos": 3.76},
      ...
    ]

The MOS `mean` values are real human Mean Opinion Scores on a 1-5 scale.

Usage (subset of 2000, preserving official split proportions):
    .venv/bin/python src/audio_scoring/build_somos_labels.py --limit 2000

Usage (full 20,100 clips):
    .venv/bin/python src/audio_scoring/build_somos_labels.py
"""
import argparse
import csv
import json
import os
import random

SOMOS_ROOT = os.environ.get(
    "SOMOS_ROOT",
    "scratch_dataset_samples/somos_full/extracted",
)
SPLIT_DIR = os.path.join(SOMOS_ROOT, "training_files", "split1", "full")
AUDIO_DIR = os.path.join(SOMOS_ROOT, "audios")
OUT_FILE = os.environ.get(
    "SOMOS_LABELS_FILE",
    "data/training_dataset/labels/somos_mos_labels.json",
)

SPLIT_FILES = {
    "train": "train_mos_list.txt",
    "valid": "valid_mos_list.txt",
    "test": "test_mos_list.txt",
}
MOS_MIN, MOS_MAX = 1.0, 5.0


def read_split(split_name, fname):
    path = os.path.join(SPLIT_DIR, fname)
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip utteranceId,mean
        for parts in reader:
            if len(parts) < 2:
                continue
            clip, mean = parts[0].strip(), parts[1].strip()
            try:
                mos = float(mean)
            except ValueError:
                continue
            # Clamp defensively; SOMOS means already lie in [1,5].
            mos = max(MOS_MIN, min(MOS_MAX, mos))
            sample_id = clip[:-4] if clip.lower().endswith(".wav") else clip
            audio_path = os.path.abspath(os.path.join(AUDIO_DIR, clip))
            rows.append({
                "audio_path": audio_path,
                "sample_id": sample_id,
                "split": split_name,
                "mos": round(mos, 4),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description="Build SOMOS MOS labels JSON")
    p.add_argument("--limit", type=int, default=None,
                   help="Total number of clips to keep (sampled per-split, "
                        "proportionally). Default: all clips.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-audio", action="store_true", default=True,
                   help="Drop entries whose WAV file is missing (default on).")
    p.add_argument("--allow-missing-audio", dest="require_audio",
                   action="store_false",
                   help="Keep entries even if the WAV is missing.")
    p.add_argument("--out", default=OUT_FILE)
    args = p.parse_args()

    rng = random.Random(args.seed)

    per_split = {}
    for split_name, fname in SPLIT_FILES.items():
        rows = read_split(split_name, fname)
        if args.require_audio:
            kept = [r for r in rows if os.path.exists(r["audio_path"])]
            missing = len(rows) - len(kept)
            if missing:
                print(f"  {split_name}: {missing} entries dropped (audio missing)")
            rows = kept
        per_split[split_name] = rows
        print(f"  {split_name}: {len(rows)} clips available")

    total_available = sum(len(v) for v in per_split.values())
    if total_available == 0:
        print("ERROR: no labeled clips found. Check SOMOS_ROOT / extraction.")
        return

    # Optional proportional subsetting across the official splits.
    if args.limit and args.limit < total_available:
        labels = []
        for split_name, rows in per_split.items():
            frac = len(rows) / total_available
            k = max(1, round(args.limit * frac))
            k = min(k, len(rows))
            labels.extend(rng.sample(rows, k))
        # Trim any rounding overflow back to the requested limit.
        rng.shuffle(labels)
        labels = labels[:args.limit]
    else:
        labels = [r for rows in per_split.values() for r in rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(labels, f, indent=2)

    from collections import Counter
    dist = Counter(r["split"] for r in labels)
    mos_vals = [r["mos"] for r in labels]
    print(f"\nWrote {len(labels)} labels -> {args.out}")
    print(f"  split distribution: {dict(dist)}")
    print(f"  MOS range: {min(mos_vals):.2f} - {max(mos_vals):.2f}, "
          f"mean {sum(mos_vals)/len(mos_vals):.3f}")


if __name__ == "__main__":
    main()
