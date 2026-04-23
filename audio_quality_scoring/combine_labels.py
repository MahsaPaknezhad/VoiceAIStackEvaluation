#!/usr/bin/env python3
"""
Combine per-metric label files into a single labels.json for training.

Reads:
    evaluation_data/training_dataset/labels/naturalness.json
    evaluation_data/training_dataset/labels/noisiness.json
    evaluation_data/training_dataset/labels/loudness.json

Writes:
    evaluation_data/training_dataset/labels/labels.json

Only includes samples that have ALL three metrics labeled.
"""

import json
import os

LABELS_DIR = "evaluation_data/training_dataset/labels"
METRICS = ["naturalness", "noisiness", "loudness"]
OUTPUT = os.path.join(LABELS_DIR, "labels.json")


def main():
    # Load each metric file into {sample_id: label_entry}
    per_metric = {}
    for metric in METRICS:
        path = os.path.join(LABELS_DIR, f"{metric}.json")
        if not os.path.exists(path):
            print(f"Missing: {path} — run label_audio.py --metric {metric} first")
            return
        with open(path) as f:
            entries = json.load(f)
        per_metric[metric] = {e["sample_id"]: e for e in entries}
        print(f"  {metric}: {len(entries)} labels")

    # Find samples with all 3 metrics
    common_ids = set.intersection(*(set(d.keys()) for d in per_metric.values()))
    print(f"\nSamples with all 3 metrics: {len(common_ids)}")

    # Merge
    combined = []
    for sid in sorted(common_ids):
        entry = {
            "audio_path": per_metric["naturalness"][sid]["audio_path"],
            "sample_id": sid,
            "split": per_metric["naturalness"][sid]["split"],
        }
        for metric in METRICS:
            entry[metric] = per_metric[metric][sid][metric]
        combined.append(entry)

    with open(OUTPUT, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved {len(combined)} combined labels to {OUTPUT}")


if __name__ == "__main__":
    main()
