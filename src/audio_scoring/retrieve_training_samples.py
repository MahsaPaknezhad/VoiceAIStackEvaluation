#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Retrieve 100 diverse audio samples from VoiceAssistant-Eval splits NOT used
by the existing evaluation dataset, for fine-tuning training data.

Existing data uses only 'listening_general' (indices 1-4,6,7,9-16,19-22,28,29).
This script pulls from the remaining 12 splits plus unused listening_general samples.
"""

import json
import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Splits the existing eval script already used
EXISTING_SPLIT = "listening_general"
EXISTING_INDICES = {1,2,3,4,6,7,9,10,11,12,13,14,15,16,19,20,21,22,28,29}

# All other splits — completely unused by existing data
UNUSED_SPLITS = [
    "listening_music",
    "listening_sound",
    "listening_speech",
    "speaking_assistant",
    "speaking_emotion",
    "speaking_instruction_following",
    "speaking_multi_round",
    "speaking_reasoning",
    "speaking_robustness",
    "speaking_roleplay",
    "speaking_safety",
    "viewing_multi_discipline",
]

OUTPUT_DIR = "evaluation_data/training_dataset"
TOTAL_SAMPLES = 100
SAMPLES_PER_UNUSED_SPLIT = 8  # 8 * 12 = 96, plus 4 from listening_general unused indices


def save_audio(audio_bytes, path):
    with open(path, "wb") as f:
        f.write(audio_bytes)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    samples = []
    count = 0

    # Pull from the 12 completely unused splits
    for split in UNUSED_SPLITS:
        if count >= TOTAL_SAMPLES:
            break

        logger.info(f"Loading split: {split}")
        try:
            data = load_dataset("MathLLMs/VoiceAssistant-Eval", split)
            test_data = data["test"]
        except Exception as e:
            logger.error(f"Failed to load {split}: {e}")
            continue

        added = 0
        for i in range(len(test_data)):
            if added >= SAMPLES_PER_UNUSED_SPLIT or count >= TOTAL_SAMPLES:
                break

            sample = test_data[i]
            audio = sample.get("user_audio_0")
            if audio is None:
                continue

            sample_id = f"{split}_{added}"
            audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
            save_audio(audio, audio_path)

            transcript = ""
            if sample.get("extra") and sample["extra"].get("user_audio_transcripts"):
                transcripts = sample["extra"]["user_audio_transcripts"]
                if transcripts:
                    transcript = transcripts[0]

            samples.append({
                "id": sample_id,
                "audio_path": os.path.abspath(audio_path),
                "split": split,
                "transcript": transcript,
                "original_index": i,
            })
            added += 1
            count += 1

        logger.info(f"  Added {added} samples from {split} (total: {count})")

    # Fill remaining from listening_general using UNUSED indices
    if count < TOTAL_SAMPLES:
        logger.info(f"Loading listening_general for {TOTAL_SAMPLES - count} more samples")
        try:
            data = load_dataset("MathLLMs/VoiceAssistant-Eval", EXISTING_SPLIT)
            test_data = data["test"]

            for i in range(len(test_data)):
                if count >= TOTAL_SAMPLES:
                    break
                if i in EXISTING_INDICES:
                    continue

                sample = test_data[i]
                audio = sample.get("user_audio_0")
                if audio is None:
                    continue

                sample_id = f"listening_general_extra_{count}"
                audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
                save_audio(audio, audio_path)

                transcript = ""
                if sample.get("extra") and sample["extra"].get("user_audio_transcripts"):
                    transcripts = sample["extra"]["user_audio_transcripts"]
                    if transcripts:
                        transcript = transcripts[0]

                samples.append({
                    "id": sample_id,
                    "audio_path": os.path.abspath(audio_path),
                    "split": EXISTING_SPLIT,
                    "transcript": transcript,
                    "original_index": i,
                })
                count += 1
        except Exception as e:
            logger.error(f"Failed to load listening_general: {e}")

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "samples.json")
    with open(manifest_path, "w") as f:
        json.dump({"total": len(samples), "samples": samples}, f, indent=2)

    # Print summary
    from collections import Counter
    split_counts = Counter(s["split"] for s in samples)
    print(f"\nSaved {len(samples)} samples to {OUTPUT_DIR}/")
    print(f"Manifest: {manifest_path}")
    print("\nSamples per split:")
    for split, c in sorted(split_counts.items()):
        print(f"  {split}: {c}")


if __name__ == "__main__":
    main()
