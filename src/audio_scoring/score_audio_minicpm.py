#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Score audio files using MiniCPM-o 4.5 for naturalness and clarity.

Walks evaluation JSON files, resolves the matching local .wav from tts_audio/,
scores each with MiniCPM-o, and writes results back into the JSON.

Usage:
    # Single file
    python score_audio_minicpm.py <wav_file>

    # Batch: score all TTS audio referenced by evaluation JSONs
    python score_audio_minicpm.py --batch \
        --eval-dir lily-data/evaluation_output/evaluation \
        --audio-dir lily-data/tts_audio

    # Resume from where you left off (skips already-scored entries)
    python score_audio_minicpm.py --batch --resume

    # Process a single combo folder
    python score_audio_minicpm.py --batch --combo deepgram_nova2_cartesia
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback

import librosa
import torch
from transformers import AutoModel

# Defaults (relative to server/)
DEFAULT_EVAL_DIR = "lily-data/evaluation_output/evaluation"
DEFAULT_AUDIO_DIR = "lily-data/tts_audio"

# Mapping for folder-name mismatches between evaluation/ and tts_audio/
EVAL_TO_AUDIO_FOLDER = {
    "whisper_large": "whisper_large_v2",
}

SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud.\n\n"
    'Respond ONLY with JSON: {"naturalness": <int>, "noisiness": <int>, "loudness": <int>}'
)


def resolve_audio_folder(eval_folder_name: str) -> str:
    """Map an evaluation folder name to its tts_audio counterpart."""
    for old, new in EVAL_TO_AUDIO_FOLDER.items():
        eval_folder_name = eval_folder_name.replace(old, new)
    return eval_folder_name


def resolve_audio_path(eval_entry: dict, eval_folder_name: str, audio_dir: str) -> str | None:
    """Resolve the local .wav path for an evaluation entry.

    Tries:
      1. tts_audio_path field → extract combo/filename, look locally
      2. question_id → construct expected filename
    """
    s3_path = eval_entry.get("tts_audio_path", "")
    audio_folder = resolve_audio_folder(eval_folder_name)

    # Try from S3 path: .../tts_audio/<combo>/<file>.wav
    if s3_path:
        parts = s3_path.rstrip("/").split("/")
        # Find "tts_audio" in the path and take the rest
        try:
            idx = parts.index("tts_audio")
            rel = "/".join(parts[idx + 1:])
            # The S3 combo folder may also differ from local
            s3_combo = parts[idx + 1] if len(parts) > idx + 1 else ""
            local_combo = resolve_audio_folder(s3_combo)
            filename = parts[-1] if parts else ""
            candidate = os.path.join(audio_dir, local_combo, filename)
            if os.path.isfile(candidate):
                return candidate
        except ValueError:
            pass

    # Fallback: construct from question_id
    qid = eval_entry.get("question_id", "")
    if qid:
        candidate = os.path.join(audio_dir, audio_folder, f"{qid}_response.wav")
        if os.path.isfile(candidate):
            return candidate

    return None


def load_model(mode: str = "baseline", lora_path: str | None = None):
    """Load MiniCPM-o 4.5 with optional LoRA."""
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-o-4_5",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    if mode == "finetuned" and lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval().cuda()
    return model


def score_audio(model, audio_path: str) -> dict:
    """Score a single audio file, returning {"naturalness": int, "clarity": int}."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    msgs = [{"role": "user", "content": [SCORING_PROMPT, audio]}]
    response = model.chat(
        msgs=msgs,
        do_sample=False,
        max_new_tokens=256,
        use_tts_template=False,
        generate_audio=False,
    )
    try:
        scores = json.loads(response)
        # Validate expected keys
        if "naturalness" in scores and "noisiness" in scores and "loudness" in scores:
            return scores
        return {"raw_response": response, "error": "Missing expected keys"}
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{[^}]+\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_response": response, "error": "Failed to parse JSON"}


def run_single(model, wav_file: str):
    """Score a single .wav and print results."""
    result = score_audio(model, wav_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def score_key(mode: str) -> str:
    """Return the voice_quality sub-key for the given mode."""
    return "minicpm_finetuned" if mode == "finetuned" else "minicpm"


def has_minicpm_scores(entry: dict, mode: str = "baseline") -> bool:
    """Check if an evaluation entry already has MiniCPM scores for the given mode."""
    vq = entry.get("voice_quality") or {}
    minicpm = vq.get(score_key(mode)) or {}
    return bool(minicpm) and "error" not in minicpm


def run_batch(model, eval_dir: str, audio_dir: str, resume: bool = False,
              combo_filter: str | None = None, mode: str = "baseline"):
    """Walk evaluation JSONs, score referenced audio, write results back."""
    combo_dirs = sorted(glob.glob(os.path.join(eval_dir, "*")))
    if combo_filter:
        combo_dirs = [d for d in combo_dirs if os.path.basename(d) == combo_filter]
        if not combo_dirs:
            print(f"No combo folder matching '{combo_filter}' found in {eval_dir}")
            return

    # Collect all (json_path, eval_folder_name) pairs
    all_jsons = []
    for combo_path in combo_dirs:
        if not os.path.isdir(combo_path):
            continue
        combo_name = os.path.basename(combo_path)
        for jf in sorted(glob.glob(os.path.join(combo_path, "*_evaluation.json"))):
            all_jsons.append((jf, combo_name))

    total = len(all_jsons)
    scored = 0
    skipped = 0
    errors = 0
    missing_audio = 0

    print(f"Found {total} evaluation JSONs across {len(combo_dirs)} combo folders")

    for idx, (json_path, combo_name) in enumerate(all_jsons, 1):
        short_name = f"{combo_name}/{os.path.basename(json_path)}"
        print(f"\n[{idx}/{total}] {short_name}")

        try:
            with open(json_path, "r") as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ERROR reading JSON: {e}")
            errors += 1
            continue

        evaluations = eval_data.get("evaluations", [])
        if not evaluations:
            print("  No evaluations found, skipping")
            continue

        modified = False
        for entry_idx, entry in enumerate(evaluations):
            qid = entry.get("question_id", f"entry_{entry_idx}")

            if resume and has_minicpm_scores(entry, mode):
                skipped += 1
                continue

            audio_path = resolve_audio_path(entry, combo_name, audio_dir)
            if not audio_path:
                print(f"  [{entry_idx}] {qid}: audio not found locally, skipping")
                missing_audio += 1
                continue

            try:
                result = score_audio(model, audio_path)
                if "voice_quality" not in entry:
                    entry["voice_quality"] = {}
                entry["voice_quality"][score_key(mode)] = result
                modified = True
                scored += 1

                nat = result.get("naturalness", "?")
                noi = result.get("noisiness", "?")
                lou = result.get("loudness", "?")
                print(f"  [{entry_idx}] {qid}: naturalness={nat}, noisiness={noi}, loudness={lou}")
            except Exception as e:
                print(f"  [{entry_idx}] {qid}: ERROR scoring - {e}")
                traceback.print_exc()
                errors += 1

        if modified:
            with open(json_path, "w") as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            print(f"  -> Updated {json_path}")

    print(f"\nDone. Scored: {scored}, Skipped (resume): {skipped}, "
          f"Missing audio: {missing_audio}, Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Score audio with MiniCPM-o 4.5 (naturalness & clarity)"
    )
    parser.add_argument("wav_file", nargs="?", help="Path to a .wav file (single mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Score all TTS audio referenced by evaluation JSONs")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                        help=f"Evaluation JSON directory (default: {DEFAULT_EVAL_DIR})")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,
                        help=f"Local TTS audio directory (default: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--mode", choices=["baseline", "finetuned"], default="baseline",
                        help="baseline = original model, finetuned = LoRA-adapted")
    parser.add_argument("--lora-path", default="data/trained_models/minicpm_audio_scorer_lora",
                        help="Path to LoRA weights (finetuned mode only)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries that already have minicpm scores")
    parser.add_argument("--combo", default=None,
                        help="Process only this combo folder (e.g. deepgram_nova2_cartesia)")
    args = parser.parse_args()

    if not args.batch and not args.wav_file:
        parser.error("Provide a wav_file or use --batch")

    print(f"Loading MiniCPM-o 4.5 ({args.mode})...")
    model = load_model(args.mode, args.lora_path)

    if args.batch:
        run_batch(model, args.eval_dir, args.audio_dir,
                  resume=args.resume, combo_filter=args.combo, mode=args.mode)
    else:
        print(f"Scoring: {args.wav_file}")
        run_single(model, args.wav_file)


if __name__ == "__main__":
    main()
