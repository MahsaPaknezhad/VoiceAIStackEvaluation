#!/usr/bin/env python3
"""Score audio files using MiniCPM-o 4.5 (baseline or fine-tuned).

Usage:
    # Single file
    python score_audio_minicpm.py <wav_file> --mode baseline

    # Batch: score all TTS audio from evaluation_output
    python score_audio_minicpm.py --batch --mode baseline
"""

import argparse
import glob
import json
import os
import re

import librosa
import torch
from transformers import AutoModel

LORA_PATH = "trained_models/minicpm_audio_scorer_lora"
TTS_AUDIO_DIR = "evaluation_output/tts_audio"

# Known STT prefixes to strip from folder names to get TTS model
STT_PREFIXES = [
    "aws_transcribe", "deepgram_nova2", "deepgram_nova3",
    "nvidia_riva", "whisper_large_v2", "whisper_small", "whisper_turbo",
]

SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud.\n\n"
    'Respond ONLY with JSON: {"naturalness": <int>, "noisiness": <int>, "loudness": <int>}'
)


def extract_tts_model(folder_name):
    """Extract TTS model name from '<stt>_<tts>' folder name."""
    for prefix in sorted(STT_PREFIXES, key=len, reverse=True):
        if folder_name.startswith(prefix + "_"):
            return folder_name[len(prefix) + 1:]
    return folder_name


def load_model(mode: str):
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-o-4_5",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    if mode == "finetuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval().cuda()
    return model


def score_audio(model, audio_path: str) -> dict:
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
        return json.loads(response)
    except json.JSONDecodeError:
        return {"raw_response": response, "error": "Failed to parse JSON"}


def run_single(model, wav_file, mode):
    result = score_audio(model, wav_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_dir = os.path.join("scoring_output", mode)
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(wav_file))[0]
    output_file = os.path.join(output_dir, f"{basename}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_file}")


def run_batch(model, mode):
    """Score all TTS audio files, grouped by TTS model."""
    folders = sorted(glob.glob(os.path.join(TTS_AUDIO_DIR, "*")))
    tts_scores = {}  # tts_model -> list of per-file scores

    total_files = sum(len(glob.glob(os.path.join(f, "*.wav"))) for f in folders)
    processed = 0

    for folder in folders:
        folder_name = os.path.basename(folder)
        tts_model = extract_tts_model(folder_name)
        wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))

        if not wav_files:
            continue

        if tts_model not in tts_scores:
            tts_scores[tts_model] = []

        for wav_path in wav_files:
            processed += 1
            fname = os.path.basename(wav_path)
            print(f"[{processed}/{total_files}] {folder_name}/{fname}")

            result = score_audio(model, wav_path)
            result["file"] = fname
            result["source_folder"] = folder_name
            tts_scores[tts_model].append(result)

    # Save per-TTS-model results
    output_dir = os.path.join("scoring_output", mode, "tts_models")
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for tts_model, scores in tts_scores.items():
        # Save individual scores
        out_file = os.path.join(output_dir, f"{tts_model}.json")
        with open(out_file, "w") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        # Compute averages
        valid = [s for s in scores if "error" not in s]
        if valid:
            summary[tts_model] = {
                "num_files": len(scores),
                "avg_naturalness": round(sum(s["naturalness"] for s in valid) / len(valid), 2),
                "avg_noisiness": round(sum(s["noisiness"] for s in valid) / len(valid), 2),
                "avg_loudness": round(sum(s["loudness"] for s in valid) / len(valid), 2),
            }

        print(f"  {tts_model}: {len(scores)} files -> {out_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Score audio with MiniCPM-o 4.5")
    parser.add_argument("wav_file", nargs="?", help="Path to a .wav file (single mode)")
    parser.add_argument("--mode", choices=["baseline", "finetuned"], default="baseline",
                        help="baseline = original model, finetuned = LoRA-adapted model")
    parser.add_argument("--batch", action="store_true",
                        help="Score all TTS audio in evaluation_output/tts_audio/")
    args = parser.parse_args()

    if not args.batch and not args.wav_file:
        parser.error("Provide a wav_file or use --batch")

    print(f"Loading MiniCPM-o 4.5 ({args.mode})...")
    model = load_model(args.mode)

    if args.batch:
        run_batch(model, args.mode)
    else:
        print(f"Scoring: {args.wav_file}")
        run_single(model, args.wav_file, args.mode)


if __name__ == "__main__":
    main()
