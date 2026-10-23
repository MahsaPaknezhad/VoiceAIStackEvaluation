#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Score audio files using Google Gemma 4 for naturalness, noisiness, loudness.

Gemma counterpart of score_audio_minicpm.py. Walks evaluation JSON files,
resolves the matching local .wav from tts_audio/, scores each with Gemma 4, and
writes results back into the JSON under voice_quality.gemma4[_finetuned].

Two scoring modes (chosen with --mode):
  * generative : zero-shot Gemma emits JSON scores via generate() (parsed like
    the MiniCPM baseline).
  * regression : LoRA adapter + a trained linear regression head
    (reg_head.pt) produced by finetune_audio_scorer_reg_gemma.py. The head maps
    the final-token hidden state to 3 continuous scores. This is the path that
    mirrors the paper's "LoRA + regression-head" model.

Usage:
    # Single file, zero-shot generative
    python score_audio_gemma.py <wav_file>

    # Single file, fine-tuned regression head
    python score_audio_gemma.py <wav_file> --mode regression \
        --lora-path data/trained_models/gemma4_audio_scorer_reg

    # Batch: score all TTS audio referenced by evaluation JSONs
    python score_audio_gemma.py --batch \
        --eval-dir lily-data/evaluation_output/evaluation \
        --audio-dir lily-data/tts_audio

    # Resume (skips already-scored entries), one combo only
    python score_audio_gemma.py --batch --resume --combo deepgram_nova2_cartesia

Model: an audio-capable Gemma 4 checkpoint (E2B / E4B / 12B). Default is E4B to
fit a single mid-size GPU; override with --model or AUDIO_SCORER_MODEL. Confirm
the exact Hugging Face id before running.
"""

import argparse
import glob
import json
import os
import re
import traceback

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_EVAL_DIR = "lily-data/evaluation_output/evaluation"
DEFAULT_AUDIO_DIR = "lily-data/tts_audio"
DEFAULT_MODEL = os.environ.get("AUDIO_SCORER_MODEL", "google/gemma-4-E4B-it")

EVAL_TO_AUDIO_FOLDER = {
    "whisper_large": "whisper_large_v2",
}

METRIC_NAMES = ["naturalness", "noisiness", "loudness"]
AUDIO_SR = 16000
SCORE_MIN, SCORE_MAX = 1.0, 10.0

SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud.\n\n"
    'Respond ONLY with JSON: {"naturalness": <int>, "noisiness": <int>, "loudness": <int>}'
)


def _denorm(x):
    v = x * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    return float(np.clip(v, SCORE_MIN, SCORE_MAX))


def resolve_audio_folder(eval_folder_name: str) -> str:
    for old, new in EVAL_TO_AUDIO_FOLDER.items():
        eval_folder_name = eval_folder_name.replace(old, new)
    return eval_folder_name


def resolve_audio_path(eval_entry: dict, eval_folder_name: str, audio_dir: str):
    """Resolve the local .wav path for an evaluation entry (same logic as MiniCPM)."""
    s3_path = eval_entry.get("tts_audio_path", "")
    audio_folder = resolve_audio_folder(eval_folder_name)

    if s3_path:
        parts = s3_path.rstrip("/").split("/")
        try:
            idx = parts.index("tts_audio")
            s3_combo = parts[idx + 1] if len(parts) > idx + 1 else ""
            local_combo = resolve_audio_folder(s3_combo)
            filename = parts[-1] if parts else ""
            candidate = os.path.join(audio_dir, local_combo, filename)
            if os.path.isfile(candidate):
                return candidate
        except ValueError:
            pass

    qid = eval_entry.get("question_id", "")
    if qid:
        candidate = os.path.join(audio_dir, audio_folder, f"{qid}_response.wav")
        if os.path.isfile(candidate):
            return candidate
    return None


def load_model(model_name: str, mode: str = "generative", lora_path: str | None = None):
    """Load Gemma 4 and, for regression mode, attach LoRA + the regression head.

    Returns (model, processor, reg_head). reg_head is None in generative mode.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    reg_head = None
    if mode == "regression":
        if not lora_path:
            raise ValueError("--lora-path is required in regression mode")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        cfg = model.base_model.model.config
        hidden_size = getattr(cfg, "text_config", cfg).hidden_size
        reg_head = torch.nn.Linear(hidden_size, len(METRIC_NAMES)).to(torch.bfloat16)
        head_path = os.path.join(lora_path, "reg_head.pt")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"regression head not found: {head_path}")
        reg_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        reg_head = reg_head.cuda()
        reg_head.eval()
    model.eval().cuda()
    return model, processor, reg_head


def _build_inputs(processor, audio, add_generation_prompt: bool):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": SCORING_PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )


def _parse_scores(response: str) -> dict:
    """Parse a generative response into {metric: int}, or an error dict."""
    try:
        scores = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{[^}]+\}", response or "")
        scores = None
        if match:
            try:
                scores = json.loads(match.group())
            except json.JSONDecodeError:
                scores = None
        if scores is None:
            return {"raw_response": response, "error": "Failed to parse JSON"}
    if all(k in scores for k in METRIC_NAMES):
        return {k: scores[k] for k in METRIC_NAMES}
    return {"raw_response": response, "error": "Missing expected keys"}


def score_audio(model, processor, reg_head, audio_path: str, mode: str) -> dict:
    """Score a single audio file -> {naturalness, noisiness, loudness} or error."""
    audio, _ = librosa.load(audio_path, sr=AUDIO_SR, mono=True)

    if mode == "regression":
        inputs = _build_inputs(processor, audio, add_generation_prompt=True)
        data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}
        attn = data.get("attention_mask")
        with torch.no_grad():
            out = model(**data, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[-1]
            last_idx = (int(attn[0].sum().item()) - 1) if attn is not None else hs.shape[1] - 1
            pred = reg_head(hs[0, last_idx, :].to(reg_head.weight.dtype))
        torch.cuda.empty_cache()
        return {m: round(_denorm(float(pred[i].item())), 3)
                for i, m in enumerate(METRIC_NAMES)}

    # generative
    inputs = _build_inputs(processor, audio, add_generation_prompt=True)
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
    prompt_len = data["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**data, do_sample=False, max_new_tokens=256)
    torch.cuda.empty_cache()
    response = processor.batch_decode(
        gen[:, prompt_len:], skip_special_tokens=True
    )[0]
    return _parse_scores(response)


def run_single(model, processor, reg_head, wav_file: str, mode: str):
    result = score_audio(model, processor, reg_head, wav_file, mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def score_key(mode: str) -> str:
    return "gemma4_finetuned" if mode == "regression" else "gemma4"


def has_scores(entry: dict, mode: str) -> bool:
    vq = entry.get("voice_quality") or {}
    s = vq.get(score_key(mode)) or {}
    return bool(s) and "error" not in s


def run_batch(model, processor, reg_head, eval_dir: str, audio_dir: str,
              resume: bool = False, combo_filter: str | None = None,
              mode: str = "generative"):
    combo_dirs = sorted(glob.glob(os.path.join(eval_dir, "*")))
    if combo_filter:
        combo_dirs = [d for d in combo_dirs if os.path.basename(d) == combo_filter]
        if not combo_dirs:
            print(f"No combo folder matching '{combo_filter}' found in {eval_dir}")
            return

    all_jsons = []
    for combo_path in combo_dirs:
        if not os.path.isdir(combo_path):
            continue
        combo_name = os.path.basename(combo_path)
        for jf in sorted(glob.glob(os.path.join(combo_path, "*_evaluation.json"))):
            all_jsons.append((jf, combo_name))

    total = len(all_jsons)
    scored = skipped = errors = missing_audio = 0
    print(f"Found {total} evaluation JSONs across {len(combo_dirs)} combo folders")

    for idx, (json_path, combo_name) in enumerate(all_jsons, 1):
        print(f"\n[{idx}/{total}] {combo_name}/{os.path.basename(json_path)}")
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
            if resume and has_scores(entry, mode):
                skipped += 1
                continue

            audio_path = resolve_audio_path(entry, combo_name, audio_dir)
            if not audio_path:
                print(f"  [{entry_idx}] {qid}: audio not found locally, skipping")
                missing_audio += 1
                continue

            try:
                result = score_audio(model, processor, reg_head, audio_path, mode)
                entry.setdefault("voice_quality", {})[score_key(mode)] = result
                modified = True
                scored += 1
                print(f"  [{entry_idx}] {qid}: "
                      f"naturalness={result.get('naturalness', '?')}, "
                      f"noisiness={result.get('noisiness', '?')}, "
                      f"loudness={result.get('loudness', '?')}")
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
        description="Score audio with Gemma 4 (naturalness, noisiness, loudness)"
    )
    parser.add_argument("wav_file", nargs="?", help="Path to a .wav file (single mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Score all TTS audio referenced by evaluation JSONs")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                        help=f"Evaluation JSON directory (default: {DEFAULT_EVAL_DIR})")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,
                        help=f"Local TTS audio directory (default: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemma 4 checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--mode", choices=["generative", "regression"], default="generative",
                        help="generative = zero-shot JSON; regression = LoRA + reg head")
    parser.add_argument("--lora-path", default="data/trained_models/gemma4_audio_scorer_reg",
                        help="Path to LoRA adapter + reg_head.pt (regression mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries that already have gemma4 scores")
    parser.add_argument("--combo", default=None,
                        help="Process only this combo folder")
    args = parser.parse_args()

    if not args.batch and not args.wav_file:
        parser.error("Provide a wav_file or use --batch")

    print(f"Loading Gemma 4 ({args.model}) in {args.mode} mode...")
    model, processor, reg_head = load_model(args.model, args.mode, args.lora_path)

    if args.batch:
        run_batch(model, processor, reg_head, args.eval_dir, args.audio_dir,
                  resume=args.resume, combo_filter=args.combo, mode=args.mode)
    else:
        print(f"Scoring: {args.wav_file}")
        run_single(model, processor, reg_head, args.wav_file, args.mode)


if __name__ == "__main__":
    main()
