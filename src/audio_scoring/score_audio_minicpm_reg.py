#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Score audio with the MiniCPM-o 4.5 LoRA + REGRESSION-HEAD model.

The regression model produced by finetune_audio_scorer_reg.py does NOT emit a
JSON string; instead a linear head maps the final-token hidden state of the
audio-conditioned multimodal forward pass to three continuous scores
(naturalness, noisiness, loudness). This script therefore:

  1. loads the base MiniCPM-o with the LoRA adapter (PeftModel),
  2. loads the trained regression head (reg_head.pt) saved alongside it,
  3. runs the SAME audio-fused forward + last-token pooling as training
     (get_vllm_embedding + get_omni_embedding via model.base_model.model),
  4. writes the de-normalised 1-10 scores back into each evaluation JSON under
     voice_quality.minicpm_finetuned, matching the schema rank_combinations.py
     already consumes.

Usage:
    # Score all TTS audio referenced by evaluation JSONs with best_full
    python src/audio_scoring/score_audio_minicpm_reg.py --batch \
        --eval-dir output/eval-results/evaluation_output/evaluation \
        --audio-dir output/tts_audio \
        --lora-path data/trained_models/best_full --resume

    # Single file (prints scores)
    python src/audio_scoring/score_audio_minicpm_reg.py path/to/file.wav \
        --lora-path data/trained_models/best_full
"""

import argparse
import glob
import json
import os
import traceback
from copy import deepcopy

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import PeftModel

MODEL_NAME = "openbmb/MiniCPM-o-4_5"
METRIC_NAMES = ["naturalness", "noisiness", "loudness"]
SCORE_MIN, SCORE_MAX = 1.0, 10.0

DEFAULT_EVAL_DIR = "output/eval-results/evaluation_output/evaluation"
DEFAULT_AUDIO_DIR = "output/tts_audio"
DEFAULT_LORA_PATH = "data/trained_models/best_full"

# Same prompt the regression trainer used (user turn only, no JSON instruction).
SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud."
)

# Folder-name mismatches between evaluation/ and tts_audio/ (mirrors the
# generative scorer).
EVAL_TO_AUDIO_FOLDER = {"whisper_large": "whisper_large_v2"}


def _denorm(x: float) -> float:
    v = x * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    return float(np.clip(v, SCORE_MIN, SCORE_MAX))


def resolve_audio_folder(name: str) -> str:
    for old, new in EVAL_TO_AUDIO_FOLDER.items():
        name = name.replace(old, new)
    return name


def resolve_audio_path(entry: dict, eval_folder_name: str, audio_dir: str):
    s3_path = entry.get("tts_audio_path", "")
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
    qid = entry.get("question_id", "")
    if qid:
        candidate = os.path.join(audio_dir, audio_folder, f"{qid}_response.wav")
        if os.path.isfile(candidate):
            return candidate
    return None


def load_model_and_head(lora_path: str):
    """Load base MiniCPM-o + LoRA adapter + trained regression head."""
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    model.prepare_processor(processor=None, tokenizer=None)
    model.processor.tokenizer.padding_side = "right"

    if not hasattr(model, "get_input_embeddings"):
        from types import MethodType
        model.get_input_embeddings = MethodType(
            lambda self: self.llm.get_input_embeddings(), model
        )

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval().cuda()

    hidden_size = model.base_model.model.llm.config.hidden_size
    reg_head = nn.Linear(hidden_size, len(METRIC_NAMES)).cuda().to(torch.bfloat16)
    head_path = os.path.join(lora_path, "reg_head.pt")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(
            f"Regression head not found at {head_path}. This script requires the "
            f"reg_head.pt saved by finetune_audio_scorer_reg.py."
        )
    reg_head.load_state_dict(torch.load(head_path, map_location="cuda"))
    reg_head.eval()
    return model, reg_head


def build_prompt_inputs(model, audio_array):
    """Tokenize prompt + audio into processor inputs (identical to training)."""
    msgs = [{"role": "user", "content": [SCORING_PROMPT, audio_array]}]
    copy_msgs = deepcopy(msgs)
    audios, audio_parts = [], []
    for i, msg in enumerate(copy_msgs):
        cur = []
        for c in msg["content"]:
            if isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(i)
                cur.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur.append(c)
        msg["content"] = "\n".join(cur)
    prompt_text = model.processor.tokenizer.apply_chat_template(
        copy_msgs, tokenize=False, add_generation_prompt=True, use_tts_template=True,
    )
    return model.processor(
        [prompt_text], [[]], [audios], [audio_parts],
        return_tensors="pt", max_length=4096,
    )


@torch.no_grad()
def score_audio(model, reg_head, audio_path: str) -> dict:
    """Return {"naturalness": int, "noisiness": int, "loudness": int}."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = build_prompt_inputs(model, audio)
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
    data.pop("image_sizes", None)
    data["vision_hidden_states"] = [[]]
    input_ids = data["input_ids"]
    attn = data.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids)
    data["position_ids"] = (attn.long().cumsum(-1) - 1).clamp(min=0)
    outputs = model.base_model.model(data, output_hidden_states=True, use_cache=False)
    hs = outputs.hidden_states[-1]
    am = inputs.get("attention_mask")
    last_idx = int(am[0].sum().item()) - 1 if am is not None else hs.shape[1] - 1
    pooled = hs[0, last_idx, :]
    pred = reg_head(pooled.to(reg_head.weight.dtype))
    torch.cuda.empty_cache()
    # Round to integers to match the existing minicpm_finetuned schema.
    return {m: int(round(_denorm(float(pred[i].item()))))
            for i, m in enumerate(METRIC_NAMES)}


def has_scores(entry: dict) -> bool:
    vq = entry.get("voice_quality") or {}
    m = vq.get("minicpm_finetuned") or {}
    return bool(m) and "error" not in m


def run_batch(model, reg_head, eval_dir, audio_dir, resume=False, combo_filter=None):
    combo_dirs = sorted(glob.glob(os.path.join(eval_dir, "*")))
    if combo_filter:
        combo_dirs = [d for d in combo_dirs if os.path.basename(d) == combo_filter]
        if not combo_dirs:
            print(f"No combo folder matching '{combo_filter}' in {eval_dir}")
            return

    all_jsons = []
    for combo_path in combo_dirs:
        if not os.path.isdir(combo_path):
            continue
        combo_name = os.path.basename(combo_path)
        for jf in sorted(glob.glob(os.path.join(combo_path, "*_evaluation.json"))):
            all_jsons.append((jf, combo_name))

    total = len(all_jsons)
    scored = skipped = errors = missing = 0
    print(f"Found {total} evaluation JSONs across {len(combo_dirs)} combo folders")

    for idx, (json_path, combo_name) in enumerate(all_jsons, 1):
        print(f"[{idx}/{total}] {combo_name}/{os.path.basename(json_path)}")
        try:
            with open(json_path) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ERROR reading JSON: {e}")
            errors += 1
            continue

        modified = False
        for ei, entry in enumerate(eval_data.get("evaluations", [])):
            if resume and has_scores(entry):
                skipped += 1
                continue
            audio_path = resolve_audio_path(entry, combo_name, audio_dir)
            if not audio_path:
                missing += 1
                continue
            try:
                result = score_audio(model, reg_head, audio_path)
                # Some entries carry an explicit "voice_quality": null, in which
                # case setdefault() returns that None (it only inserts when the
                # key is absent). Coerce any non-dict value to a fresh dict so we
                # can always write the score back.
                if not isinstance(entry.get("voice_quality"), dict):
                    entry["voice_quality"] = {}
                entry["voice_quality"]["minicpm_finetuned"] = result
                modified = True
                scored += 1
            except Exception as e:
                print(f"  [{ei}] ERROR scoring: {e}")
                traceback.print_exc()
                errors += 1

        if modified:
            with open(json_path, "w") as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Scored: {scored}, Skipped: {skipped}, "
          f"Missing audio: {missing}, Errors: {errors}")


def main():
    p = argparse.ArgumentParser(description="Score audio with MiniCPM-o regression head")
    p.add_argument("wav_file", nargs="?", help="Single .wav to score")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    p.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    p.add_argument("--resume", action="store_true",
                   help="Skip entries already scored (overwrite disabled)")
    p.add_argument("--combo", default=None)
    args = p.parse_args()

    if not args.batch and not args.wav_file:
        p.error("Provide a wav_file or use --batch")

    print(f"Loading MiniCPM-o 4.5 + regression head from {args.lora_path} ...")
    model, reg_head = load_model_and_head(args.lora_path)

    if args.batch:
        run_batch(model, reg_head, args.eval_dir, args.audio_dir,
                  resume=args.resume, combo_filter=args.combo)
    else:
        print(json.dumps(score_audio(model, reg_head, args.wav_file), indent=2))


if __name__ == "__main__":
    main()
