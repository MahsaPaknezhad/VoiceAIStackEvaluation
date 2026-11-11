#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Score ONLY the 100 human-study radar clips with the SOMOS MOS checkpoint.

Non-destructive: writes the predicted 1-5 MOS into
    voice_quality.minicpm_finetuned.somos_mos  (+ per-clip stats)
and NEVER touches the existing voice_quality.minicpm_finetuned.naturalness
produced by the prior naturalness model.

Reuses the exact crop-and-aggregate inference from
score_audio_minicpm_naturalness.py (same 3.0s/1.5s windows, same audio-fused
forward + last-token pooling + 1-output reg head), so the only difference vs
that script is (a) the write field and (b) scoping to the 100 target clips.

Usage:
    .venv/bin/python src/audio_scoring/score_somos_mos_radar100.py \
        --lora-path data/trained_models/minicpm_somos_mos_full \
        --eval-dir output/eval-results/evaluation_output/evaluation \
        --audio-dir output/tts_audio \
        --key analysis/reviewer_response/results/provider_mos/hidden_key.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

# Reuse the proven loader + scorer from the naturalness scorer.
import score_audio_minicpm_naturalness as nat_scorer
from score_audio_minicpm_naturalness import (
    load_model_and_head, score_audio, resolve_audio_path,
    CROP_WINDOW_S, CROP_HOP_S, AGG_STAT,
)

# CRITICAL: the naturalness scorer de-normalizes model outputs with its own
# module-level SCORE_MIN/MAX = 1.0/10.0 (naturalness scale). Our SOMOS head was
# trained on a 1-5 MOS scale, so we must override those module globals to 1-5
# before scoring; otherwise [0,1] outputs get mapped onto 1-10 and land in the
# impossible 6-7 region. _denorm() reads these globals at call time.
nat_scorer.SCORE_MIN, nat_scorer.SCORE_MAX = 1.0, 5.0

MOS_FIELD = "somos_mos"


def load_target_keys(key_path):
    """Return set of (stt, tts, question_id) identifying the 100 study clips."""
    key = json.load(open(key_path))
    want = set()
    for _clip_id, meta in key.items():
        want.add((meta["stt"], meta["provider"], meta["question_id"]))
    return want


def main():
    p = argparse.ArgumentParser(description="Score the 100 radar clips with SOMOS MOS head")
    p.add_argument("--lora-path", default="data/trained_models/minicpm_somos_mos_full")
    p.add_argument("--eval-dir", default="output/eval-results/evaluation_output/evaluation")
    p.add_argument("--audio-dir", default="output/tts_audio")
    p.add_argument("--key", default="analysis/reviewer_response/results/provider_mos/hidden_key.json")
    p.add_argument("--window", type=float, default=CROP_WINDOW_S)
    p.add_argument("--hop", type=float, default=CROP_HOP_S)
    p.add_argument("--agg-stat", default=AGG_STAT, choices=["mean", "median", "min", "max"])
    p.add_argument("--resume", action="store_true", help="skip clips already having somos_mos")
    args = p.parse_args()

    want = load_target_keys(args.key)
    print(f"Target study clips: {len(want)} (stt,tts,question_id) keys")

    print(f"Loading MiniCPM-o 4.5 + SOMOS MOS head from {args.lora_path} ...")
    model, reg_head = load_model_and_head(args.lora_path)

    # KNOWN_TTS suffix resolution mirrors the radar script's combo parsing.
    from plot_cross_method_radar import KNOWN_TTS

    scored = skipped = missing = errors = 0
    for combo_path in sorted(glob.glob(os.path.join(args.eval_dir, "*"))):
        if not os.path.isdir(combo_path):
            continue
        combo = os.path.basename(combo_path)
        tts = next((t for t in KNOWN_TTS if combo.endswith(t)), None)
        if tts is None:
            continue
        stt = combo[: -(len(tts) + 1)]

        for jf in sorted(glob.glob(os.path.join(combo_path, "*_evaluation.json"))):
            try:
                data = json.load(open(jf))
            except (json.JSONDecodeError, OSError):
                errors += 1
                continue
            modified = False
            for entry in data.get("evaluations", []):
                qid = entry.get("question_id", "")
                if (stt, tts, qid) not in want:
                    continue  # not one of the 100 study clips
                vq = entry.get("voice_quality")
                if not isinstance(vq, dict):
                    vq = {}; entry["voice_quality"] = vq
                mf = vq.get("minicpm_finetuned")
                if not isinstance(mf, dict):
                    mf = {}; vq["minicpm_finetuned"] = mf
                if args.resume and MOS_FIELD in mf:
                    skipped += 1
                    continue
                audio_path = resolve_audio_path(entry, combo, args.audio_dir)
                if not audio_path:
                    missing += 1
                    continue
                try:
                    res = score_audio(model, reg_head, audio_path,
                                      window_s=args.window, hop_s=args.hop,
                                      agg_stat=args.agg_stat)
                    # res["naturalness"] is the 1-5 MOS from our head (the shared
                    # scorer names the primary output "naturalness"); store it
                    # under somos_mos + keep the per-clip stats namespaced.
                    mf[MOS_FIELD] = res.get("naturalness")
                    mf[f"{MOS_FIELD}_stats"] = {
                        k: v for k, v in res.items() if k != "naturalness"
                    }
                    modified = True
                    scored += 1
                except Exception as e:
                    print(f"  ERROR scoring {qid}: {e}")
                    errors += 1
            if modified:
                with open(jf, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Scored: {scored}, Skipped: {skipped}, "
          f"Missing audio: {missing}, Errors: {errors}")
    print(f"Wrote predictions to voice_quality.minicpm_finetuned.{MOS_FIELD} "
          f"(existing 'naturalness' field left untouched).")


if __name__ == "__main__":
    main()
