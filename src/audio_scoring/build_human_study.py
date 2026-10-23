#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Build a blinded human-rating study from the TTS outputs of ONE STT model.

Takes N prompts, collects the audio each of the five TTS providers produced for
each prompt, loudness-normalises it, assigns random clip IDs, and writes a
manifest (the private key) plus the audio the rating UI will serve.

Usage:
    python src/audio_scoring/build_human_study.py --stt aws_transcribe --n-prompts 60

Output (data/human_study/, gitignored):
    audio/<clip_id>.wav   normalised audio, named by random ID only
    manifest.json         PRIVATE key: clip_id -> prompt, provider, text
                          (never sent to raters)

Notes:
  * Each STT x TTS run calls the LLM separately (temperature 0.7), so the text
    a provider speaks can differ from another provider's for the same prompt.
    The script prefers prompts whose response text is IDENTICAL across all five
    providers and reports how many it found. Different text is a confound.
  * Audio sample rates differ by provider (NVIDIA Magpie is 16 kHz, the rest
    24 kHz). This is left as-is: it is a property of the provider's output.
"""

import argparse
import array
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import wave

EVAL = "output/eval-results"
TTS_IDS = ["aws_polly", "cartesia", "deepgram_aura", "groq", "nvidia_magpie"]


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_runs(stt):
    """tts_id -> {question_id: (llm_response, wav_path)} for successful runs."""
    runs = {}
    for tts in TTS_IDS:
        run = f"{stt}_{tts}"
        found = {}
        pattern = f"{EVAL}/evaluation_output/results/{run}/*_results.json"
        for f in glob.glob(pattern):
            with open(f) as fh:
                data = json.load(fh)
            for r in data.get("results", []):
                qid = r.get("question_id", "")
                if r.get("status") != "success" or not r.get("llm_response"):
                    continue
                if not qid.startswith("speaking_reasoning"):
                    continue
                wav = f"{EVAL}/tts_audio/{run}/{qid}_response.wav"
                if os.path.exists(wav):
                    found[qid] = (r["llm_response"], wav)
        runs[tts] = found
    return runs


def normalize_wav(src, dst, target_rms=0.08):
    """RMS-normalise 16-bit mono PCM so raters are not swayed by volume.
    Returns False (and copies unchanged) for any other format."""
    with wave.open(src) as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        shutil.copy(src, dst)
        return False
    a = array.array("h")
    a.frombytes(frames)
    if sys.byteorder == "big":
        a.byteswap()
    n = len(a)
    if n == 0:
        shutil.copy(src, dst)
        return False
    rms = math.sqrt(sum(x * x for x in a) / n) / 32768.0
    peak = max(abs(x) for x in a) or 1
    if rms == 0:
        shutil.copy(src, dst)
        return False
    gain = min(target_rms / rms, 0.97 * 32768.0 / peak)
    out = array.array("h", (max(-32768, min(32767, int(round(x * gain)))) for x in a))
    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(dst, "wb") as w:
        w.setparams(params)
        w.writeframes(out.tobytes())
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stt", default="aws_transcribe", help="STT run to take TTS outputs from")
    ap.add_argument("--n-prompts", type=int, default=60)
    ap.add_argument("--repeat-frac", type=float, default=0.10, help="fraction of clips repeated for test-retest")
    ap.add_argument("--same-text-only", action="store_true", help="fail if not enough identical-text prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/human_study")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs = load_runs(args.stt)
    for tts, d in runs.items():
        print(f"  {args.stt}_{tts}: {len(d)} usable clips")

    common = set.intersection(*(set(d) for d in runs.values())) if runs else set()
    same = sorted(q for q in common if len({norm_text(runs[t][q][0]) for t in TTS_IDS}) == 1)
    diff = sorted(common - set(same))
    print(f"Prompts available for all 5 providers: {len(common)} "
          f"({len(same)} with identical response text, {len(diff)} with differing text)")
    if not common:
        sys.exit("No prompts have audio for all five providers. Check --stt and the output/ paths.")

    rng.shuffle(same)
    rng.shuffle(diff)
    if args.same_text_only and len(same) < args.n_prompts:
        sys.exit(f"Only {len(same)} identical-text prompts; need {args.n_prompts}.")
    chosen = (same + diff)[: args.n_prompts]
    n_same = sum(q in set(same) for q in chosen)
    print(f"Selected {len(chosen)} prompts ({n_same} identical-text, {len(chosen) - n_same} differing-text)")
    if n_same < len(chosen):
        print("  WARNING: some prompts have different text per provider (content confound).")

    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    used, clips = set(), []

    def new_id():
        while True:
            cid = f"c{rng.getrandbits(32):08x}"
            if cid not in used:
                used.add(cid)
                return cid

    for q in chosen:
        for tts in TTS_IDS:
            text, wav = runs[tts][q]
            cid = new_id()
            normalize_wav(wav, os.path.join(audio_dir, f"{cid}.wav"))
            clips.append({"clip_id": cid, "prompt_id": q, "tts": tts, "stt": args.stt,
                          "text": text, "same_text_across_providers": q in set(same),
                          "repeat_of": None})

    originals = list(clips)
    n_rep = math.ceil(args.repeat_frac * len(originals))
    for src in rng.sample(originals, n_rep):
        cid = new_id()
        shutil.copy(os.path.join(audio_dir, f"{src['clip_id']}.wav"), os.path.join(audio_dir, f"{cid}.wav"))
        clips.append({**src, "clip_id": cid, "repeat_of": src["clip_id"]})

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"seed": args.seed, "stt": args.stt, "clips": clips}, f, indent=1)

    print(f"\nWrote {len(clips)} clips ({len(originals)} unique + {n_rep} repeats) to {args.out}/")
    print(f"Each rater will rate {len(clips)} clips, roughly {len(clips) * 25 // 60} min.")
    print("manifest.json is the private key: do not share it with raters.")


if __name__ == "__main__":
    main()
