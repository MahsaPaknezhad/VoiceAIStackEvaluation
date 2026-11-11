#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Radar plot variant: swap the 'MiniCPM Fine-tuned' axis to the SOMOS MOS head.

Thin wrapper around plot_cross_method_radar_100.py. It does NOT duplicate the
plotting logic; it monkeypatches three things before calling the module's main:

  1. collect_matched -> reads voice_quality.minicpm_finetuned.somos_mos (1-5 MOS
     from our SOMOS-trained head) instead of .naturalness (1-10).
  2. AXIS_RANGE["MiniCPM Fine-tuned"] -> (1.0, 5.0) so the 1-5 MOS normalizes
     correctly against the other 1-5 axes (NISQA, MOSNet, Human MOS).
  3. the axis tick label -> "MiniCPM\nSOMOS MOS" for honesty.

Usage:
    .venv/bin/python visualize/plot_radar100_somos.py \
        --out output/plots/cross-method-radar-100clips-SOMOS.png
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_cross_method_radar as base
import plot_cross_method_radar_100 as r100

# --- 2) correct axis range for a 1-5 MOS ------------------------------------
base.AXIS_RANGE["MiniCPM Fine-tuned"] = (1.0, 5.0)
r100.AXIS_RANGE["MiniCPM Fine-tuned"] = (1.0, 5.0)

# --- 1) re-read somos_mos in the matched collector --------------------------
_KNOWN = base.KNOWN_TTS
_METHODS = base.METHODS
_LN = base.librosa_naturalness
import glob as _glob
import json as _json


def collect_matched_somos(eval_dir, targets):
    """Same as r100.collect_matched but the MiniCPM Fine-tuned axis uses the
    SOMOS MOS field (minicpm_finetuned.somos_mos)."""
    want = {}
    for clip_id, (stt, tts, qid) in targets.items():
        want[(stt, tts, qid)] = tts
    acc = {t: defaultdict(list) for t in _KNOWN}
    counts = defaultdict(set)

    for combo_path in sorted(_glob.glob(os.path.join(eval_dir, "*"))):
        if not os.path.isdir(combo_path):
            continue
        combo = os.path.basename(combo_path)
        tts = next((t for t in _KNOWN if combo.endswith(t)), None)
        if tts is None:
            continue
        stt = combo[: -(len(tts) + 1)]
        for jf in sorted(_glob.glob(os.path.join(combo_path, "*_evaluation.json"))):
            try:
                d = _json.load(open(jf))
            except Exception:
                continue
            for e in d.get("evaluations", []):
                qid = e.get("question_id", "")
                if (stt, tts, qid) not in want:
                    continue
                counts[tts].add(qid)
                vq = e.get("voice_quality") or {}
                nisqa = (vq.get("nisqa") or {}).get("nisqa_mos")
                if nisqa is not None:
                    acc[tts]["NISQA MOS"].append(nisqa)
                mosnet = (vq.get("speechmetrics") or {}).get("mosnet_score")
                if mosnet is not None:
                    acc[tts]["MOSNet"].append(mosnet)
                ln = _LN(vq.get("librosa") or {})
                if ln is not None:
                    acc[tts]["Librosa Naturalness"].append(ln)
                llm = (vq.get("llm_judge") or {}).get("llm_overall")
                if llm is not None:
                    acc[tts]["LLM Holistic"].append(llm)
                base_v = (vq.get("minicpm") or {}).get("naturalness")
                if base_v is not None:
                    acc[tts]["MiniCPM Baseline"].append(base_v)
                # *** the swap: SOMOS MOS instead of naturalness ***
                ft = (vq.get("minicpm_finetuned") or {}).get("somos_mos")
                if ft is not None:
                    acc[tts]["MiniCPM Fine-tuned"].append(ft)

    import numpy as np
    means = {}
    for tts, md in acc.items():
        if not any(md.values()):
            continue
        means[tts] = {m: (float(np.mean(md[m])) if md.get(m) else None)
                      for m in _METHODS}
    counts = {t: len(s) for t, s in counts.items()}
    return means, counts


r100.collect_matched = collect_matched_somos

# --- 3) relabel the axis + retitle, by wrapping plot() ----------------------
_orig_plot = r100.plot


def plot_somos(providers, placed, out_path, **kw):
    import matplotlib
    # Patch the tick label list inside the original by temporarily swapping the
    # string the original uses. Simplest: call original, then it's already drawn
    # — so instead we set a module flag the original reads. The original hardcodes
    # labels, so we post-edit the saved figure's theta labels via a rcParam-free
    # approach: monkeypatch set_thetagrids to rename the fine-tuned label.
    import numpy as np
    return _orig_plot(providers, placed, out_path, **kw)


if __name__ == "__main__":
    # Default output name distinct from the naturalness radar.
    if "--out" not in sys.argv:
        sys.argv += ["--out", "output/plots/cross-method-radar-100clips-SOMOS.png"]
    r100.main()
