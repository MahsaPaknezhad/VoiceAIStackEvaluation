"""
STT + TTS Combination Ranking Script
=====================================

FORMULA
-------

For each STT+TTS combination c, the composite score is:

    CompositeScore(c) = Σ_i [ w_i * Norm_i(c) ] / Σ_i [ w_i ]

where:

    Norm_i(c) = (clamp(x_i(c)) - lo_i) / (hi_i - lo_i)       if higher is better
    Norm_i(c) = 1 - (clamp(x_i(c)) - lo_i) / (hi_i - lo_i)   if lower is better

    x_i(c)  = mean value of metric i across all samples for combination c
    lo_i    = max(min_i, Q1_i - 1.5 * IQR_i)   (robust lower bound)
    hi_i    = min(max_i, Q1_i + 1.5 * IQR_i)   (robust upper bound)
    clamp() = clips x to [lo_i, hi_i] to limit outlier influence
    w_i     = weight assigned to metric i

    If a metric is missing for combination c, the median normalized score
    across all other combinations is imputed (avoids inflating scores for
    combinations with incomplete data).

METRICS (7 dimensions)
-----------------------------------------------------

  i  Metric                  Source                              Direction  Weight
  -- ----------------------  ----------------------------------  ---------  ------
  1  WER                     jiwer word error rate               lower      30%
  2  LLM Judge Overall       LLM-as-judge overall (0-10)        higher     10%
  3  Total Latency           STT + LLM + TTS end-to-end ms      lower      30%
  4  LLM Voice Overall       LLM holistic voice judge (1-5)     higher     10%
  5  MiniCPM Naturalness     fine-tuned vision-language (0-10)   higher      7%
  6  MiniCPM Noisiness       fine-tuned vision-language (0-10)   lower       7%
  7  MiniCPM Loudness        fine-tuned vision-language (0-10)   higher      6%

Usage:
    python rank_combinations.py [--eval-dir PATH]
    python rank_combinations.py --weights 30,10,30,10,7,7,6
"""

import json
import os
import glob
import statistics
import argparse

KNOWN_TTS = ["aws_polly", "deepgram_aura", "cartesia", "groq", "nvidia_magpie"]

METRIC_NAMES = ["wer_mean", "judge_mean", "total_lat_mean", "llm_voice_mean", "minicpm_nat_mean", "minicpm_noise_mean", "minicpm_loud_mean"]
METRIC_LABELS = ["WER", "Judge", "Latency", "VoiceLLM", "mCPM_Nat", "mCPM_Noi", "mCPM_Loud"]
METRIC_DIRS = ["lower", "higher", "lower", "higher", "higher", "lower", "higher"]


def parse_args():
    p = argparse.ArgumentParser(description="Rank STT+TTS combinations")
    p.add_argument("--eval-dir",
                   default=os.path.join(os.path.dirname(__file__), "output/eval-results/evaluation_output/evaluation"),
                   help="Path to evaluation directory")
    p.add_argument("--weights", default="30,10,30,10,7,7,6",
                   help="Comma-separated weights for: WER,Judge,Latency,VoiceLLM,mCPM_Nat,mCPM_Noi,mCPM_Loud (sum to 100)")
    return p.parse_args()


def split_stt_tts(combo_dir):
    for t in KNOWN_TTS:
        if combo_dir.endswith("_" + t):
            return combo_dir[:-(len(t) + 1)], t
    return combo_dir, "unknown"


def collect_metrics(eval_dir):
    combos = {}
    for combo_dir in sorted(os.listdir(eval_dir)):
        combo_path = os.path.join(eval_dir, combo_dir)
        if not os.path.isdir(combo_path):
            continue

        wers, judges, total_lats, stt_lats, tts_lats = [], [], [], [], []
        llm_voice_list, minicpm_nat_list, minicpm_noise_list, minicpm_loud_list = [], [], [], []
        judge_sub = {"correctness": [], "relevance": [], "completeness": [], "clarity": []}
        llm_voice_sub = {"fluency": [], "naturalness": []}

        for f in glob.glob(os.path.join(combo_path, "*_evaluation.json")):
            with open(f) as fh:
                data = json.load(fh)
            for ev in data.get("evaluations", []):
                if ev.get("wer") is not None:
                    wers.append(ev["wer"])
                if ev.get("stt_latency_ms") is not None:
                    stt_lats.append(ev["stt_latency_ms"])
                if ev.get("tts_latency_ms") is not None:
                    tts_lats.append(ev["tts_latency_ms"])
                if ev.get("total_latency_ms") is not None:
                    total_lats.append(ev["total_latency_ms"])

                js = ev.get("judge_scores") or {}
                if js.get("overall") is not None:
                    judges.append(js["overall"])
                for k in judge_sub:
                    if js.get(k) is not None:
                        judge_sub[k].append(js[k])

                vq = ev.get("voice_quality") or {}
                lj = vq.get("llm_judge") or {}
                if lj.get("llm_overall") is not None:
                    llm_voice_list.append(lj["llm_overall"])
                if lj.get("llm_fluency_score") is not None:
                    llm_voice_sub["fluency"].append(lj["llm_fluency_score"])
                if lj.get("llm_naturalness_score") is not None:
                    llm_voice_sub["naturalness"].append(lj["llm_naturalness_score"])

                mcpm = vq.get("minicpm_finetuned") or vq.get("minicpm") or {}
                if mcpm.get("naturalness") is not None:
                    minicpm_nat_list.append(mcpm["naturalness"])
                if mcpm.get("noisiness") is not None:
                    minicpm_noise_list.append(mcpm["noisiness"])
                if mcpm.get("loudness") is not None:
                    minicpm_loud_list.append(mcpm["loudness"])

        if not wers:
            continue

        stt, tts = split_stt_tts(combo_dir)
        m = lambda lst: statistics.mean(lst) if lst else None

        combos[combo_dir] = {
            "stt": stt, "tts": tts, "n": len(wers),
            "wer_mean": statistics.mean(wers),
            "judge_mean": m(judges),
            "stt_lat_mean": m(stt_lats),
            "tts_lat_mean": m(tts_lats),
            "total_lat_mean": m(total_lats),
            "llm_voice_mean": m(llm_voice_list),
            "minicpm_nat_mean": m(minicpm_nat_list),
            "minicpm_noise_mean": m(minicpm_noise_list),
            "minicpm_loud_mean": m(minicpm_loud_list),
            "judge_correctness": m(judge_sub["correctness"]),
            "judge_relevance": m(judge_sub["relevance"]),
            "judge_completeness": m(judge_sub["completeness"]),
            "judge_clarity": m(judge_sub["clarity"]),
            "llm_fluency": m(llm_voice_sub["fluency"]),
            "llm_naturalness": m(llm_voice_sub["naturalness"]),
        }
    return combos


def compute_composite(combos, weights):
    stats = {}
    for m in weights:
        vals = sorted(c[m] for c in combos.values() if c[m] is not None)
        if not vals:
            continue
        med = statistics.median(vals)
        q1 = statistics.median(vals[:len(vals) // 2]) if len(vals) >= 4 else (vals[0] if vals else med)
        q3 = statistics.median(vals[(len(vals) + 1) // 2:]) if len(vals) >= 4 else (vals[-1] if vals else med)
        iqr = q3 - q1
        # Use IQR-based robust range; fall back to min-max if IQR is zero
        lo, hi = min(vals), max(vals)
        if iqr > 0:
            lo = max(lo, q1 - 1.5 * iqr)
            hi = min(hi, q3 + 1.5 * iqr)
        stats[m] = (lo, hi)

    # Pre-compute median normalized score per metric for missing-value imputation
    metric_norms = {}
    for m, cfg in weights.items():
        if m not in stats:
            continue
        lo, hi = stats[m]
        norms = []
        for c in combos.values():
            if c[m] is None:
                continue
            clamped = max(lo, min(c[m], hi))
            if hi == lo:
                norms.append(0.5)
            elif cfg["direction"] == "lower":
                norms.append(1.0 - (clamped - lo) / (hi - lo))
            else:
                norms.append((clamped - lo) / (hi - lo))
        metric_norms[m] = statistics.median(norms) if norms else 0.5

    total_w = sum(cfg["weight"] for cfg in weights.values())
    for c in combos.values():
        score = 0.0
        for m, cfg in weights.items():
            if m not in stats:
                continue
            lo, hi = stats[m]
            if c[m] is None:
                norm = metric_norms[m]
            else:
                clamped = max(lo, min(c[m], hi))
                if hi == lo:
                    norm = 0.5
                elif cfg["direction"] == "lower":
                    norm = 1.0 - (clamped - lo) / (hi - lo)
                else:
                    norm = (clamped - lo) / (hi - lo)
            score += norm * cfg["weight"]
        c["composite"] = score / total_w if total_w > 0 else 0.0


def fmt(val, width, decimals=0):
    if val is None:
        return "N/A".rjust(width)
    return f"{val:>{width}.{decimals}f}"


def print_overall(combos):
    ranked = sorted(combos.items(), key=lambda x: x[1]["composite"], reverse=True)
    hdr = (f"{'Rk':<4} {'STT':<18} {'TTS':<15} {'Score':>5} "
           f"{'WER%':>6} {'Judge':>5} {'TotLat':>7} {'VcLLM':>5} {'mcNat':>5} {'mcNoi':>5} {'mcLod':>5} "
           f"{'STTms':>6} {'TTSms':>6} {'N':>4}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for i, (_, c) in enumerate(ranked, 1):
        print(f"{i:<4} {c['stt']:<18} {c['tts']:<15} {c['composite']:>5.3f} "
              f"{c['wer_mean']:>6.1f} {fmt(c['judge_mean'],5,2)} {fmt(c['total_lat_mean'],7)} "
              f"{fmt(c['llm_voice_mean'],5,2)} {fmt(c['minicpm_nat_mean'],5,1)} "
              f"{fmt(c['minicpm_noise_mean'],5,2)} {fmt(c['minicpm_loud_mean'],5,2)} "
              f"{fmt(c['stt_lat_mean'],6)} {fmt(c['tts_lat_mean'],6)} {c['n']:>4}")


def print_detail(combos):
    ranked = sorted(combos.items(), key=lambda x: x[1]["composite"], reverse=True)[:10]
    print(f"\n{'Rk':<4} {'STT+TTS':<35} {'Corr':>5} {'Relv':>5} {'Comp':>5} {'Clar':>5} "
          f"{'Flncy':>5} {'Natrl':>5}")
    print("-" * 72)
    for i, (_, c) in enumerate(ranked, 1):
        label = f"{c['stt']}+{c['tts']}"
        print(f"{i:<4} {label:<35} "
              f"{fmt(c['judge_correctness'],5,2)} {fmt(c['judge_relevance'],5,2)} "
              f"{fmt(c['judge_completeness'],5,2)} {fmt(c['judge_clarity'],5,2)} "
              f"{fmt(c['llm_fluency'],5,2)} {fmt(c['llm_naturalness'],5,2)}")


def print_stt_ranking(combos):
    agg = {}
    for c in combos.values():
        s = c["stt"]
        if s not in agg:
            agg[s] = {"wer": [], "lat": [], "judge": []}
        agg[s]["wer"].append(c["wer_mean"])
        if c["stt_lat_mean"] is not None:
            agg[s]["lat"].append(c["stt_lat_mean"])
        if c["judge_mean"] is not None:
            agg[s]["judge"].append(c["judge_mean"])

    print(f"\n{'STT':<20} {'Avg WER%':>8} {'Avg STT ms':>10} {'Avg Judge':>10}")
    print("-" * 52)
    for s, v in sorted(agg.items(), key=lambda x: statistics.mean(x[1]["wer"])):
        lat = fmt(statistics.mean(v["lat"]) if v["lat"] else None, 10)
        jdg = fmt(statistics.mean(v["judge"]) if v["judge"] else None, 10, 2)
        print(f"{s:<20} {statistics.mean(v['wer']):>8.2f} {lat} {jdg}")


def print_tts_ranking(combos):
    agg = {}
    for c in combos.values():
        t = c["tts"]
        if t not in agg:
            agg[t] = {"lat": [], "voice": [], "minicpm_nat": [], "minicpm_noi": [], "minicpm_loud": []}
        if c["tts_lat_mean"] is not None:
            agg[t]["lat"].append(c["tts_lat_mean"])
        if c["llm_voice_mean"] is not None:
            agg[t]["voice"].append(c["llm_voice_mean"])
        if c["minicpm_nat_mean"] is not None:
            agg[t]["minicpm_nat"].append(c["minicpm_nat_mean"])
        if c["minicpm_noise_mean"] is not None:
            agg[t]["minicpm_noi"].append(c["minicpm_noise_mean"])
        if c["minicpm_loud_mean"] is not None:
            agg[t]["minicpm_loud"].append(c["minicpm_loud_mean"])

    print(f"\n{'TTS':<16} {'TTSms':>7} {'VoiceLLM':>8} {'mcNat':>6} {'mcNoi':>6} {'mcLod':>6}")
    print("-" * 54)
    for t, v in sorted(agg.items(),
                       key=lambda x: statistics.mean(x[1]["voice"]) if x[1]["voice"] else 0,
                       reverse=True):
        m = lambda lst: statistics.mean(lst) if lst else None
        print(f"{t:<16} {fmt(m(v['lat']),7)} {fmt(m(v['voice']),8,2)} "
              f"{fmt(m(v['minicpm_nat']),6,1)} {fmt(m(v['minicpm_noi']),6,2)} "
              f"{fmt(m(v['minicpm_loud']),6,2)}")


def main():
    args = parse_args()
    w = [float(x) / 100.0 for x in args.weights.split(",")]
    assert len(w) == 7, f"Need 7 weights, got {len(w)}"
    assert abs(sum(w) - 1.0) < 0.02, f"Weights must sum to ~100, got {sum(w)*100:.1f}"

    weights = {}
    for i, name in enumerate(METRIC_NAMES):
        weights[name] = {"weight": w[i], "direction": METRIC_DIRS[i]}

    combos = collect_metrics(args.eval_dir)
    compute_composite(combos, weights)

    wl = args.weights.split(",")
    print("Metric weights: " + ", ".join(f"{METRIC_LABELS[i]}={wl[i]}%" for i in range(len(METRIC_LABELS))))
    print("Method: Min-max normalization per metric, weighted composite. Higher score = better.\n")

    print("=== Overall STT + TTS Rankings ===")
    print_overall(combos)


if __name__ == "__main__":
    main()
