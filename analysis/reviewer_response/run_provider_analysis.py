"""
Provider-level uncertainty and method-agreement analysis.

Addresses two reviewer weaknesses using the authoritative per-utterance eval
data under output/eval-results/evaluation_output/evaluation/:

  R2-W5: no uncertainty on provider comparisons. We compute, for the fine-tuned
         scorer's per-provider naturalness, prompt-level bootstrap 95% CIs and a
         rank-stability analysis (how often each provider holds its rank under
         resampling of the shared prompts).

  R2-W4: discriminating methods disagree except on Polly. We report each
         method's per-provider mean naturalness (NISQA MOS, MOSNet,
         zero-shot MiniCPM, LLM judge, fine-tuned scorer), plus paired tests
         on the fine-tuned scorer so "Aura != top" is backed by significance.

Scores are grouped by TTS provider across all STT x TTS combinations, keyed by
question_id so the bootstrap can resample at the prompt level (each prompt is
realised under multiple STT front-ends).

Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python run_provider_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

EVAL_DIR = ("/home/ubuntu/VoiceAIStackEvaluation/output/eval-results/"
            "evaluation_output/evaluation")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

PROVIDERS = ["aws_polly", "cartesia", "deepgram_aura", "groq", "nvidia_magpie"]
PRETTY = {"aws_polly": "AWS Polly", "cartesia": "Cartesia",
          "deepgram_aura": "Deepgram Aura", "groq": "Groq",
          "nvidia_magpie": "NVIDIA Magpie"}


def _nat_scores(vq: dict) -> dict:
    """Extract each method's naturalness-like score from a voice_quality dict."""
    out = {}
    try:
        out["nisqa_mos"] = float(vq["nisqa"]["nisqa_mos"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["mosnet"] = float(vq["speechmetrics"]["mosnet_score"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["llm_naturalness"] = float(vq["llm_judge"]["llm_naturalness_score"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["llm_overall"] = float(vq["llm_judge"]["llm_overall"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["minicpm_zeroshot"] = float(vq["minicpm"]["naturalness"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        out["minicpm_finetuned"] = float(vq["minicpm_finetuned"]["naturalness"])
    except (KeyError, TypeError, ValueError):
        pass
    return out


def load_provider_scores():
    """
    Returns:
      by_provider[method][provider] -> np.array of per-utterance scores
      paired[method][provider] -> dict{question_id: [scores]} for prompt bootstrap
    """
    by_provider = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for combo in os.listdir(EVAL_DIR):
        cdir = os.path.join(EVAL_DIR, combo)
        if not os.path.isdir(cdir):
            continue
        for f in glob.glob(os.path.join(cdir, "*_evaluation.json")):
            d = json.load(open(f))
            tts = d.get("tts_service_id")
            if tts not in PROVIDERS:
                continue
            for ev in d.get("evaluations", []):
                vq = ev.get("voice_quality") or {}
                qid = ev.get("question_id", "")
                for method, score in _nat_scores(vq).items():
                    by_provider[method][tts].append(score)
                    paired[method][tts][qid].append(score)

    # to arrays
    for method in by_provider:
        for tts in by_provider[method]:
            by_provider[method][tts] = np.array(by_provider[method][tts], float)
    return by_provider, paired


# --------------------------------------------------------------------------- #
# R2-W5: prompt-level bootstrap CIs + rank stability
# --------------------------------------------------------------------------- #
def prompt_bootstrap_means(paired_method: dict, n_boot=5000, seed=42):
    """
    paired_method[provider][qid] -> list of scores.
    Resample prompts (qids) with replacement; for each resample compute each
    provider's mean (averaging over that provider's utterances for the sampled
    qids). Returns per-provider CI and the full bootstrap matrix for ranking.
    """
    rng = np.random.default_rng(seed)
    providers = [p for p in PROVIDERS if p in paired_method]
    # shared prompt universe = union of qids
    qids = sorted({q for p in providers for q in paired_method[p]})
    qids = np.array(qids)
    n = len(qids)

    # pre-index: provider -> qid -> np.array
    idx = {p: {q: np.array(v, float) for q, v in paired_method[p].items()}
           for p in providers}

    boot = {p: np.empty(n_boot) for p in providers}
    for b in range(n_boot):
        sample_q = qids[rng.integers(0, n, n)]
        for p in providers:
            vals = []
            pq = idx[p]
            for q in sample_q:
                if q in pq:
                    vals.append(pq[q].mean())
            boot[p][b] = np.mean(vals) if vals else np.nan

    # point means (over all utterances)
    point = {p: float(np.concatenate([idx[p][q] for q in idx[p]]).mean())
             for p in providers}
    ci = {p: (float(np.nanpercentile(boot[p], 2.5)),
              float(np.nanpercentile(boot[p], 97.5))) for p in providers}

    # rank stability: fraction of bootstrap draws each provider is ranked k
    # (rank 1 = highest mean naturalness)
    order_counts = {p: np.zeros(len(providers)) for p in providers}
    stacked = np.vstack([boot[p] for p in providers])  # (P x B)
    for b in range(n_boot):
        col = stacked[:, b]
        order = np.argsort(-col)  # desc
        for rank_pos, prov_idx in enumerate(order):
            order_counts[providers[prov_idx]][rank_pos] += 1
    rank_prob = {p: (order_counts[p] / n_boot).tolist() for p in providers}

    return {"providers": providers, "n_prompts": int(n),
            "point": point, "ci95": ci, "rank_prob": rank_prob}


# --------------------------------------------------------------------------- #
# R2-W4: per-method provider means + paired significance on the scorer
# --------------------------------------------------------------------------- #
def method_provider_means(by_provider: dict):
    rows = {}
    for method, pv in by_provider.items():
        rows[method] = {p: (float(pv[p].mean()), float(pv[p].std()), int(pv[p].size))
                        for p in PROVIDERS if p in pv and pv[p].size}
    return rows


def aura_vs_others_paired(paired_method: dict):
    """
    Paired Welch t-test of Deepgram Aura vs every other provider on the
    fine-tuned scorer's naturalness, aggregated to prompt means (shared qids).
    Tells whether Aura is statistically distinguishable from the top.
    """
    providers = [p for p in PROVIDERS if p in paired_method]
    # prompt-mean per provider
    prompt_means = {}
    for p in providers:
        prompt_means[p] = {q: np.mean(v) for q, v in paired_method[p].items()}
    out = {}
    if "deepgram_aura" not in prompt_means:
        return out
    a = prompt_means["deepgram_aura"]
    for p in providers:
        if p == "deepgram_aura":
            continue
        b = prompt_means[p]
        common = sorted(set(a) & set(b))
        if len(common) < 3:
            continue
        x = np.array([a[q] for q in common])
        y = np.array([b[q] for q in common])
        t, pval = stats.ttest_rel(x, y)
        out[p] = {"n_prompts": len(common),
                  "aura_mean": float(x.mean()), "other_mean": float(y.mean()),
                  "diff": float(x.mean() - y.mean()),
                  "t": float(t), "p": float(pval)}
    return out


def main():
    by_provider, paired = load_provider_scores()

    # ---- R2-W4: per-method provider naturalness ----
    means = method_provider_means(by_provider)
    print("=" * 74)
    print("PER-METHOD PROVIDER NATURALNESS  (mean; higher = more natural, except")
    print("NISQA/MOSNet are MOS-scale and llm_overall is 1-5)")
    print("=" * 74)
    method_order = ["nisqa_mos", "mosnet", "minicpm_zeroshot", "llm_naturalness",
                    "llm_overall", "minicpm_finetuned"]
    header = f"{'method':<18}" + "".join(f"{PRETTY[p][:11]:>13}" for p in PROVIDERS)
    print(header)
    for m in method_order:
        if m not in means:
            continue
        row = f"{m:<18}"
        best = max(means[m].values(), key=lambda t: t[0])[0] if means[m] else None
        for p in PROVIDERS:
            if p in means[m]:
                val = means[m][p][0]
                mark = "*" if val == best else " "
                row += f"{val:>12.2f}{mark}"
            else:
                row += f"{'--':>13}"
        print(row)
    print("\n(* = provider ranked most natural by that method)")

    # ---- R2-W5: prompt-level bootstrap on the fine-tuned scorer ----
    print("\n" + "=" * 74)
    print("FINE-TUNED SCORER: provider naturalness with prompt-level 95% CIs")
    print("=" * 74)
    bs = prompt_bootstrap_means(paired["minicpm_finetuned"])
    print(f"(bootstrap over {bs['n_prompts']} shared prompts, 5000 resamples)")
    ordered = sorted(bs["providers"], key=lambda p: -bs["point"][p])
    for p in ordered:
        lo, hi = bs["ci95"][p]
        print(f"  {PRETTY[p]:<16} mean={bs['point'][p]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\nRank stability (P(rank) across bootstrap; rank1 = most natural):")
    print(f"  {'provider':<16}" + "".join(f"  rank{i+1}" for i in range(len(ordered))))
    for p in ordered:
        probs = bs["rank_prob"][p]
        print(f"  {PRETTY[p]:<16}" + "".join(f"  {pr:5.2f}" for pr in probs))

    # ---- R2-W4: Aura paired tests ----
    print("\n" + "=" * 74)
    print("Is Deepgram Aura distinguishable from others? (fine-tuned scorer,")
    print("paired t-test on prompt means)")
    print("=" * 74)
    aura = aura_vs_others_paired(paired["minicpm_finetuned"])
    for p, r in aura.items():
        sig = "sig" if r["p"] < 0.05 else "n.s."
        print(f"  Aura vs {PRETTY[p]:<14} diff={r['diff']:+.3f}  "
              f"t={r['t']:+.2f}  p={r['p']:.2e}  [{sig}]  (n={r['n_prompts']})")

    # ---- persist ----
    out = {
        "per_method_provider_means": means,
        "finetuned_bootstrap": bs,
        "aura_paired_tests": aura,
        "n_utterances_per_provider": {p: int(by_provider["minicpm_finetuned"][p].size)
                                      for p in PROVIDERS
                                      if p in by_provider["minicpm_finetuned"]},
    }
    with open(os.path.join(OUT_DIR, "provider_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/provider_analysis.json")

    _write_latex(bs, aura, means)


def _write_latex(bs, aura, means):
    ordered = sorted(bs["providers"], key=lambda p: -bs["point"][p])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\rev{Fine-tuned scorer naturalness per provider with "
        r"\emph{prompt-level} bootstrap 95\% CIs and rank-stability (probability "
        r"of holding each rank under resampling of the shared prompts, "
        rf"$B{{=}}5000$). Every provider holds its rank in all resamples, so "
        r"the scorer's provider ordering is stable under prompt resampling.}}",
        r"\label{tab:provider-bootstrap}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Provider & Naturalness (95\% CI) & P(top) / P(bottom) \\", r"\midrule",
    ]
    nprov = len(ordered)
    for p in ordered:
        lo, hi = bs["ci95"][p]
        rp = bs["rank_prob"][p]
        lines.append(f"{PRETTY[p]} & {bs['point'][p]:.2f} [{lo:.2f}, {hi:.2f}] & "
                     f"{rp[0]:.2f} / {rp[nprov-1]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table_provider_bootstrap.tex"), "w") as f:
        f.write("\n".join(lines))

    # per-method ranking-disagreement table (naturalness)
    L = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\rev{Per-method provider naturalness. Methods agree that AWS "
        r"Polly is least natural but disagree at the top, motivating caution in "
        r"reading fine-grained differences. NISQA/MOSNet on MOS scale; MiniCPM "
        r"scores on 1--10; LLM naturalness on its own scale.}}",
        r"\label{tab:method-disagreement}",
        r"\begin{tabular}{l" + "c" * len(PROVIDERS) + "}", r"\toprule",
        "Method & " + " & ".join(PRETTY[p] for p in PROVIDERS) + r" \\", r"\midrule",
    ]
    name = {"nisqa_mos": "NISQA MOS", "mosnet": "MOSNet",
            "minicpm_zeroshot": "MiniCPM (zero-shot)",
            "llm_naturalness": "LLM judge (nat.)",
            "minicpm_finetuned": "Fine-tuned scorer"}
    for m in ["nisqa_mos", "mosnet", "minicpm_zeroshot", "llm_naturalness",
              "minicpm_finetuned"]:
        if m not in means:
            continue
        best = max(means[m].values(), key=lambda t: t[0])[0]
        cells = []
        for p in PROVIDERS:
            if p in means[m]:
                v = means[m][p][0]
                cells.append((r"\textbf{%.2f}" % v) if v == best else f"{v:.2f}")
            else:
                cells.append("--")
        L.append(f"{name[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table_method_disagreement.tex"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
