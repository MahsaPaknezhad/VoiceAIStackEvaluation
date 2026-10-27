"""
Reviewer-response analysis: inter-annotator agreement (IAA) "human ceiling" and
model-vs-human comparison on the 76-sample held-out test set.

Directly addresses the two weaknesses both reviewers raised:
  R1-W1 / R2-W1 : "single annotator, no inter-annotator agreement reported"
  R2-W2         : "the per-clip human validation uses only one rater"

Analyses produced:
  1. Pairwise annotator correlations (Pearson + Spearman) per dimension.
  2. Krippendorff's alpha (interval) and ICC(2,1)/ICC(2,k) as the agreement ceiling.
  3. Mean human-human correlation vs. the fine-tuned scorer's human correlation,
     with the scorer evaluated against (a) the original annotator and
     (b) the 3-annotator consensus (mean), including bootstrap 95% CIs.
  4. Model-vs-unseen-annotator correlation (brian, joe contributed no training
     labels) to show the scorer captures shared perception, not idiosyncrasy.

Outputs JSON + LaTeX tables under results/.
Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python run_iaa_analysis.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import agreement_stats as ag
import data_io as io

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def analyse_dimension(dim: str, test_ids: list[str]) -> dict:
    mat = io.build_rater_matrix(dim, test_ids)

    pairwise = ag.pairwise_correlations(mat, io.RATERS)
    mean_r = ag.mean_pairwise(pairwise, "pearson_r")
    mean_rho = ag.mean_pairwise(pairwise, "spearman_rho")
    alpha = ag.krippendorff_alpha_interval(mat)
    icc = ag.icc_2(mat)

    # consensus = mean over available raters per item
    consensus = np.nanmean(mat, axis=1)

    return {
        "dimension": dim,
        "n_items": int((~np.isnan(mat).all(axis=1)).sum()),
        "pairwise": pairwise,
        "mean_human_human_pearson": mean_r,
        "mean_human_human_spearman": mean_rho,
        "krippendorff_alpha_interval": alpha,
        "icc_2_1": icc["icc_2_1"],
        "icc_2_k": icc["icc_2_k"],
        "_matrix": mat,
        "_consensus": consensus,
    }


def model_vs_human(dim: str, test_ids: list[str], dim_result: dict,
                   preds: dict) -> dict:
    mat = dim_result["_matrix"]
    consensus = dim_result["_consensus"]
    pred_key = f"pred_{dim}"

    model = np.array([preds[sid][pred_key] if sid in preds and pred_key in preds[sid]
                      else np.nan for sid in test_ids])

    res = {"dimension": dim}
    # model vs each rater
    for idx, name in enumerate(io.RATERS):
        ci = ag.bootstrap_ci(model, mat[:, idx], ag.pearson_stat)
        res[f"model_vs_{name}"] = ci
    # model vs consensus
    res["model_vs_consensus"] = ag.bootstrap_ci(model, consensus, ag.pearson_stat)
    return res


def fmt_p(p):
    if np.isnan(p):
        return "--"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.3g}"


def write_latex_iaa(results: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-annotator agreement on the held-out test set ($n{=}76$), "
        r"three independent annotators. Mean pairwise Pearson $r$ is the "
        r"\emph{human agreement ceiling}; Krippendorff's $\alpha$ (interval) and "
        r"ICC(2,k) measure overall reliability.}",
        r"\label{tab:iaa}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dimension & Mean $r$ & Mean $\rho$ & Kripp. $\alpha$ & ICC(2,1) & ICC(2,k) \\",
        r"\midrule",
    ]
    for res in results:
        lines.append(
            f"{res['dimension'].capitalize()} & "
            f"{res['mean_human_human_pearson']:.3f} & "
            f"{res['mean_human_human_spearman']:.3f} & "
            f"{res['krippendorff_alpha_interval']:.3f} & "
            f"{res['icc_2_1']:.3f} & "
            f"{res['icc_2_k']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_latex_pairwise(results: list[dict]) -> str:
    # Build pair keys dynamically from the configured raters, in the same
    # order pairwise_correlations emits them (i<j).
    raters = io.RATERS
    pair_keys = [f"{raters[i]}-{raters[j]}"
                 for i in range(len(raters)) for j in range(i + 1, len(raters))]
    header = " & ".join(k.replace("-", "--") for k in pair_keys)
    col_spec = "l" + "c" * len(pair_keys)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pairwise annotator Pearson $r$ (Spearman $\rho$) per dimension, "
        r"$n{=}76$.}",
        r"\label{tab:iaa-pairwise}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Dimension & {header} \\\\",
        r"\midrule",
    ]
    for res in results:
        cells = {}
        for pw in res["pairwise"]:
            cells[pw["pair"]] = f"{pw['pearson_r']:.3f} ({pw['spearman_rho']:.3f})"
        row = " & ".join(cells.get(k, "--") for k in pair_keys)
        lines.append(f"{res['dimension'].capitalize()} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_latex_ceiling(iaa: list[dict], mvh: list[dict],
                        model_metrics: dict) -> str:
    """Compares the fine-tuned scorer's human correlation to the human ceiling."""
    mvh_by_dim = {m["dimension"]: m for m in mvh}
    iaa_by_dim = {r["dimension"]: r for r in iaa}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The fine-tuned scorer relative to the human agreement ceiling "
        r"($n{=}76$). `Model vs.\ labeler1' reproduces the paper's headline $r$ "
        r"(labeler1 is the original training annotator); `Model vs.\ consensus' and "
        r"`Model vs.\ unseen (labeler2/3)' show the scorer generalises beyond the "
        r"single training annotator. 95\% bootstrap CIs in brackets.}",
        r"\label{tab:human-ceiling}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dimension & Human ceiling & Model vs.\ labeler1 & "
        r"Model vs.\ consensus & Model vs.\ labeler2/3 \\",
        r"\midrule",
    ]
    for dim in io.DIMENSIONS:
        m = mvh_by_dim[dim]
        ceil = iaa_by_dim[dim]["mean_human_human_pearson"]
        orig = m["model_vs_labeler1"]
        cons = m["model_vs_consensus"]
        unseen_raters = [r for r in io.RATERS if r != "labeler1"]
        unseen = np.nanmean([m[f"model_vs_{r}"]["point"] for r in unseen_raters])
        lines.append(
            f"{dim.capitalize()} & "
            f"{ceil:.3f} & "
            f"{orig['point']:.3f} [{orig['ci_low']:.3f}, {orig['ci_high']:.3f}] & "
            f"{cons['point']:.3f} [{cons['ci_low']:.3f}, {cons['ci_high']:.3f}] & "
            f"{unseen:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    test_ids = io.load_test_ids()
    preds, model_metrics = io.load_model_predictions()

    iaa_results = [analyse_dimension(d, test_ids) for d in io.DIMENSIONS]
    mvh_results = [model_vs_human(d, test_ids, r, preds)
                   for d, r in zip(io.DIMENSIONS, iaa_results)]

    # ---- console summary ----
    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT  (held-out test set, n=76, 3 annotators)")
    print("=" * 70)
    for res in iaa_results:
        print(f"\n[{res['dimension'].upper()}]  items={res['n_items']}")
        for pw in res["pairwise"]:
            print(f"  {pw['pair']:<18} r={pw['pearson_r']:.3f}  rho={pw['spearman_rho']:.3f}"
                  f"  (n={pw['n']})")
        print(f"  mean human-human r   = {res['mean_human_human_pearson']:.3f}  "
              f"(ceiling)")
        print(f"  Krippendorff alpha   = {res['krippendorff_alpha_interval']:.3f}")
        print(f"  ICC(2,1)={res['icc_2_1']:.3f}   ICC(2,k)={res['icc_2_k']:.3f}")

    print("\n" + "=" * 70)
    print("MODEL vs HUMAN  (fine-tuned scorer, 95% bootstrap CI)")
    print("=" * 70)
    for m in mvh_results:
        print(f"\n[{m['dimension'].upper()}]")
        for name in io.RATERS + ["consensus"]:
            c = m[f"model_vs_{name}"]
            print(f"  vs {name:<10} r={c['point']:.3f}  "
                  f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]  (n={c['n']})")
        ceil = next(r["mean_human_human_pearson"] for r in iaa_results
                    if r["dimension"] == m["dimension"])
        print(f"  -> human ceiling r = {ceil:.3f};  "
              f"model reaches {100 * m['model_vs_consensus']['point'] / ceil:.0f}% "
              f"of ceiling (vs consensus)")

    # ---- persist JSON (strip numpy internals) ----
    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}
    json_out = {
        "n_test": len(test_ids),
        "raters": io.RATERS,
        "iaa": [clean(r) for r in iaa_results],
        "model_vs_human": mvh_results,
        "paper_reported_model_metrics": model_metrics,
    }
    with open(os.path.join(OUT_DIR, "iaa_results.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # ---- LaTeX tables ----
    with open(os.path.join(OUT_DIR, "table_iaa.tex"), "w") as f:
        f.write(write_latex_iaa(iaa_results))
    with open(os.path.join(OUT_DIR, "table_iaa_pairwise.tex"), "w") as f:
        f.write(write_latex_pairwise(iaa_results))
    with open(os.path.join(OUT_DIR, "table_human_ceiling.tex"), "w") as f:
        f.write(write_latex_ceiling(iaa_results, mvh_results, model_metrics))

    print(f"\nWrote JSON + 3 LaTeX tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
