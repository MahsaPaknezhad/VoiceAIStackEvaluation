"""
Reusable statistics for inter-annotator agreement (IAA) and reliability.

Implemented with only numpy/scipy so the reviewer-response analysis adds no new
dependencies. Covers the methods that are standard for subjective perceptual
(MOS-style) ratings:

  - Pearson / Spearman correlation (pairwise annotator agreement)
  - Krippendorff's alpha (interval level) for >=2 raters with missing data
  - Intraclass Correlation Coefficient, ICC(2,1) and ICC(2,k) (two-way random,
    absolute agreement) following Shrout & Fleiss (1979)

All functions operate on an (n_items x n_raters) matrix with np.nan for missing
ratings so partial overlap between annotators is handled gracefully.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Pairwise correlation
# --------------------------------------------------------------------------- #
def pairwise_correlations(matrix: np.ndarray, rater_names: list[str]) -> list[dict]:
    """Pearson + Spearman for every rater pair, over their common non-nan items."""
    out = []
    n_raters = matrix.shape[1]
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            a, b = matrix[:, i], matrix[:, j]
            mask = ~(np.isnan(a) | np.isnan(b))
            x, y = a[mask], b[mask]
            if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                out.append({"pair": f"{rater_names[i]}-{rater_names[j]}", "n": int(mask.sum()),
                            "pearson_r": float("nan"), "pearson_p": float("nan"),
                            "spearman_rho": float("nan"), "spearman_p": float("nan")})
                continue
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            out.append({"pair": f"{rater_names[i]}-{rater_names[j]}", "n": int(mask.sum()),
                        "pearson_r": float(pr), "pearson_p": float(pp),
                        "spearman_rho": float(sr), "spearman_p": float(sp)})
    return out


def mean_pairwise(correlations: list[dict], key: str = "pearson_r") -> float:
    vals = [c[key] for c in correlations if not np.isnan(c[key])]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (interval metric)
# --------------------------------------------------------------------------- #
def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """
    Krippendorff's alpha for interval data.

    matrix: (n_items x n_raters) with np.nan for missing values.
    Uses the squared-difference metric. alpha = 1 - Do/De.
    """
    data = np.asarray(matrix, dtype=float)

    # Observed disagreement: average squared diff over all rating pairs within items
    num_obs = 0.0
    den_obs = 0.0
    for row in data:
        vals = row[~np.isnan(row)]
        m = len(vals)
        if m < 2:
            continue
        # sum of squared pairwise differences within the item / (m - 1)
        diffs = np.subtract.outer(vals, vals) ** 2
        num_obs += diffs.sum() / (m - 1)
        den_obs += m
    if den_obs == 0:
        return float("nan")
    Do = num_obs / den_obs

    # Expected disagreement: squared diffs over all value pairs in the dataset
    all_vals = data[~np.isnan(data)]
    N = len(all_vals)
    if N < 2:
        return float("nan")
    diffs_all = np.subtract.outer(all_vals, all_vals) ** 2
    De = diffs_all.sum() / (N * (N - 1))
    if De == 0:
        return float("nan")
    return float(1.0 - Do / De)


# --------------------------------------------------------------------------- #
# Intraclass Correlation Coefficient ICC(2,1) / ICC(2,k)
# --------------------------------------------------------------------------- #
def icc_2(matrix: np.ndarray) -> dict:
    """
    Two-way random-effects, absolute-agreement ICC (Shrout & Fleiss 1979).
    Returns ICC(2,1) (single rater) and ICC(2,k) (mean of k raters).

    Requires complete rows; rows with any nan are dropped.
    """
    data = np.asarray(matrix, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 2 or k < 2:
        return {"icc_2_1": float("nan"), "icc_2_k": float("nan"), "n": int(n), "k": int(k)}

    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Mean squares
    SS_total = ((data - grand) ** 2).sum()
    SS_rows = k * ((row_means - grand) ** 2).sum()           # between targets
    SS_cols = n * ((col_means - grand) ** 2).sum()           # between raters
    SS_err = SS_total - SS_rows - SS_cols

    MS_rows = SS_rows / (n - 1)
    MS_cols = SS_cols / (k - 1)
    MS_err = SS_err / ((n - 1) * (k - 1))

    # ICC(2,1): absolute agreement, single measures
    denom1 = MS_rows + (k - 1) * MS_err + (k * (MS_cols - MS_err) / n)
    icc_2_1 = (MS_rows - MS_err) / denom1 if denom1 != 0 else float("nan")

    # ICC(2,k): absolute agreement, average measures
    denomk = MS_rows + (MS_cols - MS_err) / n
    icc_2_k = (MS_rows - MS_err) / denomk if denomk != 0 else float("nan")

    return {"icc_2_1": float(icc_2_1), "icc_2_k": float(icc_2_k), "n": int(n), "k": int(k)}


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval helper
# --------------------------------------------------------------------------- #
def bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray, stat_fn,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI for a paired statistic (e.g. Pearson r)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    point = stat_fn(a, b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stat_fn(a[idx], b[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": float(point), "ci_low": lo, "ci_high": hi, "n": int(n), "n_boot": n_boot}


def pearson_stat(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return stats.pearsonr(x, y)[0]
