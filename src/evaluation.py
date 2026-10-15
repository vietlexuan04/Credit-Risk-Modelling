"""
Discrimination, rank ordering, calibration, cut-off and stability helpers.

Convention: `p` is the predicted probability of BAD (higher = riskier) and
`y` is 1 for bad, 0 for good. Scores are the opposite: higher score = safer.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    """Largest gap between the cumulative distributions of bad and good over the score."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination_metrics(y, p) -> dict:
    """AUC, Gini (= 2*AUC - 1) and KS."""
    auc = float(roc_auc_score(y, p))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, p)}


def grade_edges(score_dev, n_grades: int = 10) -> np.ndarray:
    """Score cut points (quantiles of the DEVELOPMENT score) that define the grades."""
    qs = np.linspace(0, 1, n_grades + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(score_dev, dtype=float), qs))


def assign_grade(score, edges: np.ndarray) -> pd.Series:
    """Grade 1 = best (highest scores) ... grade n = worst (lowest scores)."""
    s = pd.Series(score)
    idx = np.searchsorted(edges, s.to_numpy(), side="right")      # 0 = lowest scores
    return pd.Series(len(edges) + 1 - idx, index=s.index, name="grade")


def rank_ordering_table(y, score, pd_pred, edges) -> pd.DataFrame:
    """Bad rate by grade. Healthy: the bad rate increases steadily from grade 1 to grade n."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score), "pd": np.asarray(pd_pred)})
    df["grade"] = assign_grade(df["score"], edges).to_numpy()
    g = df.groupby("grade").agg(rows=("y", "size"), bad=("y", "sum"), bad_rate=("y", "mean"),
                                avg_pd=("pd", "mean"), min_score=("score", "min"), max_score=("score", "max"))
    g["pct_rows"] = g["rows"] / len(df)
    return g.reset_index()


def calibration_table(y, pd_pred, grades) -> pd.DataFrame:
    """Predicted PD vs observed bad rate per grade, with a two-sided binomial test."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_pred), "grade": np.asarray(grades)})
    g = df.groupby("grade").agg(rows=("y", "size"), bad=("y", "sum"), avg_pd=("pd", "mean"), bad_rate=("y", "mean"))
    g["ratio_obs_pred"] = g["bad_rate"] / g["avg_pd"]
    g["binom_p_value"] = [stats.binomtest(int(r.bad), int(r.rows), float(r.avg_pd)).pvalue for r in g.itertuples()]
    return g.reset_index()


def hosmer_lemeshow(y, pd_pred, grades, dof_reduction: int = 2) -> dict:
    """Hosmer-Lemeshow chi-square over the given groups.

    df = groups - dof_reduction. Use 2 (the classic choice) when the PDs were fitted on the same data;
    use 0 on an independent sample where both the groups and the PDs were fixed beforehand
    (e.g. hold-out validation), as recommended by Hosmer and Lemeshow for external validation.
    """
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_pred), "grade": np.asarray(grades)})
    g = df.groupby("grade").agg(n=("y", "size"), obs=("y", "sum"), exp=("pd", "sum"))
    stat = float((((g["obs"] - g["exp"]) ** 2) / (g["exp"] * (1 - g["exp"] / g["n"]))).sum())
    dof = len(g) - dof_reduction
    return {"hl_statistic": stat, "df": dof, "p_value": float(1 - stats.chi2.cdf(stat, dof))}


def probability_scores(y, pd_pred) -> dict:
    """Brier score and log loss (lower = better) plus mean predicted vs observed bad rate."""
    return {"brier": float(brier_score_loss(y, pd_pred)), "log_loss": float(log_loss(y, pd_pred)),
            "mean_pd": float(np.mean(pd_pred)), "observed_bad_rate": float(np.mean(y))}


def bootstrap_ci(y, p, n_boot: int = 500, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap confidence interval of AUC and Gini."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        aucs[i] = roc_auc_score(y[idx], p[idx])
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return {"auc_lo": float(lo), "auc_hi": float(hi), "gini_lo": float(2 * lo - 1), "gini_hi": float(2 * hi - 1)}


def psi(expected, actual, edges) -> float:
    """Population Stability Index of a score between two populations, using the given bin edges."""
    e = np.bincount(np.searchsorted(edges, np.asarray(expected, dtype=float), side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, np.asarray(actual, dtype=float), side="right"), minlength=len(edges) + 1)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


def cutoff_table(y, score, cutoffs) -> pd.DataFrame:
    """For each cut-off (approve if score >= cut-off): approval rate and bad rates."""
    y, s = np.asarray(y), np.asarray(score, dtype=float)
    total_bad, total_good = y.sum(), (1 - y).sum()
    rows = []
    for c in cutoffs:
        ok = s >= c
        rows.append({
            "cutoff": float(c), "approval_rate": ok.mean(),
            "bad_rate_approved": y[ok].mean() if ok.any() else np.nan,
            "bad_rate_rejected": y[~ok].mean() if (~ok).any() else np.nan,
            "bads_rejected_share": y[~ok].sum() / total_bad,
            "goods_rejected_share": (1 - y[~ok]).sum() / total_good,
        })
    return pd.DataFrame(rows)
