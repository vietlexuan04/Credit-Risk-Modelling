import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import discrimination_metrics  # noqa: E402
from modeling import backward_select, vif_table  # noqa: E402


def _data(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "noise": rng.normal(size=n)})
    logit = -2.7 - 1.0 * X["a"] - 0.5 * X["b"]
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_backward_select_drops_noise_variable():
    X, y = _data()
    cols, model, log = backward_select(X, y)
    assert "noise" not in cols and set(cols) == {"a", "b"}
    assert (model.params.drop("const") < 0).all()


def test_backward_select_drops_wrong_sign_variable():
    X, y = _data()
    X["wrong"] = -X["a"] + np.random.default_rng(2).normal(scale=0.3, size=len(X))
    cols, _, log = backward_select(X, y)
    assert "wrong" not in cols


def test_vif_detects_collinearity():
    X, _ = _data()
    X["a_copy"] = X["a"] + np.random.default_rng(3).normal(scale=0.05, size=len(X))
    v = vif_table(X)
    assert v["a"] > 10 and v["noise"] < 2


def test_metrics_on_perfect_and_random_scores():
    y = pd.Series([0, 0, 1, 1])
    assert np.isclose(discrimination_metrics(y, pd.Series([0.1, 0.2, 0.8, 0.9]))["gini"], 1.0)
    assert np.isclose(discrimination_metrics(y, pd.Series([0.5, 0.5, 0.5, 0.5]))["auc"], 0.5)


def test_rank_ordering_calibration_and_psi_helpers():
    from evaluation import (assign_grade, calibration_table, cutoff_table, grade_edges,
                            hosmer_lemeshow, psi, rank_ordering_table)
    rng = np.random.default_rng(0)
    score = rng.normal(600, 40, 50000)
    p = 1 / (1 + np.exp((score - 520) / 28.85))
    y = (rng.uniform(size=len(p)) < p).astype(int)
    edges = grade_edges(score, 10)
    grades = assign_grade(score, edges)
    t = rank_ordering_table(y, score, p, edges)
    assert grades.min() == 1 and grades.max() == 10 and np.all(np.diff(t["bad_rate"]) > 0)
    assert hosmer_lemeshow(y, p, grades)["p_value"] > 0.01
    assert (calibration_table(y, p, grades)["binom_p_value"] > 0.001).all()
    assert psi(score, score, edges) < 1e-9
    assert psi(score, score - 40, edges) > 0.25
    c = cutoff_table(y, score, [500, 600])
    assert c["approval_rate"].iloc[0] > c["approval_rate"].iloc[1]
