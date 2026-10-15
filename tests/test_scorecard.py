import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_engineering import WoEBinner  # noqa: E402
from modeling import fit_logit  # noqa: E402
from scorecard import Scorecard, score_to_pd, scaling_params  # noqa: E402


def test_scaling_matches_the_textbook_example():
    factor, offset = scaling_params(600, 50, 20)
    assert np.isclose(factor, 28.8539, atol=1e-3) and np.isclose(offset, 487.12, atol=1e-2)
    score = lambda odds: offset + factor * np.log(odds)
    assert np.isclose(score(50), 600) and np.isclose(score(100), 620)


def test_score_to_pd_roundtrip():
    factor, offset = scaling_params()
    assert np.isclose(score_to_pd(600, factor, offset), 1 / 51)
    assert score_to_pd(620, factor, offset) < score_to_pd(600, factor, offset)


def _toy():
    rng = np.random.default_rng(0)
    n = 40000
    df = pd.DataFrame({"x1": rng.uniform(size=n), "x2": rng.uniform(size=n)})
    p = 1 / (1 + np.exp(-(-3 + 2 * df.x1 + 1.5 * df.x2)))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return df, y


def test_points_reproduce_model_log_odds():
    df, y = _toy()
    binners = {c: WoEBinner(c, trend="increasing_risk", min_share=0.05).fit(df[c], y) for c in df}
    X = pd.DataFrame({c: binners[c].transform(df[c]) for c in df})
    m = fit_logit(X, y)
    factor, offset = scaling_params()
    coefs = {c: float(m.params[c]) for c in df}
    exact = Scorecard(binners, float(m.params["const"]), coefs, factor, offset, round_points=False)
    z = m.params["const"] + (X * m.params[list(df)]).sum(axis=1)
    assert np.allclose(exact.score(df), offset - factor * z)
    rounded = Scorecard(binners, float(m.params["const"]), coefs, factor, offset, round_points=True)
    assert np.abs(rounded.score(df) - exact.score(df)).max() <= 0.5 * len(coefs) + 1e-9
