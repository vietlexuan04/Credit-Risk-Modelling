import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_engineering import WoEBinner, _woe  # noqa: E402


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < 0.02 + 0.2 * x).astype(int)   # risk rises with x
    return pd.Series(x), pd.Series(y)


def test_woe_is_monotone_in_requested_direction():
    x, y = _synthetic()
    b = WoEBinner("x", trend="increasing_risk", min_share=0.05).fit(x, y)
    assert np.all(np.diff(b.woe_) < 0)
    b2 = WoEBinner("x", trend="decreasing_risk", min_share=0.05).fit(-x, y)
    assert np.all(np.diff(b2.woe_) > 0)


def test_bin_size_and_bin_count_constraints():
    x, y = _synthetic()
    b = WoEBinner("x", trend="increasing_risk", min_share=0.05, max_bins=6).fit(x, y)
    assert len(b.woe_) <= 6
    assert (b.table_["pct_rows"] >= 0.05 - 1e-9).all()


def test_iv_matches_manual_computation():
    x, y = _synthetic()
    b = WoEBinner("x", trend="increasing_risk").fit(x, y)
    t = b.table_
    G, B = (y == 0).sum(), (y == 1).sum()
    manual = ((t.good / G - t.bad / B) * np.log((t.good / G) / (t.bad / B))).sum()
    assert np.isclose(manual, b.iv_)


def test_transform_matches_table_and_totals_add_up():
    x, y = _synthetic()
    b = WoEBinner("x", trend="increasing_risk").fit(x, y)
    assert b.table_["rows"].sum() == len(x)
    w = b.transform(x)
    assert set(np.round(w.unique(), 10)) == set(np.round(b.woe_, 10))


def test_missing_gets_own_bin_when_enough_bad_else_merged():
    x, y = _synthetic()
    x = x.copy()
    x.iloc[:3000] = np.nan
    b = WoEBinner("x", trend="increasing_risk").fit(x, y)
    assert b.missing_mode_ == "own_bin" and b.table_["bin"].iloc[-1] == "Missing"
    x2, y2 = _synthetic()
    x2 = x2.copy()
    x2.iloc[:50] = np.nan
    y2 = y2.copy()
    y2.iloc[:50] = 0
    b2 = WoEBinner("x", trend="increasing_risk").fit(x2, y2)
    assert b2.missing_mode_ == "merged"
    assert not b2.transform(x2).isna().any()


def test_zero_count_woe_is_finite():
    assert np.isfinite(_woe(10, 0, 1000, 100))
