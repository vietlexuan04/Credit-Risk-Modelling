"""
Scorecard: converts the WoE logistic regression into points.

Scaling (PDO method)
--------------------
Score = Offset + Factor * ln(odds good:bad)
Factor = PDO / ln(2)            (PDO = points that double the odds)
Offset = BaseScore - Factor * ln(BaseOdds)

The model gives log-odds of BAD:  z = b0 + sum_i b_i * WoE_i.
So ln(odds good:bad) = -z and
Score = Offset - Factor * (b0 + sum_i b_i * WoE_i)
      = sum_i [ -Factor * b_i * WoE_i + (Offset - Factor * b0) / n_vars ].
Each bin therefore has a fixed number of points; the intercept part is spread
equally over the variables (a common convention).
"""
import numpy as np
import pandas as pd


def scaling_params(base_score: float = 600, base_odds: float = 50, pdo: float = 20):
    """Factor and Offset of the PDO scaling (base_odds = good:bad)."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return float(factor), float(offset)


def score_to_pd(score, factor: float, offset: float):
    """PD implied by a score: odds good:bad = exp((score - offset) / factor)."""
    odds_good = np.exp((np.asarray(score, dtype=float) - offset) / factor)
    return 1.0 / (1.0 + odds_good)


class Scorecard:
    def __init__(self, binners: dict, intercept: float, coefs: dict, factor: float, offset: float,
                 round_points: bool = True):
        self.binners = {v: binners[v] for v in coefs}
        self.intercept, self.coefs = intercept, dict(coefs)
        self.factor, self.offset, self.round_points = factor, offset, round_points
        n = len(coefs)
        self.base_points = (offset - factor * intercept) / n
        self.points = {}
        for v, b in self.binners.items():
            woe_all = np.append(b.woe_, b.missing_woe_) if b.missing_mode_ == "own_bin" else b.woe_
            pts = -factor * self.coefs[v] * woe_all + self.base_points
            self.points[v] = np.round(pts) if round_points else pts

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Total score of each row (sum of the points of the bin it falls in)."""
        total = pd.Series(0.0, index=df.index)
        for v, b in self.binners.items():
            total += self.points[v][b.bin_index(df[v]).to_numpy()]
        return total

    def points_table(self, bin_tables: pd.DataFrame) -> pd.DataFrame:
        """Readable scorecard: one row per bin with its points."""
        out = []
        for v in self.binners:
            t = bin_tables[bin_tables["variable"] == v].reset_index(drop=True)
            assert len(t) == len(self.points[v]), f"bin count mismatch for {v}"
            t = t.assign(coef=self.coefs[v], points=self.points[v])
            out.append(t[["variable", "bin", "rows", "pct_rows", "bad_rate", "woe", "coef", "points"]])
        return pd.concat(out, ignore_index=True)

    def score_range(self) -> tuple:
        lo = sum(p.min() for p in self.points.values())
        hi = sum(p.max() for p in self.points.values())
        return float(lo), float(hi)
