"""
WoE binning (coarse classing) for the PD scorecard.

Core objects
------------
WoEBinner : fits bins on ONE variable using the development set, then
            transforms any data (development or hold-out) to WoE values.
            Nothing is ever fitted on the hold-out set.

Binning procedure (per variable)
--------------------------------
1. Fine classing: every distinct value (<= 30 distinct values) or ~40 quantile
   bins.
2. Coarse classing: repeatedly merge ADJACENT bins until all rules hold:
   a) WoE is monotonic in the direction given by business logic
      (trend = "increasing_risk" | "decreasing_risk"), or "free" (no constraint);
   b) every bin has at least `min_share` of all development rows and at least
      `min_bad` bad cases;
   c) adjacent bins differ by at least `min_woe_gap` in WoE (otherwise they are
      practically the same);
   d) at most `max_bins` bins remain: the pair whose merge loses the least IV
      is merged first.
3. NaN is NOT imputed. If the NaN group has at least `min_missing_bad` bad
   cases it gets its own "Missing" bin; otherwise it is assigned to the
   numeric bin with the closest WoE (too few bads to estimate its own WoE).

WoE = ln(%good / %bad) with %good, %bad measured against the totals of the
whole development set (including missing rows). IV = sum((%good - %bad) * WoE).
"""
import numpy as np
import pandas as pd

TRENDS = ("increasing_risk", "decreasing_risk", "free")


def _woe(good, bad, total_good, total_bad):
    """WoE of one bin. Adds 0.5 to both counts only if a count is zero."""
    if good == 0 or bad == 0:
        good, bad = good + 0.5, bad + 0.5
    return float(np.log((good / total_good) / (bad / total_bad)))


def iv_strength(iv: float) -> str:
    """Conventional reading of Information Value."""
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very strong (check for leakage)"


class WoEBinner:
    def __init__(self, name, trend="free", min_share=0.05, min_bad=50,
                 min_woe_gap=0.05, min_missing_bad=30, n_fine=40, max_bins=8):
        assert trend in TRENDS, f"trend must be one of {TRENDS}"
        self.name = name
        self.trend = trend
        self.min_share = min_share
        self.min_bad = min_bad
        self.min_woe_gap = min_woe_gap
        self.min_missing_bad = min_missing_bad
        self.n_fine = n_fine
        self.max_bins = max_bins

    # ------------------------------------------------------------------ fit
    def fit(self, x: pd.Series, y: pd.Series):
        x = pd.Series(x).astype(float)
        y = pd.Series(y).astype(int)
        n_total = len(x)
        self.total_bad_ = int(y.sum())
        self.total_good_ = int(n_total - self.total_bad_)
        G, B = self.total_good_, self.total_bad_
        min_n = self.min_share * n_total

        miss = x.isna()
        xv, yv = x[~miss].to_numpy(), y[~miss].to_numpy()
        self.is_integer_ = bool(np.all(np.mod(xv, 1) == 0))
        self.min_value_ = float(xv.min())

        # 1. fine classing: upper edges of right-closed bins
        uniq = np.unique(xv)
        if len(uniq) <= 30:
            edges = uniq[:-1]
        else:
            qs = np.quantile(xv, np.linspace(0, 1, self.n_fine + 1)[1:-1])
            edges = np.unique(qs)
        idx = np.searchsorted(edges, xv, side="left")
        k = len(edges) + 1
        bad = np.bincount(idx, weights=yv, minlength=k).astype(int)
        cnt = np.bincount(idx, minlength=k)
        # bins: [upper_edge, n, good, bad]; last bin has upper edge +inf
        bins = [[(edges[i] if i < len(edges) else np.inf), int(cnt[i]), int(cnt[i] - bad[i]), int(bad[i])]
                for i in range(k) if cnt[i] > 0]

        def merge(i, j):
            lo, hi = min(i, j), max(i, j)
            bins[lo] = [bins[hi][0], bins[lo][1] + bins[hi][1], bins[lo][2] + bins[hi][2], bins[lo][3] + bins[hi][3]]
            del bins[hi]

        # 2. coarse classing
        while len(bins) > 1:
            w = [_woe(b[2], b[3], G, B) for b in bins]
            if self.trend != "free":
                sign = 1 if self.trend == "decreasing_risk" else -1
                viol = [i for i in range(len(bins) - 1) if sign * (w[i + 1] - w[i]) < 0]
                if viol:
                    merge(viol[0], viol[0] + 1)
                    continue
            small = [i for i, b in enumerate(bins) if b[1] < min_n or b[3] < self.min_bad]
            if small:
                i = min(small, key=lambda t: bins[t][1])
                cand = [j for j in (i - 1, i + 1) if 0 <= j < len(bins)]
                j = min(cand, key=lambda t: abs(w[t] - w[i]))
                merge(i, j)
                continue
            gaps = [abs(w[i + 1] - w[i]) for i in range(len(bins) - 1)]
            if min(gaps) < self.min_woe_gap:
                i = int(np.argmin(gaps))
                merge(i, i + 1)
                continue
            if len(bins) > self.max_bins:
                # merge the adjacent pair whose merge loses the least Information Value
                def iv_of(g, b):
                    return (g / G - b / B) * _woe(g, b, G, B)
                loss = [iv_of(bins[i][2], bins[i][3]) + iv_of(bins[i + 1][2], bins[i + 1][3])
                        - iv_of(bins[i][2] + bins[i + 1][2], bins[i][3] + bins[i + 1][3])
                        for i in range(len(bins) - 1)]
                i = int(np.argmin(loss))
                merge(i, i + 1)
                continue
            break

        # 3. missing values
        self.missing_mode_ = "none"
        self.missing_target_bin_ = None
        n_miss, bad_miss = int(miss.sum()), int(y[miss].sum())
        miss_row = None
        if n_miss > 0:
            if bad_miss >= self.min_missing_bad:
                self.missing_mode_ = "own_bin"
                miss_row = [np.nan, n_miss, n_miss - bad_miss, bad_miss]
            else:
                self.missing_mode_ = "merged"
                wm = _woe(n_miss - bad_miss, bad_miss, G, B)
                w = [_woe(b[2], b[3], G, B) for b in bins]
                t = int(np.argmin([abs(v - wm) for v in w]))
                self.missing_target_bin_ = t
                bins[t][1] += n_miss
                bins[t][2] += n_miss - bad_miss
                bins[t][3] += bad_miss

        self.edges_ = np.array([b[0] for b in bins[:-1]], dtype=float)
        rows = list(bins) + ([miss_row] if miss_row is not None else [])
        self.table_ = self._build_table(rows, bins_numeric=len(bins))
        self.woe_ = self.table_["woe"].to_numpy()[: len(bins)]
        self.missing_woe_ = (float(self.table_["woe"].iloc[-1]) if self.missing_mode_ == "own_bin"
                             else (float(self.woe_[self.missing_target_bin_]) if self.missing_mode_ == "merged" else 0.0))
        self.iv_ = float(self.table_["iv_contrib"].sum())
        return self

    # --------------------------------------------------------------- labels
    def _build_table(self, rows, bins_numeric):
        self._tmp_edges = np.array([r[0] for r in rows[: bins_numeric - 1]], dtype=float)
        G, B = self.total_good_, self.total_bad_
        out = []
        for i, r in enumerate(rows):
            _, n, good, bad = r
            w = _woe(good, bad, G, B)
            label = "Missing" if i >= bins_numeric else self._label(i, bins_numeric)
            if self.missing_mode_ == "merged" and i == self.missing_target_bin_:
                label += " or Missing"   # NaN rows were merged into this bin (too few bads for their own)
            out.append({
                "variable": self.name, "bin": label, "rows": n,
                "pct_rows": n / (G + B), "good": good, "bad": bad,
                "bad_rate": bad / n if n else np.nan, "woe": w,
                "iv_contrib": (good / G - bad / B) * w,
            })
        return pd.DataFrame(out)

    def _label(self, i, k):
        e = self._tmp_edges
        if self.is_integer_:
            lo = self.min_value_ if i == 0 else e[i - 1] + 1
            hi = e[i] if i < k - 1 else None
            if hi is None:
                return f"{int(lo)}+"
            return f"{int(lo)}" if lo == hi else f"{int(lo)}-{int(hi)}"
        if i == 0:
            return f"<= {e[0]:.4g}" if k > 1 else "all"
        if i == k - 1:
            return f"> {e[-1]:.4g}"
        return f"({e[i - 1]:.4g}, {e[i]:.4g}]"

    # ------------------------------------------------------------ transform
    def bin_index(self, x: pd.Series) -> pd.Series:
        """Bin number of each row (NaN rows get the missing bin, or the merged bin)."""
        x = pd.Series(x).astype(float)
        idx = pd.Series(np.searchsorted(self.edges_, x.to_numpy(), side="left"), index=x.index)
        if self.missing_mode_ == "own_bin":
            idx[x.isna()] = len(self.woe_)
        elif self.missing_mode_ == "merged":
            idx[x.isna()] = self.missing_target_bin_
        return idx

    def transform(self, x: pd.Series) -> pd.Series:
        """WoE value of each row."""
        x = pd.Series(x).astype(float)
        woe = pd.Series(self.woe_[np.searchsorted(self.edges_, x.to_numpy(), side="left")], index=x.index)
        if x.isna().any():
            if self.missing_mode_ == "none":
                raise ValueError(f"{self.name}: NaN found at transform time but none at fit time")
            woe[x.isna()] = self.missing_woe_
        return woe

    # --------------------------------------------------------- persistence
    def to_dict(self):
        return {
            "name": self.name, "trend": self.trend, "edges": self.edges_.tolist(),
            "woe": self.woe_.tolist(), "missing_mode": self.missing_mode_,
            "missing_woe": self.missing_woe_, "missing_target_bin": self.missing_target_bin_,
            "is_integer": self.is_integer_, "iv": self.iv_,
        }

    @classmethod
    def from_dict(cls, d):
        b = cls(d["name"], trend=d["trend"])
        b.edges_ = np.array(d["edges"], dtype=float)
        b.woe_ = np.array(d["woe"], dtype=float)
        b.missing_mode_ = d["missing_mode"]
        b.missing_woe_ = d["missing_woe"]
        b.missing_target_bin_ = d["missing_target_bin"]
        b.is_integer_ = d["is_integer"]
        b.iv_ = d["iv"]
        return b


def iv_summary(binners: dict) -> pd.DataFrame:
    """One row per variable: IV, strength label and number of bins."""
    rows = [{"variable": n, "iv": b.iv_, "strength": iv_strength(b.iv_),
             "n_bins": len(b.table_), "missing_handling": b.missing_mode_}
            for n, b in binners.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
