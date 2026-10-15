"""
Data processing for the GMSC PD scorecard project.

Pipeline order (target-aware work happens on the development set only):
  1. Split RAW data once into development / hold-out (duplicate-aware, stratified).
  2. Explore and decide cleaning rules using the DEVELOPMENT set only.
  3. Apply the same deterministic clean() to both sets.

Cleaning philosophy (important for a scorecard built on WoE binning):
  * Rules are deterministic functions with FIXED thresholds. Nothing is fitted
    (no medians, no percentile caps), so applying the same clean() to
    development and hold-out leaks no information.
  * Thresholds are justified by domain plausibility or by target-free checks,
    and sanity-checked on the development set only (see notebook 03).
  * We do NOT impute. Missing / invalid values become NaN and get their own
    bin in the WoE step, so the model can learn their risk directly.
  * We do NOT drop rows unless the record is impossible (age < 18).
"""
import numpy as np
import pandas as pd

RENAME_MAP = {
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "n_30_59_late",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "n_open_lines",
    "NumberOfTimes90DaysLate": "n_90_late",
    "NumberRealEstateLoansOrLines": "n_real_estate",
    "NumberOfTime60-89DaysPastDueNotWorse": "n_60_89_late",
    "NumberOfDependents": "n_dependents",
}
LATE_COLS = ["n_30_59_late", "n_90_late", "n_60_89_late"]
COUNT_COLS = LATE_COLS + ["n_open_lines", "n_real_estate", "n_dependents"]

# ---- Cleaning thresholds (fixed constants; basis documented per constant) ----
SPECIAL_CODES = (96, 98)   # codes stored in the "times late" columns
MIN_VALID_AGE = 18         # basis: legal lending age; younger records are impossible
MIN_VALID_INCOME = 100     # basis (target-free): below ~100/month the median debt_ratio explodes
                           # (>10 in dev), i.e. debt_ratio is an absolute amount divided by a placeholder income
MAX_VALID_UTIL = 10        # basis: domain plausibility (balance above 10x the limit). Only 18 rows lie
                           # in (10, 100], so the exact cut inside that band has negligible effect.

# Groups of columns that are NaN on exactly the same rows after clean(), so their "Missing"
# WoE bins coincide. Verified in notebook 03. With one WoE column per variable this does NOT
# create perfect collinearity (the columns differ on all other rows; VIF < 1.4 in notebook 05),
# but the risk of the shared group is spread over several coefficients.
MISSING_GROUPS = {
    "income_invalid": ["monthly_income", "debt_ratio"],
    "late_special_code": LATE_COLS,
}

# ---- Split settings ----
SEED = 42
HOLDOUT_FRAC = 0.30


def load_raw(path: str) -> pd.DataFrame:
    """Read cs-training.csv and rename columns to short snake_case names."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = "id"
    return df.rename(columns=RENAME_MAP)


def split_dev_holdout(df: pd.DataFrame, holdout_frac: float = HOLDOUT_FRAC, seed: int = SEED):
    """Stratified, duplicate-aware split of RAW data into (development, hold-out).

    Fully identical rows (all columns incl. target) form one group and always
    land on the same side, so an exact copy of a development row can never
    appear in the hold-out set. The split is stratified on `target` at group
    level. Done ONCE with a fixed seed.
    """
    from sklearn.model_selection import train_test_split

    gid = df.groupby(list(df.columns), dropna=False).ngroup()
    groups = pd.DataFrame({"gid": gid, "target": df["target"]}).drop_duplicates("gid")
    _, g_hold = train_test_split(
        groups, test_size=holdout_frac, random_state=seed, stratify=groups["target"]
    )
    is_hold = gid.isin(g_hold["gid"])
    return df.loc[~is_hold].sort_index(), df.loc[is_hold].sort_index()


def clean(df: pd.DataFrame):
    """Apply the deterministic cleaning rules. Returns (clean_df, log_df).

    log_df records how many rows (and cells, where different) each rule
    touched, for the model development document.
    """
    out = df.copy()
    log = []

    def record(rule, n_rows, n_cells=None):
        log.append({
            "rule": rule,
            "rows_affected": int(n_rows),
            "cells_affected": int(n_rows if n_cells is None else n_cells),
            "rows_after": len(out),
        })

    # R1. Drop impossible ages.
    mask = out["age"] < MIN_VALID_AGE
    out = out.loc[~mask]
    record(f"R1 drop rows with age < {MIN_VALID_AGE}", mask.sum())

    # R2. Codes 96/98 in the delinquency-count columns are not real counts.
    #     Keep the information as a flag, set the counts to NaN (own bin later).
    special = out[LATE_COLS].isin(SPECIAL_CODES)
    out["late_special_code_flag"] = special.any(axis=1).astype(int)
    out[LATE_COLS] = out[LATE_COLS].mask(special)
    record("R2 recode 96/98 in late-count columns to NaN + flag",
           out["late_special_code_flag"].sum(), special.sum().sum())

    # R3. Zero or placeholder income (< MIN_VALID_INCOME) is treated as "not reported".
    mask = out["monthly_income"] < MIN_VALID_INCOME
    out.loc[mask, "monthly_income"] = np.nan
    record(f"R3 monthly_income < {MIN_VALID_INCOME} (zero / placeholder) -> NaN", mask.sum())

    # R4. When income is missing or invalid, debt_ratio holds an absolute amount,
    #     not a ratio (see notebook 03 evidence). Not comparable, so set to NaN.
    mask = out["monthly_income"].isna() & out["debt_ratio"].notna()
    out.loc[mask, "debt_ratio"] = np.nan
    record("R4 debt_ratio -> NaN where monthly_income is NaN", mask.sum())

    # R5. Implausible utilization values become NaN (own bin later).
    mask = out["revolving_util"] > MAX_VALID_UTIL
    out.loc[mask, "revolving_util"] = np.nan
    record(f"R5 revolving_util > {MAX_VALID_UTIL} -> NaN", mask.sum())

    # R6. Fully duplicated rows are KEPT (documented, not removed). They are not
    #     random: concentrated in the income-missing, low-risk segment.
    record("R6 duplicated rows: kept (no action)", 0)

    return out, pd.DataFrame(log)


def validate_clean(df: pd.DataFrame) -> None:
    """Sanity assertions on the cleaned data. Raises AssertionError on failure."""
    assert df.index.is_unique, "index (id) must be unique"
    assert set(df["target"].unique()) <= {0, 1}, "target must be binary"
    assert (df["age"] >= MIN_VALID_AGE).all(), "age below minimum remains"
    assert not df[LATE_COLS].isin(SPECIAL_CODES).any().any(), "96/98 codes remain"
    assert not (df[COUNT_COLS] < 0).any().any(), "negative counts present"

    # Flag <=> late counts are NaN (all three columns together)
    flag = df["late_special_code_flag"] == 1
    assert (df[LATE_COLS].isna().all(axis=1) == flag).all(), "flag and late NaNs disagree"
    assert (df[LATE_COLS].isna().any(axis=1) == flag).all(), "late NaNs not all-or-nothing"

    # Income / debt_ratio consistency
    assert not (df["monthly_income"] < MIN_VALID_INCOME).any(), "tiny/zero income remains"
    assert (df["monthly_income"].isna() == df["debt_ratio"].isna()).all(), \
        "income and debt_ratio missing patterns differ"
    assert (df["revolving_util"].dropna() <= MAX_VALID_UTIL).all(), "extreme utilization remains"
