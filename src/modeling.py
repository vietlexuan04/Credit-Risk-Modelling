"""
Logistic regression on WoE variables: fitting, diagnostics and variable selection.

Sign convention: target = 1 means BAD and WoE = ln(%good / %bad), so a higher WoE
means a SAFER bin. Every coefficient is therefore expected to be NEGATIVE
(a higher WoE lowers the log-odds of bad). A positive coefficient is a "wrong sign".
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

from evaluation import discrimination_metrics


def fit_logit(X: pd.DataFrame, y: pd.Series):
    """Plain (unweighted, unpenalised) logistic regression with an intercept."""
    return sm.Logit(y.astype(int), sm.add_constant(X, has_constant="add")).fit(disp=0)


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor of each column (intercept included in the design)."""
    Xc = sm.add_constant(X, has_constant="add")
    return pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                     index=X.columns, name="vif")


def coef_table(model, expected_sign: int = -1) -> pd.DataFrame:
    """Coefficients, standard errors, p-values and a sign check (intercept excluded from the check)."""
    t = pd.DataFrame({"coef": model.params, "std_err": model.bse, "z": model.tvalues, "p_value": model.pvalues})
    t["sign_ok"] = np.sign(t["coef"]) == expected_sign
    t.loc["const", "sign_ok"] = True
    return t


def backward_select(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05, max_vif: float = 5.0,
                    expected_sign: int = -1):
    """Backward elimination on WoE variables. One variable is removed per step, in this order:
       1. wrong coefficient sign (the one with the largest p-value),
       2. VIF above `max_vif` (the largest VIF),
       3. p-value above `alpha` (the largest p-value).
    Stops when no rule fires. Returns (kept_columns, final_model, log_dataframe).
    """
    cols, log = list(X.columns), []
    step = 0
    while True:
        model = fit_logit(X[cols], y)
        t = coef_table(model, expected_sign).drop(index="const")
        vif = vif_table(X[cols]) if len(cols) > 1 else pd.Series(1.0, index=cols)
        wrong = t[~t["sign_ok"]]
        if len(wrong):
            drop, reason, value = wrong["p_value"].idxmax(), "wrong sign", float(wrong["p_value"].max())
        elif vif.max() > max_vif:
            drop, reason, value = vif.idxmax(), f"VIF > {max_vif}", float(vif.max())
        elif t["p_value"].max() > alpha:
            drop, reason, value = t["p_value"].idxmax(), f"p-value > {alpha}", float(t["p_value"].max())
        else:
            break
        step += 1
        log.append({"step": step, "dropped": drop, "reason": reason, "value": value, "n_left": len(cols) - 1})
        cols.remove(drop)
    return cols, model, pd.DataFrame(log, columns=["step", "dropped", "reason", "value", "n_left"])


def cv_metrics(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified k-fold metrics on the development set (coefficients refitted in every fold)."""
    rows = []
    for k, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y), 1):
        m = fit_logit(X.iloc[tr], y.iloc[tr])
        p = m.predict(sm.add_constant(X.iloc[te], has_constant="add"))
        rows.append({"fold": k, **discrimination_metrics(y.iloc[te], p)})
    return pd.DataFrame(rows)
