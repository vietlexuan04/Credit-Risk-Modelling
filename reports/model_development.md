# Model Development Document

**Model:** Application scorecard for the probability of default (PD), WoE + logistic regression
**Data:** Give Me Some Credit (GMSC), `cs-training.csv`
**Status:** portfolio / learning project. Not a production model and not approved for any real credit decision.

This document describes *what was built and why*. Independent-style testing results are in
[`model_validation.md`](model_validation.md); known weaknesses are collected in [`limitations.md`](limitations.md).
Every number below is produced by the notebooks in `notebooks/` and can be re-created by running them in order.

---

## 1. Purpose and scope

- **Purpose:** rank retail loan applicants by the likelihood of serious delinquency and give each a PD, using only information available at application time.
- **Type:** application scorecard (not a behavioural scorecard).
- **Output:** an integer score (higher = safer) built from additive points per bin, and the PD implied by that score.
- **Out of scope:** LGD, EAD, expected loss, IFRS 9 staging, reject inference, pricing, fairness analysis.

## 2. Data

| Item | Value |
|---|---|
| Source | Kaggle competition "Give Me Some Credit", file `cs-training.csv` |
| Rows / predictors | 150,000 borrowers, 10 predictors |
| `cs-test.csv` | not used (its target column is empty) |
| Time information | none (no application or observation date) |
| Population | borrowers whose loans were granted (see limitations: no reject inference) |

### 2.1 Target definition

`SeriousDlqin2yrs = 1` ("bad") means the borrower became **90 days or more past due within 2 years** of the observation date; `0` ("good") otherwise.
Performance window: 24 months. The exact observation date is not in the data.

The 90+ days past due definition is the common default convention for retail credit. Its relation to Vietnamese regulatory loan-classification groups
should be checked against the current State Bank of Vietnam circular before being stated in any application; it is *not* asserted here.

| | Rows | Bad | Bad rate |
|---|---|---|---|
| Full data | 150,000 | 10,026 | 6.68% |

Odds good:bad is about 14:1. The class imbalance is moderate and there are ten thousand bad cases, enough to estimate bins. **No SMOTE and no class weights are used**, because both distort predicted probabilities and the model output must remain a real PD.

### 2.2 Predictors (renamed to short snake_case in `src/data_processing.py`)

| Name | Original column | Meaning |
|---|---|---|
| `revolving_util` | RevolvingUtilizationOfUnsecuredLines | balance on revolving lines divided by credit limits |
| `age` | age | age in years |
| `n_30_59_late` | NumberOfTime30-59DaysPastDueNotWorse | times 30-59 days past due |
| `n_60_89_late` | NumberOfTime60-89DaysPastDueNotWorse | times 60-89 days past due |
| `n_90_late` | NumberOfTimes90DaysLate | times 90+ days past due |
| `debt_ratio` | DebtRatio | monthly debt payments divided by monthly income |
| `monthly_income` | MonthlyIncome | monthly income (currency and units not documented) |
| `n_open_lines` | NumberOfOpenCreditLinesAndLoans | open loans and credit lines |
| `n_real_estate` | NumberRealEstateLoansOrLines | real-estate loans or lines |
| `n_dependents` | NumberOfDependents | number of dependents |

## 3. Development process (and the order of operations)

```
raw data -> (01) understand, target-free -> (02) split 70/30 -> (03) cleaning rules, evidence on DEV only
         -> (04) WoE binning on DEV -> (05) variable selection + logit on DEV -> (06) scorecard points
         -> (07) validation on HOLD-OUT (first time it is opened)
```

The split comes **before** any analysis that looks at the target. The same deterministic `clean()` function is then applied to both sets.
This order was adopted after a review of an earlier draft in which cleaning thresholds were explored on the full data; see section 4.2 for what that review changed.

### 3.1 Development / hold-out split (notebook 02)

- 70 / 30, **stratified on target**, fixed seed 42, done once.
- **Duplicate-aware:** fully identical rows (including target) are kept together on the same side, so an exact copy of a development row can never sit in the hold-out set.

| Set | Rows (raw) | Bad | Bad rate |
|---|---|---|---|
| Development | 104,987 | 7,020 | 6.69% |
| Hold-out | 45,013 | 3,006 | 6.68% |

(The hold-out has 45,012 rows after cleaning because one impossible record is dropped; see rule R1.)

## 4. Data quality and cleaning (notebook 03)

### 4.1 Principles

- Rules are functions with **fixed thresholds**; nothing is fitted (no medians, no percentile caps), so applying them to the hold-out set leaks nothing.
- **No imputation.** Invalid or missing values become `NaN` and receive their own bin in the WoE step, so the model learns their actual risk.
- Rows are dropped only when the record is impossible.
- Evidence for every rule is computed on the development set only (notebook 03, part A).

### 4.2 Rules

| Rule | Action | Basis | Dev rows | Hold-out rows |
|---|---|---|---|---|
| R1 | Drop age < 18 | impossible record (one row of age 0 in the whole file) | 0 | 1 |
| R2 | 96/98 in the three late-count columns -> `NaN`, plus `late_special_code_flag` | special codes, not real counts; the same rows carry them in all three columns and show a very different bad rate | 170 (510 cells) | 99 (297 cells) |
| R3 | `monthly_income` < 100 -> `NaN` | zero and placeholder incomes (e.g. 1); below about 100 per month the "debt ratio" is an absolute amount divided by a placeholder. Threshold chosen from the behaviour of `debt_ratio`, **not** from the bad rate | 1,576 | 701 |
| R4 | `debt_ratio` -> `NaN` where income is `NaN` | when income is missing the column holds an absolute amount, not a ratio (median in the hundreds or thousands versus about 0.3 when income is valid) | 22,424 | 9,584 |
| R5 | `revolving_util` > 10 -> `NaN` | domain plausibility (balance above ten times the limit) | 175 | 66 |
| R6 | Fully duplicated rows are **kept** | about 0.4% of rows; not random (concentrated in the income-missing, low-risk segment), so documented rather than removed | 0 | 0 |

**What an external review changed.** An earlier draft (a) pooled zero-income rows into the "income present" evidence, (b) only treated income = 0 as invalid, (c) described duplicates as "mostly thin-file" on weak evidence,
and (d) chose the utilization threshold after looking at bad rates on the *full* data including what became the hold-out. All four were confirmed against the data and corrected:
the evidence tables now exclude invalid incomes, R3 covers placeholder incomes, the duplicate description now matches the numbers (duplicates have a bad rate of about 2.8% on the full file and 3.4% on the development set, versus 6.7% overall, and 84% of them are income-missing),
the split was moved ahead of cleaning, and the R5 threshold is now justified on domain grounds. **Residual disclosure:** the value 10 was first picked after viewing full-data bad rates;
only a handful of rows (13 to 18) fall between 10 and 100, so the exact cut has negligible effect, but the original choice was not independent of the target.

### 4.3 Test of keeping the absolute debt amount (rejected)

Instead of discarding `debt_ratio` where income is missing, its value could have been kept as a separate variable. On the development set the bad rate across eight quantile groups of that amount is not monotonic and the Spearman correlation with the target is about -0.012, so it carries no usable signal and no extra variable was created (notebook 03, section A8).

### 4.4 Missing values after cleaning (development set)

| Variable | `NaN` rows | Share | Bad rate of the `NaN` group |
|---|---|---|---|
| `monthly_income`, `debt_ratio` (identical rows) | 22,424 | 21.4% | 5.3% |
| `n_dependents` | 2,723 | 2.6% | 4.6% |
| `revolving_util` | 175 | 0.2% | about 5% (9 bad cases) |
| late-count columns (identical rows = special codes) | 170 | 0.2% | 58.8% |

Some columns are `NaN` on exactly the same rows, so their "Missing" WoE bins coincide. With one WoE column per variable this does **not** make the columns collinear (they differ on all other rows; VIF < 1.4, section 6.2), but the risk of a shared group is spread over several coefficients. `monthly_income` was dropped for its own reasons (section 6.2); the three late-count variables all stay, so the 96/98 group's points are the sum of their three Missing bins (21 + 30 + 16 = 67 points, versus 205 for a customer with no delinquency).

## 5. Binning, WoE and Information Value (notebook 04)

### 5.1 Definitions

- **WoE of a bin** = ln(%good in bin / %bad in bin), percentages measured against totals of the whole development set. Negative = riskier than average.
- **IV of a variable** = sum over bins of (%good - %bad) x WoE. Conventional reading: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 suspicious (check for leakage).

### 5.2 Procedure (`src/feature_engineering.py`, class `WoEBinner`, unit tested)

1. **Fine classing:** each distinct value (if at most 30 distinct) or about 40 quantile bins.
2. **Coarse classing:** merge **adjacent** bins until all of the following hold:
   - WoE is **monotonic** in the direction given by business logic (or `free` where a U-shape is meaningful);
   - every numeric bin has at least a minimum share of the sample (1% to 5%, per variable) and **at least 50 bad cases**;
   - adjacent bins differ by at least 0.05 in WoE;
   - at most 8 bins; when more remain, the merge that loses the least IV goes first.
3. **Missing values:** own "Missing" bin if the group has at least 30 bad cases (no minimum share), otherwise merged into the numeric bin with the closest WoE (only `revolving_util`: its 175 `NaN` rows join the bin labelled `(0.3065, 0.4989] or Missing`).
4. A zero count in a bin adds 0.5 to good and bad before computing WoE.

### 5.3 Business direction per variable

Decided from credit logic, then checked against the data:

| Variable | Trend | Rationale |
|---|---|---|
| `revolving_util` | increasing risk | higher utilization means more financial stress |
| `age` | decreasing risk | older borrowers historically more stable |
| `n_30_59_late`, `n_60_89_late`, `n_90_late` | increasing risk | more delinquency events means higher risk |
| `debt_ratio` | increasing risk | higher burden means higher risk |
| `monthly_income` | decreasing risk | more capacity to repay |
| `n_open_lines` | decreasing risk | very few lines means a thin credit history |
| `n_real_estate` | **free** | a U-shape is plausible (no property versus over-leveraged); the shape seen in development (8.3% bad at 0, 5.2% at 1, 11.4% at 4+) supports it. This choice was made after seeing the shape and is disclosed as such |
| `n_dependents` | increasing risk | more dependents means more burden |

### 5.4 Information Value (development set)

| Variable | IV | Reading | Bins |
|---|---|---|---|
| `revolving_util` | 1.157 | very strong | 8 |
| `n_90_late` | 0.884 | very strong | 4 |
| `n_30_59_late` | 0.736 | very strong | 5 |
| `n_60_89_late` | 0.590 | very strong | 4 |
| `age` | 0.270 | medium | 8 |
| `n_open_lines` | 0.099 | weak | 5 |
| `monthly_income` | 0.082 | weak | 8 |
| `debt_ratio` | 0.080 | weak | 6 |
| `n_real_estate` | 0.062 | weak | 5 |
| `n_dependents` | 0.039 | weak | 6 |

Four variables are above the 0.5 "suspicious" line. This alone is not evidence of leakage: they are credit-bureau behaviour variables, the target is "90+ days past due in two years", and they are known to be highly predictive on this dataset.
But the dataset has **no dates**, so the timing of each field relative to the observation date cannot be verified. The practical safeguard is the hold-out AUC: about 0.86 is plausible, a value near 0.95 would have indicated a problem (it did not occur).

Checks performed: one bin recomputed by hand from raw counts (`n_30_59_late == 1`: WoE -0.8899, IV contribution 0.1244) matches the function; WoE re-estimated on two random halves of the development set differs by less than 0.25 for most variables
(0.28 for the three late-count variables because their shared Missing bin has only 170 rows); `monthly_income` is the only variable whose monotonic order is not preserved in both halves (two middle bins have almost equal WoE).

Full bin tables: `outputs/scorecard/woe_bin_tables.csv`; saved binning specification: `outputs/scorecard/binning_spec.json`; profiles: `outputs/figures/woe_profiles.png`.

## 6. Variable selection and logistic regression (notebook 05)

### 6.1 Model form

Unweighted, unpenalised logistic regression (`statsmodels`) of the target on WoE variables:

`ln(PD / (1 - PD)) = b0 + sum_i b_i x WoE_i`

**Sign convention:** target 1 is bad and a higher WoE means a *safer* bin, so every coefficient is expected to be **negative**. A positive coefficient is a wrong sign.

### 6.2 Selection layers

1. **IV filter (>= 0.02):** all ten variables pass.
2. **Correlation of WoE variables:** no pair above 0.5 in absolute value; the largest is `n_60_89_late` / `n_90_late` at 0.345.
3. **Full model:** nine of ten coefficients negative and significant at 5% (all except `n_open_lines` with p < 0.003); VIF below 1.4 everywhere.
4. **Backward elimination** (wrong sign first, then VIF > 5, then p-value > 0.05): one step only, removing `monthly_income` (coefficient +0.02, p = 0.69).
   Its information is largely carried by `debt_ratio` (WoE correlation 0.30) and `age`, and its Missing group is identical to that of `debt_ratio`.
5. **Business judgment:** `n_open_lines` is the weakest survivor (p = 0.027) and is kept because it is significant, has the expected sign and a clear meaning (thin credit history).

### 6.3 Final model (development set, 9 variables)

| Variable | Coefficient | Std. error | p-value | VIF |
|---|---|---|---|---|
| intercept | -2.6058 | 0.0151 | < 0.001 | |
| `revolving_util` | -0.5986 | 0.0146 | < 0.001 | 1.34 |
| `age` | -0.4135 | 0.0311 | < 0.001 | 1.17 |
| `n_30_59_late` | -0.4847 | 0.0161 | < 0.001 | 1.22 |
| `n_60_89_late` | -0.3709 | 0.0172 | < 0.001 | 1.22 |
| `n_90_late` | -0.5341 | 0.0138 | < 0.001 | 1.26 |
| `debt_ratio` | -0.9079 | 0.0512 | < 0.001 | 1.17 |
| `n_open_lines` | -0.1048 | 0.0474 | 0.027 | 1.27 |
| `n_real_estate` | -0.6708 | 0.0604 | < 0.001 | 1.14 |
| `n_dependents` | -0.2279 | 0.0730 | 0.002 | 1.13 |

Coefficients lie between -0.1 and -0.9 rather than near -1: a single WoE variable would have a coefficient near -1 (WoE is already a log-odds ratio); with partially overlapping variables each coefficient shrinks. What matters is that none has a wrong sign.

### 6.4 Development performance

| | AUC | Gini | KS |
|---|---|---|---|
| In-sample | 0.858 | 0.717 | 0.554 |
| 5-fold cross-validation (mean, std) | 0.858 (0.008) | 0.716 (0.015) | 0.558 (0.014) |

Mean predicted PD of the (unrounded) model equals the development bad rate (6.69%), as it must for a logistic regression with an intercept. The PD implied by the *rounded* scorecard is slightly lower (6.54%, see section 7); the validation report uses the rounded scorecard. Cross-validation re-fits coefficients per fold but the **bins are not re-learned per fold**, so the cross-validated figures are very slightly optimistic; the hold-out is the real test.

## 7. Scorecard scaling (notebook 06)

**PDO method.** `Score = Offset + Factor x ln(odds good:bad)`, `Factor = PDO / ln 2`, `Offset = BaseScore - Factor x ln(BaseOdds)`.

| Parameter | Value |
|---|---|
| Base score | 600 |
| Base odds (good:bad) | 50:1 (PD = 1/51 = 1.96% at 600) |
| PDO | 20 points (every +20 doubles the odds of good) |
| Factor / Offset | 28.854 / 487.123 |

Because the model gives log-odds of *bad* `z`, `ln(odds good:bad) = -z` and

`Score = sum_i [ -Factor x b_i x WoE_i + (Offset - Factor x b0) / n ]`, with `n = 9`.

Each bin therefore has fixed points; the intercept part (62.48 points) is spread equally over the nine variables, and points are rounded to integers.
Rounding changes a customer's score by about 1 point on average (at most 3.3 on the development set), small against a 20-point PDO. It is not symmetric: the rounded score is on average **1.0 point higher** than the exact score (the large bins that hold most customers happen to round up), which raises the odds of good by about 3.5% and lowers the mean score-implied PD on development from 6.69% to 6.54%.
The PD implied by a score is `1 / (1 + exp((Score - Offset) / Factor))`; scores of 540, 600 and 620 correspond to PDs of about 13.8%, 2.0% and 1.0%.

The largest point ranges (max minus min over the bins of a variable): `revolving_util` 62, `n_90_late` 53, `n_30_59_late` 49, `n_60_89_late` 36, `debt_ratio` 24, `age` 21, `n_real_estate` 16, `n_dependents` 6, `n_open_lines` 4.
Development scores range from 390 to 632 (median 592); the theoretical range is 363 to 634.

### 7.1 Worked example (development customer, hand-checked in notebook 06)

| Variable | Value | Points |
|---|---|---|
| `revolving_util` | 0.957 | 42 |
| `age` | 40 | 58 |
| `n_30_59_late` | 0 | 70 |
| `n_60_89_late` | 0 | 66 |
| `n_90_late` | 0 | 69 |
| `debt_ratio` | 0.122 | 65 |
| `n_open_lines` | 4 | 63 |
| `n_real_estate` | 0 | 58 |
| `n_dependents` | 1 | 62 |
| **Score** | | **553** |

A score of 553 implies a PD of about 9.3%. (With unrounded points the exact score is 551.3 and the model PD is 9.77%; the gap of about 0.5 percentage points comes from integer rounding.)
The full points table is in `outputs/scorecard/scorecard_points.csv`.

## 8. Grades and master scale

Ten grades are defined with **score quantiles of the development set** (cut points 540, 559, 572, 583, 592, 600, 606, 612, 620); grade 1 is the safest.
`outputs/scorecard/grade_master_scale.csv` stores, per grade, the development bad rate (0.48% for grade 1 up to 36.2% for grade 10), which is used as the grade PD where grade-level calibration matters (see validation report, section 4).

## 9. Reproducibility

- Seed 42 for the split, cross-validation and bootstrap.
- Run notebooks `01` to `07` in order from the `notebooks/` folder; they read and write relative paths (`../data`, `../src`, `../outputs`).
- Core logic lives in `src/` (`data_processing.py`, `feature_engineering.py`, `modeling.py`, `scorecard.py`, `evaluation.py`) with 14 unit tests in `tests/` (`pytest tests`).
- Environment used: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, statsmodels 0.15.0, scipy 1.18.1 (pinned in `requirements.txt`).
- Data is not committed (`data/README.md` explains how to obtain it).
