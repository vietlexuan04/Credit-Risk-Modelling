# Model Validation Report

**Model:** GMSC application scorecard (WoE + logistic regression), see [`model_development.md`](model_development.md)
**Validation data:** hold-out set, 45,012 customers, 3,006 bad (6.68%)
**Notebook:** `notebooks/07_model_validation.ipynb`

> **Independence caveat.** The same person developed and validated this model. A real validation is performed by an independent team.
> This report applies the checks such a team would run, but it is a self-assessment and should be read as one.

---

## 1. Scope and approach

The hold-out set was **not used** for binning, variable selection, coefficient estimation or scaling. Binning (`binning_spec.json`), coefficients (`model_spec.json`)
and scaling (`scorecard_spec.json`) were frozen before it was opened. Grade cut points come from the **development** score.

Checks performed: discrimination, rank ordering, calibration (overall and by grade), stability (PSI/CSI), and a cut-off analysis.
Conventions: higher score = safer; metrics that expect "risk" receive the negative score or the PD.

## 2. Integrity of the hold-out set (read this first)

- Split once, stratified, seed 42, duplicate-aware (no identical row on both sides, verified by code).
- The hold-out was opened for the first time in notebook 07.
- **The calibration remedies in section 4.3 were chosen after seeing the hold-out calibration.** Both candidate remedies were *fitted* on development data only, but the decision to present the grade master scale was made after observing hold-out results.
  The hold-out result for that remedy (Hosmer-Lemeshow p about 0.27) is therefore **not a clean out-of-sample confirmation**. A fresh sample, or an out-of-time sample, would be needed for that.
- A quick preview of the hold-out metrics was computed before the notebook was written. The bins, variables, coefficients and scaling were **not** changed as a result; the only consequence was adding the calibration remedies of section 4.3.

## 3. Discrimination

| | AUC | Gini | KS |
|---|---|---|---|
| Development | 0.8585 | 0.7169 | 0.5537 |
| **Hold-out** | **0.8596** | **0.7192** | **0.5650** |
| Hold-out 95% bootstrap CI (500 resamples) | 0.8523 - 0.8666 | 0.7046 - 0.7332 | |
| 5-fold CV on development (mean, std) | 0.8582 (0.0076) | 0.7164 (0.0151) | 0.5577 (0.0135) |

- Hold-out AUC is slightly *higher* than on development, but the difference is far inside the bootstrap interval. There is **no sign of overfitting**.
- A hold-out AUC around 0.86 sits in the range expected for this kind of data. A value near 0.95 would have pointed to leakage; it did not occur.
- Gini = 2 x AUC - 1 carries the same information on another scale. Whether Gini 0.72 / KS 0.56 is "good" depends on product and data; the figures are a sanity check, not a pass mark.
- The KS gap, recomputed by hand from the two cumulative score curves, matches the library value (figure `outputs/figures/roc_and_ks_holdout.png`).

## 4. Calibration

Discrimination only needs the *order* to be right. Calibration asks whether the *level* of the PD is right.

### 4.1 Overall calibration: good

| | Mean predicted PD | Observed bad rate | Brier | Log loss |
|---|---|---|---|---|
| Development | 6.54% | 6.69% | 0.0502 | 0.1828 |
| Hold-out | 6.64% | 6.68% | 0.0502 | 0.1827 |

**Which PD?** All PDs in this report are those implied by the **rounded** scorecard score, as a bank would use them. Rounding raises scores by 1.0 point on average, so the development mean PD is 6.54% rather than exactly the 6.69% bad rate reproduced by the unrounded model (development report, sections 6.4 and 7).

Calibration regression (outcome on the logit of the PD): intercept -0.025, slope 0.984 on the hold-out (intercept 0.018, slope 0.994 on development). Intercept near 0 and slope near 1 mean the PD level is right on average and its spread is right, without class weights or resampling.

### 4.2 Calibration by grade: not perfect (the honest finding)

Ten grades defined on development score quantiles. Hold-out:

| Grade | Rows | Bad | Predicted PD | Observed bad rate | Observed / predicted |
|---|---|---|---|---|---|
| 1 (safest) | 4,735 | 28 | 0.88% | 0.59% | 0.67 |
| 2 | 4,556 | 33 | 1.17% | 0.72% | 0.62 |
| 3 | 4,639 | 44 | 1.47% | 0.95% | 0.65 |
| 4 | 4,000 | 52 | 1.79% | 1.30% | 0.72 |
| 5 | 4,940 | 77 | 2.28% | 1.56% | 0.68 |
| 6 | 4,351 | 122 | 3.06% | 2.80% | 0.92 |
| 7 | 4,489 | 198 | 4.29% | 4.41% | 1.03 |
| 8 | 4,407 | 292 | 6.31% | 6.63% | 1.05 |
| 9 | 4,361 | 514 | 10.07% | 11.79% | 1.17 |
| 10 (riskiest) | 4,534 | 1,646 | 35.26% | 36.30% | 1.03 |

- The safest five grades are **over-predicted** by roughly a factor of 1.4 to 1.6 (observed / predicted about 0.6 to 0.7). Grade 9 is mildly under-predicted (1.17).
- The Hosmer-Lemeshow test (10 groups, 10 degrees of freedom because neither the groups nor the PDs were fitted on the hold-out) rejects strongly (statistic 56.5, p far below 0.001). With 45,000 rows this test is very sensitive and flags even modest deviations; because the pattern is consistent across grades it is not dismissed.
- The pattern is not caused by rounding: with the unrounded model PD the safest grades are over-predicted slightly *more* (observed / predicted 0.60 to 0.70).
- The **same shape appears on the development set** (grade 1: predicted 0.88%, observed 0.48%; grade 2: 1.17% versus 0.58%). It is therefore a **structural limitation of the model form**
  (a logistic regression on a few coarse WoE bins cannot bend enough in the low-risk tail), not noise in the hold-out sample.
- Direction: **conservative** (it overstates the risk of the best customers). That matters little for approve/decline decisions, which depend on ranking, but it does matter for **pricing and provisioning**, where the level of the PD is used directly.
- The safest grades have only 28 to 77 bad cases each in the hold-out, so their observed rates are inherently noisy.

### 4.3 Remedies tested (fitted on development only)

| PD used | Brier | Log loss | Mean PD | Hosmer-Lemeshow p |
|---|---|---|---|---|
| Original model PD | 0.05024 | 0.18269 | 6.64% | < 0.001 |
| Linear recalibration (logistic of outcome on logit of PD) | 0.05021 | 0.18270 | 6.79% | < 0.001 |
| **Grade master scale** (development bad rate per grade) | 0.05141 | 0.18419 | 6.78% | 0.27 |

- **Linear recalibration does not help.** The problem is the *shape*, not a global shift, which is exactly what a slope close to 1 already indicated.
- **The grade master scale repairs grade-level calibration** (ratios mostly close to 1; only grade 5 at 0.77 remains individually significant), at a small cost in Brier score because every customer in a grade shares one PD.
  Master scale values are in `outputs/scorecard/grade_master_scale.csv`.
- Practical conclusion: use the **score for ranking and decisions** and report **PD per grade from the master scale** for provisioning and pricing. See section 2 for the caveat on how this conclusion was reached.

## 5. Rank ordering

Bad rate by grade rises **strictly** from grade 1 to grade 10 on both sets, with no inversion (asserted in code).

| Grade | Score range | Dev bad rate | Hold-out bad rate |
|---|---|---|---|
| 1 | 620-632 | 0.48% | 0.59% |
| 2 | 612-619 | 0.58% | 0.72% |
| 3 | 606-611 | 1.03% | 0.95% |
| 4 | 600-605 | 1.20% | 1.30% |
| 5 | 592-599 | 2.02% | 1.56% |
| 6 | 583-591 | 2.79% | 2.80% |
| 7 | 572-582 | 4.82% | 4.41% |
| 8 | 559-571 | 7.14% | 6.63% |
| 9 | 540-558 | 11.79% | 11.79% |
| 10 | 390-539 | 36.16% | 36.30% |

The riskiest grade is about 61 times (hold-out) to 75 times (development) as risky as the safest. Grade 10 is wide because the score distribution has a long left tail, and it contains more than half of all bad cases (53% on development, 55% on hold-out).

## 6. Stability

| Measure (development versus hold-out) | Value | Reading |
|---|---|---|
| Score PSI (10 grades) | 0.0005 | stable (< 0.1) |
| Largest variable CSI (`age`) | 0.0003 | stable |
| All nine variable CSIs | between 0.0001 and 0.0003 | stable |

Both sets come from one random split, so near-zero values are expected. They show that the split is clean; they say **nothing about stability over time**, which this dataset cannot test (no dates).

## 7. Cut-off analysis (hold-out)

Approve if score >= cut-off. No-scorecard bad rate is 6.68%.

| Cut-off | Approval rate | Bad rate, approved | Bad rate, rejected | Share of all bad rejected | Share of all good rejected |
|---|---|---|---|---|---|
| 500 | 96.8% | 5.14% | 53.7% | 25.5% | 1.6% |
| 520 | 94.3% | 4.28% | 46.0% | 39.6% | 3.3% |
| 540 | 89.9% | 3.36% | 36.3% | 54.8% | 6.9% |
| 550 | 86.2% | 2.94% | 30.1% | 62.0% | 10.3% |
| 560 | 79.5% | 2.31% | 23.6% | 72.5% | 16.8% |
| 570 | 72.0% | 1.83% | 19.1% | 80.3% | 24.3% |
| 580 | 62.9% | 1.42% | 15.6% | 86.6% | 33.5% |
| 590 | 52.6% | 1.06% | 12.9% | 91.7% | 44.2% |
| 600 | 39.8% | 0.88% | 10.5% | 94.8% | 57.7% |

Example reading: at 560 the scorecard approves about 80% of applicants at a 2.3% bad rate (versus 6.68% without it) while rejecting about 72% of the bad customers and 17% of the good ones.
**Choosing the cut-off is a business decision** (margin per good account, loss per bad account, risk appetite, capacity); the table shows the trade-off and does not choose. One row was re-computed by hand and matches.

## 8. Proposed monitoring plan (hypothetical; nothing below has been run)

There is no post-development data, so this is a design, not a result.

| What | How | Frequency | Alert (rule of thumb) |
|---|---|---|---|
| Score distribution shift | PSI on the 10 development grades versus recent applications | monthly | 0.1 amber, 0.25 red |
| Input drift | CSI per variable, same bins | monthly | 0.1 amber, 0.25 red |
| Discrimination | Gini / KS on matured accounts (90+ days past due within 24 months needs a 24-month wait) | quarterly, after outcomes mature | Gini drop of more than 5 points versus development |
| Calibration | observed bad rate versus master-scale PD by grade, binomial test | quarterly | observed outside the 95% interval for two periods running |
| Rank ordering | bad rate by grade must stay monotonic | quarterly | any inversion |
| Data quality | share of `NaN`, special codes 96/98, implausible values | monthly | share moves by more than 2 points |
| Override / policy | approval rate and cut-off use | monthly | unexplained shifts |

A recalibration or rebuild is triggered by a red alert on calibration or Gini, or by a policy change that alters who is approved.
Because outcomes need 24 months to mature, early warning relies on PSI/CSI and on shorter-horizon delinquency indicators.

## 9. Conclusions

| Area | Result | Status |
|---|---|---|
| Discrimination | AUC 0.860, Gini 0.719, KS 0.565, no overfitting | **Pass** |
| Rank ordering | strictly monotonic on both sets | **Pass** |
| Overall calibration | mean PD 6.64% versus 6.68%, slope 0.98 | **Pass** |
| Grade-level calibration | safest five grades over-predicted; HL rejects; structural | **Weakness**, mitigated by master scale (see caveat, section 2) |
| Stability (split) | PSI 0.0005 | Pass, but not informative about time |
| Out-of-time performance | not testable | **Not assessed** |

**Recommended use (if this were a real model):** ranking and approve/decline using the score; PD by grade from the master scale for provisioning and pricing; no use of the continuous PD for pricing the safest grades without recalibration.
Remaining weaknesses are listed in [`limitations.md`](limitations.md).
