# Limitations and Known Issues

This is a learning and portfolio project on a public dataset. The points below describe what it can and cannot support. They are ordered roughly by how much they matter.

## 1. Data limitations

1. **No time dimension, so no out-of-time validation.** The dataset has no application or observation date. Validation is out-of-sample (random split) only.
   A random split mixes time and typically gives a more favourable result than a later period would. Performance under changing economic or policy conditions is unknown. *Planned remedy:* Phase 2 on a dataset with dates (e.g. Lending Club) with out-of-time validation.
2. **No reject inference.** The data only contains borrowers whose loans were approved and whose outcome was observed. The scorecard learns nothing about applicants the original lender declined, so it is biased towards the approved population.
   Its performance on the full applicant pool is unknown, and the PD describes *approved* borrowers, not the through-the-door population. No reject-inference method was applied.
3. **Dated, non-local data.** US consumer-credit data from a public Kaggle competition (around 2011). Collection period, product, currency and units of `monthly_income` are not documented. Nothing here describes Vietnamese borrowers, products or credit bureau behaviour.
4. **Field timing cannot be verified.** Four variables have IV above 0.5 (`revolving_util`, `n_90_late`, `n_30_59_late`, `n_60_89_late`). They are plausible bureau-behaviour variables, but without dates it is impossible to confirm that every field was recorded *before* the observation date.
   The hold-out AUC of about 0.86 gives comfort but is not proof.
5. **Encoded and corrupted values.** 96/98 codes, placeholder incomes and absolute amounts stored in `debt_ratio` show the file is not a clean extract. The cleaning rules are judgement calls grounded in evidence (see `model_development.md`, section 4) rather than certainties.
6. **Duplicates kept.** About 0.4% of rows are exact duplicates, not random (mostly income-missing, low-risk). They stay in the data; the split keeps identical rows on the same side so they cannot inflate hold-out results.

## 2. Methodology limitations

1. **Some choices were made after looking at the data.**
   - The utilization threshold (> 10) was first chosen after viewing bad rates on the full data, including what later became the hold-out. It is now justified on domain grounds and only 13 to 18 rows lie in the band where the exact cut matters, but the original choice was not target-independent.
   - The `free` (non-monotonic) trend for `n_real_estate` was chosen after seeing a U-shape in the development set.
   - The grade master scale was presented after seeing the hold-out calibration (see `model_validation.md`, section 2), so the hold-out result for that remedy is not a clean test.
2. **Cross-validation is slightly optimistic** because the bins are learned once on the whole development set and not re-learned inside each fold.
3. **Binning rules are my own implementation** (`WoEBinner`, unit tested) rather than a reference library. Settings such as minimum bin share (1% to 5%), minimum 50 bad per bin, a WoE gap of 0.05 and at most 8 bins are reasonable but not uniquely correct.
4. **Missing-value treatment:** `monthly_income` and `debt_ratio` share an identical Missing group and the three late-count variables share another. This does not make the WoE columns collinear (VIF < 1.4), but the risk of a shared group is spread over several coefficients and cannot be attributed to one variable. `monthly_income` was removed for a wrong sign and a p-value of 0.69.
5. **No stability of the bins over time** and no sensitivity tests (different seeds, different bin settings, removal of duplicates). Only a single seed (42) was used for the final results.
6. **Self-validation.** Development and validation were done by the same person.

## 3. Model performance limitations

1. **Grade-level calibration is imperfect.** The safest five grades have their PD over-stated by about 1.4 to 1.6 times; Hosmer-Lemeshow rejects. This is structural (also visible on development). A grade master scale mitigates it but at the cost of one PD per grade.
2. **Sparse safest grades.** Only 28 to 77 bad cases per safest grade in the hold-out, so observed rates there are noisy.
3. **Only one model family at this stage.** No challenger (e.g. gradient boosting) has been compared yet, so the cost of the scorecard's simplicity is unquantified.
4. **Score is a ranking tool.** The 600 / 50:1 / PDO 20 scaling is a presentation convention, not a calibration target.
5. **Rounding of points** raises scores by about 1 point on average, so the score-implied PD is about 3.5% (relative) lower than the unrounded model PD. Small, disclosed, but it means score-implied PDs are slightly optimistic.

## 4. Regulatory, fairness and governance caveats

1. **Use of `age` as a predictor.** Age carries substantial predictive weight (IV 0.27). In many jurisdictions the use of age in consumer credit decisions is restricted or must meet specific conditions. No legal or fairness review was done and none of the standard fairness tests was run.
2. **Default definition and regulation.** The 90+ days past due definition is a common convention. Its relation to Vietnamese loan-classification rules and to any regulatory definition of default was **not** verified and must be checked against the current State Bank of Vietnam circular before being claimed anywhere.
3. **No governance artefacts** beyond these reports: no independent validation, no model approval, no change control, no override policy. The monitoring plan in `model_validation.md` is a design and has never been executed.
4. **Not for real lending.** Nothing here should be used to take decisions about real customers.

## 5. Out of scope (by design)

Loss given default, exposure at default, expected credit loss and IFRS 9 staging; behavioural scorecards; deployment, dashboards and APIs; deep-learning models; resampling methods (deliberately excluded because they distort PD).

## 6. Planned next steps

| Phase | Item |
|---|---|
| 1.5 | XGBoost challenger on the same split and metrics, SHAP explanations, bootstrap confidence intervals for the AUC gap |
| 1.5 | Sensitivity checks: other seeds, duplicates removed, alternative bin settings |
| 2 | Repeat the core on Lending Club with **out-of-time validation**, PSI/CSI per variable over time, vintage analysis, and a monitoring run |
