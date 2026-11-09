# Viva notes: likely questions and answers

Short answers grounded in the project's decisions (D-numbers in `decisions.md`) and saved results (files in `reports/tables/` unless stated). Each answer is written to be said aloud in under a minute; the bracketed file is where to point if asked "how do you know?".

---

### 1. Why did you log-transform the target?

Price is extremely right-skewed: skewness 10.11, falling to 0.56 after `log1p` [`report_key_figures.csv`]. With squared error on raw rupees, a few hundred luxury cars would dominate training and the mass market would be fitted poorly. On the log scale an error of 0.10 is roughly a 10% miss whatever the price, so cheap and expensive cars count equally. I used `TransformedTargetRegressor` so the pipeline trains on log price but takes and returns rupees (D-009, D-017).

### 2. Why was the test set locked, and how do you know it was used only once?

Any decision made after looking at test results turns the test set into a second validation set and makes its scores optimistic. The split was made straight after cleaning (D-012); `load_training_data()` refuses to read `test.csv`, which a test checks; and all model choices used five-fold CV on training data. The metrics were committed before scoring (D-023), the test set was scored once, and nothing was changed afterwards (D-025). Git history shows the order of these commits.

### 3. How did you prevent data leakage?

Split first, then fit everything inside one scikit-learn pipeline, so imputers, scalers, the rare-category grouping, one-hot encoding and the target encoder are refitted on each training fold. The target encoder also cross-fits internally, so a row never sees its own price. Exact duplicates were removed before the split, because a copy in both train and test leaks the answer (D-003). Three steps touch the full data, and each was checked: seat imputation for 2 rows using no prices, a brand-tier mapping used only for reporting, and a fixed list of luxury brands (D-022).

### 4. Why drop duplicates but keep cars with the same specification and a different price?

The 167 exact duplicates (1.084%) are the same listing twice; across a split they inflate scores. The 2,148 rows with identical specifications but different prices are different cars whose prices differ for reasons the data do not record, such as condition. That spread is real, and the model must learn to live with it [`audit_duplicates.csv`].

### 5. Why cap kilometres instead of dropping those rows?

Twelve cars claimed more than 5,00,000 km, up to 38,00,000 km, on cars only a few years old: clearly entry errors [`cleaning_log.csv`, `audit_numeric_summary.csv`]. Capping keeps the rest of each row and limits leverage on linear models (D-004). I should be honest that the flag column `km_was_capped` turned out not to be a model input (D-024); permutation importance shows it carries exactly zero weight, and with 12 rows the effect is nil.

### 6. Why keep luxury cars up to ₹ 3.95 crore? Wouldn't removing them improve your metrics?

Yes, it would, and that is why I did not. They are genuine listings. Removing them would flatter the scores while making the model useless for those cars. Instead the evaluation reports accuracy by segment, which shows honestly that luxury cars are a weakness (D-007, D-026).

### 7. Why did you stratify the split on price?

So that cheap, mid-range and luxury cars appear in the same proportions in training and test. With a heavy tail, an unlucky random split could put most supercars on one side (D-012).

### 8. Why does XGBoost beat Ridge?

On the test set, MAE is ₹ 94,648 against ₹ 1,15,850 for Ridge, an 18.30% reduction, and the bootstrap interval for the difference excludes zero [`test_metrics.csv`, `report_key_figures.csv`, `bootstrap_ci.csv`]. The reason is the shape of the problem: price rises non-linearly with power, depreciation differs between car types (the `age_x_power` interaction gave the biggest feature gain), and the car model interacts with age and power. Trees learn these interactions automatically; a linear model needs each one hand-built.

### 9. XGBoost, LightGBM and histogram boosting scored almost the same. Why XGBoost?

Because the selection rule was fixed in advance: lowest mean CV log RMSE (D-019). Tuned XGBoost scored 0.1654, the other two 0.1660 [`models/best_params.json`]. The gap is smaller than one fold standard deviation, so I describe them as tied; any of the three would have been a sound choice.

### 10. How did you tune the models?

Optuna's TPE sampler, 35 trials for each of the three boosters, seed 42, with search spaces in the config (D-018). Each trial was scored by five-fold CV log RMSE on training data. Preprocessing was fitted once per fold and reused across trials, which is identical to refitting it each trial but much faster. Every trial is logged in `tuning_trials_*.csv`.

### 11. How is the price range computed, and can it be trusted?

From the out-of-fold residuals on the log scale: their 10th and 90th percentiles, -0.1903 and 0.1915, are added to the log prediction and transformed back [`models/model_card.json`] (D-020). It assumes no particular distribution. On the test set the 80% range contains the true price for 79.96% of cars, with a confidence interval of 78.5% to 81.34%, so it is well calibrated overall [`interval_coverage.csv`]. But not for every group: coverage is 92.45% for cars up to 2 years old and 52.38% for 12+ years [`segment_errors.csv`].

### 12. What happens if a user enters a model name the system has never seen?

It is refused with a clear message. The target encoder would silently map an unknown model to the average price and produce a confident-looking wrong answer, so the predictor checks the brand and model against the training data first: "Model 'Swift' is made by Maruti, not Hyundai", or "not a known Maruti model", with a list of valid names (D-030). In the GUI it cannot happen at all, because brand and model are drop-down lists drawn from the training data (D-034). Out-of-range numbers and unknown fuels are refused the same way, all problems reported at once (D-029).

### 13. Why target-encode `model` instead of one-hot encoding it?

It has 120 levels. For the tuned XGBoost, target encoding gave a lower CV error than one-hot (0.1654 against 0.1673) and replaces about a hundred sparse columns with one [`encoder_comparison.csv`]. For a linear model the result reverses, which I report (D-014). Leakage is avoided by internal cross-fitting.

### 14. What is the model's biggest weakness?

Old cars. For cars 12 years or older only 32.74% are priced within 10%, against 50.97% overall, and the 80% range covers only 52.38% of them [`segment_errors.csv`]. The data have no condition, service or accident information, and those matter most for old cars. Luxury cars and the cheapest cars are next. The GUI warns users in words when their car falls in one of these groups (D-034).

### 15. How do you know the model is not overfitting?

The test log RMSE (0.1648) matches the cross-validation estimate (0.1654), a gap of -0.36% [`train_vs_test.csv`]. The in-sample fit is tighter (0.1387), which is normal for boosted trees, but it did not turn into a test penalty.

### 16. Why are MAE and within-10% your headline metrics, rather than R² or RMSE?

They answer the questions a car buyer asks: how many rupees off on average, and how often is it close. R² is dominated by the expensive cars on the rupee scale, and RMSE squares errors so a few luxury misses dominate it. I still report all of them, on both scales, for all three models [`test_metrics.csv`] (D-023).

### 17. Why is MAPE (12.7%) higher than the median absolute percentage error (9.75%)?

A few cheap cars have large relative errors: missing a car priced under ₹ 2 lakh by a modest rupee amount is a big percentage. The mean is pulled up by them; the median describes the typical car [`test_metrics.csv`].

### 18. Which inputs matter most, and does that agree with your EDA?

Car model, age and power. In SHAP terms, model accounts for 30.41% of the total effect, age 20.35% and power 12.78% [`shap_importance.csv`]; permutation importance gives the same top three [`permutation_importance.csv`]. That matches the EDA, where power had the strongest rank correlation with price (0.72) and age the strongest negative one [`eda_spearman_correlation.csv`]. Transmission and fuel matter less to the model than their raw price gaps suggest, because power and model already explain most of those gaps.

### 19. Why did you use non-parametric tests in the EDA?

Prices are not normally distributed, so t-tests and ANOVA would rest on a false assumption. Mann-Whitney U and Kruskal-Wallis compare ranks, and I report effect sizes, because with 15,244 rows almost any difference has a tiny p-value (D-011). For example, the dealer premium is significant but small (effect size 0.18) [`eda_hypothesis_tests.csv`].

### 20. What would you do with more time or data?

First, richer data: trim and variant, condition, accident history and city. That is where most of the remaining error lies. Second, intervals that widen for old and luxury cars, such as conformal or quantile intervals. Third, monitoring for drift and periodic retraining with a fresh hold-out set. I would not keep tuning the algorithm: the three boosters are already tied.

---

## Numbers worth remembering

| What | Value | Source |
|---|---|---|
| Rows raw / cleaned | 15,411 / 15,244 | `cleaning_log.csv` |
| Train / test | 12,195 / 3,049 | `models/model_card.json`, `test_metrics.csv` |
| CV log RMSE, tuned XGBoost | 0.1654 | `models/best_params.json` |
| Test MAE | ₹ 94,648 | `test_metrics.csv` |
| Within 10% / 20% | 50.97% / 82.09% | `test_metrics.csv` |
| Ridge test MAE | ₹ 1,15,850 | `test_metrics.csv` |
| Interval coverage | 79.96% | `interval_coverage.csv` |
| Weakest segment | 12+ years, 32.74% within 10% | `segment_errors.csv` |
