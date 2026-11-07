# Evaluation summary: used car price model (Phase 5)

The tuned XGBoost pipeline was scored **once** on the locked test set of 3,049 cars that it never saw in training or tuning (decisions D-023, D-025). Every number below comes from a file in `reports/tables/`, named in brackets.

## Headline result

On a typical car the model is off by about a tenth of the price. The average miss is **₹ 94,648** (95% CI ₹ 88,440 to ₹ 1,01,444), and **50.97%** of cars are priced within 10% of their true price (95% CI 49.23% to 52.71%) [`test_metrics.csv`, `bootstrap_ci.csv`].

## Metrics on the test set [`test_metrics.csv`]

| Metric | Final model (tuned XGBoost) | Ridge baseline | Dummy (train median) |
|---|---|---|---|
| MAE (₹) | ₹ 94,648 | ₹ 1,15,850 | ₹ 3,85,255 |
| RMSE (₹) | ₹ 2,02,801 | ₹ 2,66,575 | ₹ 8,07,416 |
| R² (₹) / adjusted | 0.9328 / 0.9319 | 0.8839 / 0.8824 | -0.0654 / -0.0654 |
| MAE (log price) | 0.1247 | 0.1457 | 0.5016 |
| RMSE (log price) | 0.1648 | 0.1881 | 0.6732 |
| R² (log) / adjusted | 0.9399 / 0.9391 | 0.9216 / 0.9206 | -0.0036 / -0.0036 |
| MAPE | 12.7% | 14.86% | 53.67% |
| Median absolute % error | 9.75% | 12.04% | 37.08% |
| Within 10% of true price | 50.97% | 42.93% | 14.43% |
| Within 20% of true price | 82.09% | 75.11% | 29.94% |
| Median signed % error (bias) | -0.34% | -0.65% | 1.08% |

The final model beats the Ridge baseline by ₹ 21,250 of MAE per car, and the 95% bootstrap interval for that gain (₹ 14,851 to ₹ 28,252) excludes zero, so the improvement is not luck of the split [`bootstrap_ci.csv`]. MAPE (12.7%) sits above the median (9.75%) because a few cheap cars with large relative misses pull the mean up, as anticipated in D-023.

## Strengths

- **No overfitting to the training folds.** Test log RMSE is 0.1648 against 0.1654 in 5-fold CV (a -0.36% gap). The in-sample fit is tighter (0.1387), which is normal for boosted trees and does not carry over into a test penalty [`train_vs_test.csv`].
- **Honest price range.** The 80% interval contains the true price for 79.96% of test cars (95% CI 78.5% to 81.34%): **well calibrated**. Misses are balanced, 10.07% below the range and 9.97% above it. The median interval is 38.44% of the predicted price wide [`interval_coverage.csv`].
- **No systematic bias.** The median log residual is 0.0034 and the median signed error is -0.34% [`residual_stats.csv`, `test_metrics.csv`].
- **Learned what the EDA predicted.** The car model, age and power dominate both SHAP and permutation importance (see below).

## Weaknesses: the five worst segments [`segment_errors.csv`]

Ranked by the lowest within-10% share among segments with at least 30 test cars (D-023):

| Rank | Segment | Test cars | Within 10% | Median abs % error | Interval coverage |
|---|---|---|---|---|---|
| 1 | Vehicle age 12+ yrs | 168 | 32.74% | 18.9% | 52.38% |
| 2 | Vehicle age 9-11 yrs | 451 | 33.92% | 14.39% | 65.41% |
| 3 | Brand: Bmw | 76 | 34.21% | 14.94% | 68.42% |
| 4 | Price band Q1: ₹0.4-3.5 lakh | 650 | 39.23% | 12.96% | 68.92% |
| 5 | Above ₹20 lakh | 136 | 41.18% | 12.18% | 71.32% |

- **Old cars.** Accuracy falls steadily with age, from 61.89% within 10% at 0-2 years to 32.74% at 12+ years. Condition varies far more between old cars, and the data does not record it.
- **Cheap cars are over-priced** (median signed error +7.26% in Q1), and **luxury cars are under-priced** (-5.76% above ₹20 lakh). Both are pulled towards the middle of the market.
- **The interval is right on average but not for every group.** Coverage is 92.45% for 0-2 year-old cars but only 52.38% for 12+ year-old cars. One fixed-width interval is too wide for new cars and too narrow for old ones.
- LPG cars score 27.27% within 10%, but there are only 11 of them, too few to rank.

## The 20 largest errors [`top_errors.csv`, `top_error_patterns.csv`]

The 20 largest rupee misses are 0.66% of test cars but 12.04% of all rupee error. 18 of the 20 are luxury brands and 17 sell for over ₹20 lakh. All 20 are automatics, and 18 fall outside their 80% interval. 12 were under-priced and 8 over-priced. The pattern is coarse model names: the data lumps very different cars under one label (for example "Rover" for every Land Rover, "C" for a Mercedes-AMG C-class), so trim and engine variants that change the price by lakhs are invisible. A few look like data or condition outliers, such as a Volkswagen Polo with 189 bhp listed at ₹ 8,75,000 and predicted 145.39% higher.

## What drives the predictions [`permutation_importance.csv`, `shap_importance.csv`]

The model's price is driven mainly by **which car it is, how old it is and how powerful it is**. In SHAP terms, `model` accounts for 30.41% of the total effect, `vehicle_age` 20.35%, `max_power` 12.78% and `engine` 8.16%. Shuffling `vehicle_age` raises test log RMSE by 0.2854, `model` by 0.2788 and `max_power` by 0.117. Older cars are priced lower, more powerful cars higher.

This agrees with the EDA: power had the strongest rank correlation with price (Spearman 0.7211), followed by engine size (0.6429) and age (-0.4778) [`eda_spearman_correlation.csv`]. Transmission and fuel matter far less in the model (SHAP shares 5.15% and 2.82%) than their raw price gaps suggest (median ₹ 10,50,000 automatic vs ₹ 5,00,000 manual [`eda_transmission_medians.csv`]). The model attributes most of those gaps to power, engine and car model, which differ systematically between automatics and manuals or diesels and petrols. `km_was_capped` has exactly zero importance, confirming that the pipeline never uses it (D-024).

## Limitations

- **Missing information:** no condition, accident history, service record, city, colour, trim or variant. These explain most of the error on old and luxury cars.
- **Thin data at the extremes:** 136 test cars above ₹20 lakh and 168 aged 12+ years. Few training examples means less reliable prices there.
- **Listing prices, not sale prices:** the target is the price advertised on CarDekho, which may differ from what the car actually sold for.
- **The interval is calibrated only on average** (see weaknesses): it should be read as "typical range", not a guarantee for every car type.
- **A snapshot in time:** prices reflect one period of the Indian market and will drift.

## Verdict

**Fit for its purpose as a first-opinion price guide for mainstream cars**: under ₹20 lakh, up to about 8 years old, from high-volume brands. Those segments score between 48.49% and 61.89% within 10%, and at least 82.32% within 20%. **Users should not rely on it** for luxury cars, cars 9 or more years old, or rare models and fuels; the GUI should say so beside the estimate. The biggest gains would come from richer data (trim and variant, condition, accident history, city), then from intervals that widen with age and price band.
