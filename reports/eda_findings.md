# Exploratory Data Analysis: Key Findings and Modelling Implications

These findings come from the exploratory analysis of the cleaned CarDekho dataset (`data/interim/cars_clean.csv`, 15,244 rows, 13 columns). Each finding names the figure in `reports/figures/` and the table in `reports/tables/` it rests on. Shares, ratios and skewness values are computed in `reports/tables/report_key_figures.csv` (by `src/carprice/report_facts.py`), not by hand.

> **Correction note (Phase 8, D-038).** An audit in Phase 8 found that several figures in the earlier version of this file did not match their own tables (for example the transmission medians, fuel means, seller medians, brand counts and the 4-seater median). They have been replaced with the table values below. The conclusions drawn for modelling did not change.

---

## Part 1: Numbered findings

### 1. Price is heavily right-skewed, so the model should predict log price
- **Finding:** Selling price has a skewness of **10.11** [`report_key_figures.csv`]. The median is **₹ 5.59 lakh** and three quarters of cars list below **₹ 8.25 lakh**, but the maximum is **₹ 3.95 crore** (mean ₹ 7.75 lakh, standard deviation ₹ 8.95 lakh) [`eda_numeric_summary.csv`].
- **Evidence:** `log1p(price)` has a skewness of **0.56**, close to symmetric [`report_key_figures.csv`].
- **Artefacts:** `reports/figures/eda_01_selling_price_raw_vs_log.png`, `reports/figures/eda_02_dist_selling_price.png`.

### 2. Power and engine size are the strongest numeric predictors
- **Finding:** `max_power` has the strongest rank correlation with price (Spearman **ρ = 0.72**; Pearson **r = 0.76** with log price). Engine displacement follows with **ρ = 0.64** [`eda_spearman_correlation.csv`, `eda_pearson_correlation.csv`].
- **Evidence:** The binned-median trend lines in the scatter plots rise steeply and non-linearly with power, which suits tree models and a log target.
- **Artefacts:** `reports/figures/eda_20_scatter_price_vs_power.png`, `reports/figures/eda_21_scatter_price_vs_engine.png`.

### 3. Automatics are dearer, and the premium survives controlling for power
- **Finding:** The median automatic lists at **₹ 10,50,000** against **₹ 5,00,000** for a manual [`eda_transmission_medians.csv`], a raw ratio of **2.10** [`report_key_figures.csv`].
- **Evidence:** Part of that gap is because automatics are more powerful cars. Within each power quartile the automatic premium remains: for example ₹ 4,49,500 vs ₹ 3,40,000 in the lowest quartile (38-74 bhp) and ₹ 16,50,000 vs ₹ 8,75,000 in the highest (118-626 bhp). Mann-Whitney U tests are significant in every quartile (p < 0.001) with rank-biserial effect sizes from **0.26 to 0.61** [`eda_hypothesis_tests.csv`].
- **Artefacts:** `reports/figures/eda_15_price_by_transmission.png`, `reports/figures/eda_24_interaction_power_transmission.png`.

### 4. Price falls steadily with age
- **Finding:** Age is the main downward driver (Spearman **ρ = -0.48**) [`eda_spearman_correlation.csv`].
- **Evidence:** For the five highest-volume brands the estimated average annual price loss ranges from **9.39%** (Mahindra) to **12.89%** (Hyundai) [`eda_depreciation_rates.csv`]. Among these five, Toyota has the highest median price, **₹ 12,75,000** [`eda_brand_medians.csv`].
- **Artefacts:** `reports/figures/eda_18_scatter_price_vs_age.png`, `reports/figures/eda_22_depreciation_curves.png`.

### 5. Diesel cars list higher than petrol cars
- **Finding:** Median diesel price **₹ 7,00,000** (mean ₹ 10.00 lakh) against **₹ 4,60,000** for petrol (mean ₹ 5.73 lakh), a ratio of **1.52**. CNG cars have a median of ₹ 3,70,000 and LPG cars ₹ 1,82,500 [`eda_fuel_medians.csv`, `report_key_figures.csv`].
- **Evidence:** A Kruskal-Wallis test across diesel, petrol and CNG is significant (H = 2527.5, p < 0.001, effect size 0.32) [`eda_hypothesis_tests.csv`]. The age-by-fuel facets show diesel above petrol at every age.
- **Artefacts:** `reports/figures/eda_14_price_by_fuel.png`, `reports/figures/eda_25_interaction_age_fuel.png`.

### 6. Two brands dominate, and brand medians span two orders of magnitude
- **Finding:** Maruti (**4,933** cars, **32.36%**) and Hyundai (**2,952**, **19.36%**) together make up **51.73%** of listings [`eda_brand_medians.csv`, `report_key_figures.csv`]. The cleaned data has **31** brands, of which **13** have fewer than 30 listings [`report_key_figures.csv`, `eda_rare_brands.csv`].
- **Evidence:** Median prices run from ₹ 3,00,000 (Datsun) and ₹ 4,65,000 (Maruti) through ₹ 12,75,000 (Toyota) and ₹ 17,00,000 (Audi) to ₹ 22,87,500 (BMW) and ₹ 50,00,000 (Porsche, 21 cars). Single listings reach ₹ 3.95 crore (Ferrari) [`eda_brand_medians.csv`].
- **Artefacts:** `reports/figures/eda_08_cat_brand.png`, `reports/figures/eda_13_price_by_brand.png`.

### 7. Dealer listings are slightly dearer than private ones
- **Finding:** Median prices are ₹ 5,90,000 for dealers, ₹ 5,40,000 for Trustmark dealers and ₹ 5,10,000 for individuals [`eda_seller_medians.csv`], a dealer/individual ratio of **1.16** [`report_key_figures.csv`].
- **Evidence:** The difference is statistically significant (Mann-Whitney U, p < 0.001) but the effect is small (rank-biserial **0.18**) [`eda_hypothesis_tests.csv`]. Dealers may stock newer cars; the test does not control for that.
- **Artefacts:** `reports/figures/eda_16_price_by_seller.png`.

### 8. Kilometres driven matter only weakly on their own
- **Finding:** `km_driven` has a weak negative rank correlation with price (**ρ = -0.17**) [`eda_spearman_correlation.csv`].
- **Evidence:** The binned trend falls over the lower kilometre range and then flattens. Kilometres are strongly tied to age (ρ = 0.58 between them), so much of their effect is already carried by age.
- **Artefacts:** `reports/figures/eda_19_scatter_price_vs_km.png`.

### 9. Engine size and power are collinear
- **Finding:** `engine` and `max_power` correlate strongly (Pearson **r = 0.81**, Spearman **ρ = 0.83**) [`eda_pearson_correlation.csv`, `eda_spearman_correlation.csv`].
- **Evidence:** VIF is **6.41** for `engine` (R² 0.8439) and **4.56** for `max_power` (R² 0.7806). The other numeric inputs stay below 2.5 (`vehicle_age` 1.43, `km_driven` 1.43, `mileage` 1.96, `seats` 2.27) [`eda_vif_summary.csv`].
- **Artefacts:** `reports/figures/eda_23_correlation_heatmaps.png`.

### 10. Seat count mostly reflects the type of car
- **Finding:** 7-seaters list at a median of **₹ 8,75,000** against **₹ 5,25,000** for 5-seaters (ratio **1.67**). The dearest groups are 2-seaters (₹ 34,25,000, 7 cars) and 4-seaters (₹ 27,50,000, 76 cars) [`eda_seats_medians.csv`, `report_key_figures.csv`].
- **Evidence:** Five seats is by far the most common layout (**83.76%** of listings) [`report_key_figures.csv`]. 2- and 4-seaters are mostly sports and luxury cars, and 7-seaters are family MPVs and SUVs, so seats act as a proxy for body type rather than a price driver on their own.
- **Artefacts:** `reports/figures/eda_12_cat_seats.png`, `reports/figures/eda_17_price_by_seats.png`.

---

## Part 2: What this meant for modelling

1. **Predict log price.** The model is wrapped in `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` (D-009, D-017), so squared-error losses are not dominated by the luxury tail.
2. **Candidate features, then measured.** The EDA proposed `power_per_litre`, `km_per_year`, brand tiers and a power × automatic interaction. Phase 3 measured each candidate by cross-validation on the training split only: `power_per_litre`, `km_per_year`, `is_luxury_brand` and `age_x_power` were kept; `power_x_automatic` and `log_km_driven` were rejected; brand tiers were not used as a model input (D-013, D-015) [`feature_checks.csv`].
3. **Collinearity.** Engine and power are collinear (finding 9), which argues for a regularised linear baseline (Ridge) and for tree ensembles, which are not harmed by it.
4. **Categorical encoding.** `model` has **120** levels and `brand` **31** in the cleaned data [`report_key_figures.csv`]. `model` is target-encoded with internal cross-fitting; rare brands are grouped into "Other" inside each training fold before one-hot encoding (D-014, D-015).
