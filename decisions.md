# Decision Log (decisions.md)

This log records every non-trivial design, modeling, cleaning, and architectural choice made during the development of the Used Car Price Prediction System. It serves as auditable evidence for academic grading, viva defence, and reproducibility.

---

## Assumptions

1. The target prediction task is regression: estimating continuous used car selling prices in Indian Rupees (INR) alongside plausible prediction intervals.
2. The raw dataset (`data/raw/cardekho_dataset.csv`, 15,411 rows, 14 columns) is treated as an immutable, read-only source of truth. All data cleaning transforms produce distinct files under `data/interim/` and `data/processed/`.
3. To eliminate data leakage, the train/test split (80/20) must be executed before fitting any data-dependent transformations, encoders, scalers, or imputation models.
4. All reusable computation, validation, and pipeline definitions reside strictly in `src/carprice/`, with Jupyter notebooks restricted to visual EDA and narrative reports (no standalone functions exceeding 10 lines).
5. All execution is governed by a fixed global random seed (42) loaded via `configs/config.yaml`.

---

## Decisions

### D-001: Selection of Tech Stack and Environment Configuration

- **Date / Phase:** 2026-11-07 / Phase 0
- **Context:** The project requires a reproducible, leak-free machine learning workflow with explainability and an interactive GUI for used car price prediction.
- **Options considered:**
  - (a) Generic ad-hoc Python notebooks with mixed libraries and unpinned dependencies.
  - (b) A fixed, disciplined tech stack: Python 3.11, virtual environment, pinned dependencies, pandas, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, Streamlit, pytest, and ruff.
- **Decision:** Adopt option (b) with Python 3.11.9, pinned in `requirements.txt`.
- **Reason:**
  - *Python 3.11.9*: Provides strong typing, enhanced traceback error reporting, and maximum stability across ML C-extensions (LightGBM, XGBoost, SHAP).
  - *Package versions pinned*: `pandas==3.1.1`, `numpy==2.4.6`, `scikit-learn==1.9.1`, `xgboost==3.2.0`, `lightgbm==4.7.0`, `optuna==5.0.0`, `shap==0.51.0`, `joblib==1.6.0`, `streamlit==1.67.0`, `pytest==9.2.0`, `ruff==0.17.4`.
  - *Streamlit*: Chosen because it creates modern, responsive multi-page web UIs for ML prediction and EDA exploration without boilerplate front-end code.
  - *XGBoost*: Chosen because its depth-wise gradient boosted trees excel at modeling complex non-linear tabular interactions with robust L1/L2 regularization.
  - *LightGBM*: Chosen because its leaf-wise tree splitting and histogram-based binning deliver ultrafast training speeds and native categorical optimization during tuning.
  - *Optuna*: Provides Bayesian optimization (TPE) with disciplined trial logging and fixed seeds.
  - *SHAP & Permutation Importance*: Ensures model transparency and local/global explainability for viva defence.
- **Evidence:** Tech stack specifications in `AGENTS.md` section A2 and `requirements.txt`.
- **Impact / how to reverse:** Configured in `requirements.txt` and `pyproject.toml`.

---

### D-002: Repository Layout, Modular Architecture, and Read-Only Raw Data

- **Date / Phase:** 2026-11-07 / Phase 0
- **Context:** Academic rigor demands clean separation of concerns, zero code duplication, leak-free pipeline encapsulation, and uncorrupted source data.
- **Options considered:**
  - (a) Monolithic repository structure with code embedded in Jupyter notebooks and in-place dataset modifications.
  - (b) Standardized modular layout separating `src/` modules, `configs/`, `notebooks/`, `app/`, `tests/`, and isolated `data/` tiers (`data/raw/`, `data/interim/`, `data/processed/`).
- **Decision:** Adopt option (b) with strict file permissions on `data/raw/cardekho_dataset.csv` marked as read-only.
- **Reason:** Prevents accidental modification of original data, ensures full testability of individual components via `pytest`, avoids notebook code sprawl, and enforces centralized parameter management via `configs/config.yaml` and `src/carprice/config.py`.
- **Evidence:** Repository layout definition in `AGENTS.md` section A3.
- **Impact / how to reverse:** Layout defined in root and tracked via Git; directories protected against direct overwrites.

---

### D-003: Deduplication Strategy (Dropping Exact Duplicates Post Identifier Removal)

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** With the arbitrary index column `Unnamed: 0` removed, 167 rows share 100% identical feature values and identical selling prices (1.08% of total volume). Additionally, 2,148 rows share identical vehicular specifications but possess differing selling prices.
- **Options considered:**
  - (a) Retain all rows as given.
  - (b) Remove all rows sharing identical vehicular specifications regardless of price.
  - (c) Drop the 167 exact duplicate rows (keeping first occurrence) and retain specification matches with distinct selling prices.
- **Decision:** Adopt option (c).
- **Reason:** Exact duplicate rows occurring across cross-validation or train/test splits lead to target leakage and artificial metric inflation. Removing 167 rows represents only 1.08% of the dataset (below the strict 2% threshold). The 2,148 records with different prices reflect genuine real-world price dispersion across conditions, sellers, and transactions and are legitimate data.
- **Evidence:** `reports/tables/audit_duplicates.csv`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` Section 2.
- **Impact / how to reverse:** Governed by `drop_exact_duplicates()` in `src/carprice/data.py`.

---

### D-004: Treatment of Extreme Odometer Outliers (Cap at 500,000 km)

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** 12 rows contain `km_driven` above 500,000 km (with an extreme maximum of 3,800,000 km on passenger cars aged 3 to 9 years).
- **Options considered:**
  - (a) Drop all 12 rows.
  - (b) Cap `km_driven` at 500,000 km and append a boolean flag column `km_was_capped`.
  - (c) Retain raw values without modification and rely solely on log transformation.
- **Decision:** Adopt option (b) with cap at 500,000 km.
- **Reason:** Capping limits undue leverage and gradient instability in linear and distance-based estimators while retaining every record. Adding the boolean `km_was_capped` flag allows non-linear decision trees to distinguish genuine high-mileage/fleet vehicles from standard usage cars.
- **Evidence:** `reports/tables/audit_km_outliers.csv`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` Section 8.
- **Impact / how to reverse:** Configured via `cleaning.km_cap` in `configs/config.yaml` and handled by `treat_km_outliers()`.

---

### D-005: Recoding of Electric Fuel Labels to Petrol Hybrids

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** Exactly 4 rows in the raw dataset are labeled with `fuel_type == 'Electric'`, but each row has a 2,494 cc internal combustion engine and 157.7 bhp output (all Toyota Camry sedans priced between ₹16.99 lakh and ₹20.00 lakh).
- **Options considered:**
  - (a) Drop the 4 rows entirely.
  - (b) Maintain the `Electric` label as a distinct category.
  - (c) Recode `fuel_type` from `Electric` to `Petrol`.
- **Decision:** Adopt option (c).
- **Reason:** A pure battery electric vehicle has 0 cc engine displacement. These vehicles are Toyota Camry Petrol-Electric Hybrids. Leaving a 4-sample category causes stratified CV splits to fail or create zero-variance test folds. Recoding to `Petrol` accurately reflects the primary internal combustion powertrain and preserves the records.
- **Evidence:** `reports/tables/audit_electric_rows.csv`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` Section 5.
- **Impact / how to reverse:** Governed by `fix_fuel_labels()` in `src/carprice/data.py`.

---

### D-006: Modal Imputation of Invalid Seat Counts (seats = 0)

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** 2 rows have `seats = 0` (row 3217: Honda City, row 12619: Nissan Kicks), which is physically impossible for road-legal production automobiles.
- **Options considered:**
  - (a) Delete both records.
  - (b) Impute with a global seat mode of 5.
  - (c) Impute dynamically with the brand-and-model-specific modal seat count.
- **Decision:** Adopt option (c).
- **Reason:** Both the Honda City and Nissan Kicks are well-documented 5-passenger passenger cars. Imputing with the model mode retains both samples with correct physical specifications and enforces that all seat values lie within $[2, 9]$.
- **Evidence:** `reports/tables/audit_seats_anomalies.csv`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` Section 8.
- **Impact / how to reverse:** Implemented in `fix_invalid_seats()` in `src/carprice/data.py`.

---

### D-007: Retention of High-End Luxury Cars and Supercars

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** The dataset contains premium luxury vehicles priced above ₹50,00,000 (and up to ₹3.95 crore), including Mercedes-Benz, BMW, Porsche, Ferrari, and Rolls-Royce models.
- **Options considered:**
  - (a) Truncate or drop cars above ₹50 lakh to artificially lower RMSE/MAE figures.
  - (b) Retain all genuine luxury vehicles and evaluate model performance by price segments.
- **Decision:** Adopt option (b).
- **Reason:** Deleting genuine high-value sales to fabricate better regression scores is academically dishonest. These cars represent genuine market transactions. Segmented performance analysis during Phase 5 will separately quantify accuracy across budget, mid-range, and luxury tiers.
- **Evidence:** `reports/tables/audit_numeric_summary.csv`, `notebooks/01_data_audit.ipynb` Section 3.
- **Impact / how to reverse:** Preserved by policy; downstream evaluation scripts will report stratified metrics by price segment.

---

### D-008: Redundant Identifier Stripping and Schema Gatekeeping

- **Date / Phase:** 2026-11-07 / Phase 1
- **Context:** `Unnamed: 0` is an arbitrary row indexing artifact, and `car_name` matches `brand + ' ' + model` with zero discrepancies across all 15,411 rows.
- **Options considered:**
  - (a) Keep all identifier columns in the interim dataset.
  - (b) Strip redundant identifiers and enforce a schema contract (`validate_schema()`) across all subsequent pipeline modules.
- **Decision:** Adopt option (b).
- **Reason:** Eliminates collinearity and index leakages while keeping clean, structured `brand` and `model` columns. Calling `validate_schema()` at the start of all downstream stages prevents data corruption from propagating silently.
- **Evidence:** `reports/tables/audit_name_consistency.csv`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` Section 6.
- **Impact / how to reverse:** Implemented in `drop_identifier_columns()` and `validate_schema()`.

---

### D-009: Target Transformation to Logarithmic Price (log1p)

- **Date / Phase:** 2026-11-07 / Phase 2
- **Context:** The raw distribution of `selling_price` is heavily right-skewed with a skewness coefficient of 10.11, median of ₹5.59 lakh, mean of ₹7.75 lakh, and a maximum reaching ₹3.95 crore.
- **Options considered:**
  - (a) Predict raw selling price directly in INR.
  - (b) Apply Box-Cox power transformation.
  - (c) Apply natural logarithmic transformation $y' = \log(1 + y)$ (`log1p`) with exponential inverse transformation $\exp(y') - 1$ (`expm1`).
- **Decision:** Adopt option (c) using `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
- **Reason:** `log1p` stabilizes error variance (homoscedasticity) across orders of magnitude and reduces skewness from 10.11 down to 0.56 (a near-Gaussian target). Direct raw regression causes squared-error loss functions to be dominated by the 1% luxury tail at the expense of typical mass-market predictions. Box-Cox requires estimating $\lambda$ on training folds, introducing unnecessary pipeline complexity for negligible gain over natural log.
- **Evidence:** `reports/figures/eda_01_selling_price_raw_vs_log.png`, `reports/tables/eda_numeric_summary.csv`, `notebooks/02_eda.ipynb` Section 2.
- **Impact / how to reverse:** Encapsulated in the target regressor inside `src/carprice/train.py`.

---

### D-010: Feature Engineering Proposals Derived from Exploratory Analysis

- **Date / Phase:** 2026-11-07 / Phase 2
- **Context:** EDA revealed domain-specific non-linear interactions, usage degradation patterns, and high brand cardinality (32 brands with 14 brands having $< 30$ rows).
- **Options considered:**
  - (a) Rely solely on raw features without domain transformations.
  - (b) Engineer interaction and ratio features:
    1. `power_per_litre`: $\text{max\_power} / (\text{engine} / 1000)$ capturing engine efficiency and turbocharging.
    2. `km_per_year`: $\text{km\_driven} / \max(\text{vehicle\_age}, 1)$ capturing fleet vs private usage wear.
    3. `brand_tier`: Grouping 32 brands into 5 hierarchical tiers (`Budget`, `Mid`, `Premium`, `Luxury`, `Ultra-Luxury`) based on median valuation quartiles.
    4. `power_x_automatic`: Interaction indicator between engine power and automatic transmission.
- **Decision:** Adopt option (b).
- **Reason:** Empirical evidence shows automatic gearboxes command a statistically significant 2.74x premium ($p < 0.001$) across all power bands; engine displacement and power exhibit collinearity (VIF 6.41 and 4.56); and rare brands cause sparse high-dimensional one-hot matrices unless consolidated into structured market tiers.
- **Evidence:** `reports/figures/eda_20_scatter_price_vs_power.png`, `reports/figures/eda_24_interaction_power_transmission.png`, `reports/tables/eda_vif_summary.csv`, `reports/tables/eda_brand_medians.csv`.
- **Impact / how to reverse:** Implemented as a leak-free scikit-learn transformer in `src/carprice/features.py` during Phase 3.
- **Revision (2026-11-08, see D-021):** these were *proposals*. Phase 3 screening (D-013) decided the final set. `power_x_automatic` was measured and rejected (ΔRMSE 0.0000). `brand_tier` was never added to the pipeline; brand information reaches the model through one-hot `brand` (rare levels grouped, D-015) and the `is_luxury_brand` flag.

---

### D-011: Statistical Hypothesis Testing Framework and Non-Parametric Metrics

- **Date / Phase:** 2026-11-07 / Phase 2
- **Context:** To substantiate market driver claims for viva defence, formal hypothesis tests were required to evaluate whether automatic transmission and dealer premiums are statistically genuine after controlling for confounding factors.
- **Options considered:**
  - (a) Standard two-sample Student's t-test and ANOVA.
  - (b) Non-parametric Mann-Whitney U test and Kruskal-Wallis H test with rank-biserial effect sizes ($r_{rb}$).
- **Decision:** Adopt option (b).
- **Reason:** Price distributions violate the normality assumption required by t-tests even after transformation. Mann-Whitney U evaluates rank-order stochastic dominance without distributional assumptions. Reporting rank-biserial effect sizes ($r_{rb} = 0.26$ to $0.61$) ensures practical significance is proven rather than relying solely on low $p$-values driven by large sample size ($N = 15,244$).
- **Evidence:** `reports/tables/eda_hypothesis_tests.csv`, `notebooks/02_eda.ipynb` Section 10.
- **Impact / how to reverse:** Governed by analysis scripts; test outputs persisted in `reports/tables/eda_hypothesis_tests.csv`.

---

### D-012: Stratified Train/Test Split (80/20 on Price Quintiles with Seed 42)

- **Date / Phase:** 2026-11-07 / Phase 3
- **Context:** The project requires a reproducible, leak-free evaluation protocol separating an 80% training set (12,195 rows) from a locked 20% test set (3,049 rows).
- **Options considered:**
  - (a) Pure random splitting without stratification.
  - (b) Stratified split based on selling price quintiles, with bins discarded after partitioning.
  - (c) Temporal split based on vehicle age.
- **Decision:** Adopt option (b) with global seed 42.
- **Reason:** Stratifying on 5 price quintiles guarantees that budget cars, mainstream commuters, and rare ultra-luxury supercars are equally represented across both train and test partitions within $< 2\%$ deviation. The test set is written to `data/processed/test.csv` and indices to `split_indices.json`, remaining completely locked from all Phase 3 and Phase 4 model training and feature fitting code.
- **Evidence:** `data/processed/split_indices.json`, `tests/test_features.py` test `test_split_shapes_and_disjoint`.
- **Impact / how to reverse:** Configured in `configs/config.yaml` (`splitting.test_size: 0.2`, `splitting.stratify_by: selling_price`, `splitting.stratify_bins: 5`, `project.random_seed: 42`) and managed by `create_train_test_split()`. Re-running `python tasks.py split` reproduces `split_indices.json` byte-for-byte (verified 2026-11-08, D-021).

---

### D-013: Candidate Feature Engineering Evaluations and Selection Decisions

- **Date / Phase:** 2026-11-07 / Phase 3 (numbers corrected 2026-11-08, see D-021)
- **Context:** EDA proposed candidate domain features (D-010). Each candidate was added to a scaled Ridge probe (`build_ridge_probe()`, alpha from `feature_checks.probe_ridge_alpha`) and scored by 5-fold CV log RMSE on `train.csv` only. Baseline CV RMSE without any candidate: **0.2020**.
  - *`km_per_year`:* $	ext{km\_driven} / \max(	ext{vehicle\_age}, 1)$ → 0.2018 (Δ −0.0002) → **KEEP**.
  - *`power_per_litre`:* $	ext{max\_power} / (	ext{engine} / 1000)$ → 0.2016 (Δ −0.0005) → **KEEP**.
  - *`log_km_driven`* (replacing raw `km_driven`) → 0.2025 (Δ +0.0004) → **REJECT**. km is already capped at 500,000 (D-004), so the log adds nothing for a linear model.
  - *`is_luxury_brand`:* binary flag for the marques in `features.luxury_brands` → 0.2009 (Δ −0.0011) → **KEEP**.
  - *`age_x_power`:* vehicle age × max power → 0.1925 (Δ −0.0095) → **KEEP**. Lets a linear model capture faster depreciation of powerful cars.
  - *`power_x_automatic`:* max power × automatic flag → 0.2021 (Δ 0.0000) → **REJECT**. The automatic premium (D-010) is already captured by one-hot `transmission_type`.
  - *All four kept features together:* **0.1903** (Δ −0.0117), the same value as the Ridge row in `cv_results.csv`.
- **Decision rule:** keep a candidate only if it lowers CV log RMSE by at least `feature_checks.min_rmse_gain` (0.0001).
- **Decision:** Retain `km_per_year`, `power_per_litre`, `is_luxury_brand`, `age_x_power`; reject `log_km_driven` and `power_x_automatic`.
- **Reason:** Every retained feature measurably improves CV accuracy on the training partition and has a physical interpretation. The gains for three of the four are small (≤ 0.0011); `age_x_power` carries most of the benefit.
- **Evidence:** `reports/tables/feature_checks.csv` (generated by `evaluate_feature_candidates()`); `tests/test_features.py::test_saved_feature_checks_match_pipeline_features` asserts that the KEEP rows equal the pipeline's engineered features.
- **Impact / how to reverse:** Implemented in `DomainFeatureEngineer` inside `src/carprice/features.py`; re-run with `python tasks.py split`.

---

### D-014: High-Cardinality Model Feature Encoding (Cross-Validated TargetEncoder)

- **Date / Phase:** 2026-11-07 / Phase 3 (evidence corrected 2026-11-08, see D-021)
- **Context:** The `model` feature contains ~120 unique levels, posing a risk of dimensionality explosion and sparse splits if one-hot encoded.
- **Options considered:**
  - (a) One-hot encoding with minimum-frequency grouping (`feature_checks.model_onehot_min_frequency: 10`).
  - (b) Scikit-learn's `TargetEncoder` with internal 5-fold cross-fitting (`smooth="auto"`).
- **Measured (5-fold CV log RMSE on `train.csv`, `reports/tables/encoder_comparison.csv`):**

  | Model family | TargetEncoder | One-hot |
  |---|---|---|
  | Ridge probe (base features) | 0.2020 | **0.1895** |
  | Tuned XGBoost (final pipeline) | **0.1654** | 0.1673 |

- **Decision:** Adopt option (b) `TargetEncoder` in the production pipeline.
- **Reason:** The deployed model is the tuned XGBoost, and for it TargetEncoder is better (0.1654 vs 0.1673) while compressing ~120 sparse columns into one dense feature. The encoder is cross-fitted inside each training fold, so it does not leak the target. For a linear model the result reverses: one-hot is clearly better, because one coefficient per car model gives Ridge more flexibility than a single encoded column.
- **Correction:** the original entry cited "0.3732 vs 0.3785" in favour of TargetEncoder for Ridge. Those figures were typed into the code, not measured, and the measured Ridge result points the other way. The decision stands because it rests on the XGBoost result, not the Ridge one.
- **Evidence:** `reports/tables/encoder_comparison.csv` (from `run_encoder_comparison()` in `src/carprice/train.py`), `tests/test_features.py::test_default_pipeline_uses_target_encoding_for_model`.
- **Impact / how to reverse:** `build_feature_pipeline(model_encoding="onehot")`; see Open question 1 about the Ridge baseline.

---

### D-015: Rare-Brand Handling and the Brand-Tier Mapping

- **Date / Phase:** 2026-11-07 / Phase 3 (corrected 2026-11-08, see D-021)
- **Context:** The cleaned dataset features 31 automotive brands, 14 of which have $< 30$ listings.
- **Options considered:**
  - (a) Unregularized one-hot dummies for every brand.
  - (b) Group rare brands into a generic "Other" level, learned per training fold.
  - (c) Replace brand with a hand-built 5-tier hierarchy (`Budget` … `Ultra-Luxury`).
- **Decision:** Option (b): `RareCategoryEncoder(min_frequency=20)` followed by one-hot encoding, plus the binary `is_luxury_brand` flag from D-013. The 5-tier `features.brand_tier_mapping` in `config.yaml` is **not** an input to the model; it is kept only as a reference grouping for reporting and the GUI.
- **Correction:** the original entry stated that option (c) was adopted. The code never used the tier mapping, so the entry was wrong, not the model.
- **Two thresholds, two purposes:** `eda.rare_brand_report_threshold: 30` is the descriptive cut-off used to list rare brands in `eda_rare_brands.csv` (Phase 2). `features.rare_category_min_frequency: 20` is the modelling threshold, applied inside each training fold so that no test rows influence which brands are kept.
- **Reason:** Fold-wise grouping avoids sparse, high-variance dummy columns while staying leak-free. A tier mapping derived from full-dataset brand medians would have let test-set prices shape a feature, which is why it is not used for modelling.
- **Evidence:** `configs/config.yaml`, `src/carprice/features.py::build_preprocessor`, `reports/tables/eda_brand_medians.csv`, `reports/tables/eda_rare_brands.csv`.
- **Impact / how to reverse:** Change `features.rare_category_min_frequency` and re-run `python tasks.py train`.

---

### D-016: Algorithm Shortlist and Justification

- **Date / Phase:** 2026-11-08 / Phase 4
- **Context:** Supervised regression on 12,195 tabular training records with mixed numeric/categorical features, high-cardinality model categories, and right-skewed selling prices.
- **Options considered:** Comparing only linear models vs only tree ensembles vs a diverse, justified hierarchy of regressors.
- **Decision:** Shortlist 10 candidate models spanning baseline, linear, instance-based, single-tree, bagging, and boosting paradigms:
  1. `dummy`: Non-parametric median baseline to establish minimum acceptable performance.
  2. `linear_regression`: Ordinary Least Squares to gauge baseline linear explanatory power.
  3. `ridge`: L2-regularized linear model to stabilize collinear feature weights.
  4. `lasso`: L1-regularized linear model to perform embedded sparse feature selection.
  5. `k_nearest_neighbors`: Distance-weighted instance contrast model.
  6. `decision_tree`: Single CART tree baseline for interpretability and overfitting contrast.
  7. `random_forest`: Bagging ensemble to reduce tree variance and capture non-linear interactions.
  8. `hist_gradient_boosting`: Fast histogram-binned boosting algorithm.
  9. `xgboost`: Scaled second-order gradient boosted trees with L1/L2 shrinkage.
  10. `lightgbm`: Leaf-wise gradient tree boosting optimized for speed and tabular accuracy.
- **Reason:** Establishes an academically rigorous benchmark across inductive biases, proving whether complex non-linear tree ensembles justify added computational cost over regularized linear baselines.
- **Evidence:** `reports/tables/model_candidates.csv`, `reports/tables/cv_results.csv`, `reports/figures/model_cv_comparison.png`.
- **Impact / how to reverse:** Configured in `configs/config.yaml` under `models.candidates`; untuned settings for each model are in `models.candidate_params`.

---

### D-017: Target Transformation Architecture (TransformedTargetRegressor)

- **Date / Phase:** 2026-11-08 / Phase 4
- **Context:** Car selling prices exhibit extreme right-skewness and heteroscedastic error dispersion. On the cleaned data used for EDA and modelling the skewness is 10.11, falling to 0.56 on the log1p scale (D-009, `reports/figures/eda_01_selling_price_raw_vs_log.png`). The raw 15,411-row file gives 10.05 and 0.57; the small difference comes from the 167 duplicates removed in D-003.
- **Options considered:**
  - (a) Direct regression on raw rupee prices with MSE loss (causes extreme gradient instability on luxury supercars).
  - (b) Ad-hoc manual target logging before training and manual expm1 inverse transformation at inference.
  - (c) Scikit-learn's `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` wrapping the complete feature preparation and estimator pipeline.
- **Decision:** Adopt option (c).
- **Reason:** Guarantees leak-free encapsulation and a clean API contract. Calling `.fit(X, y)` and `.predict(X)` operates directly on human-interpretable INR prices while internal loss minimization occurs on the symmetric, variance-stabilized log scale.
- **Evidence:** `src/carprice/train.py`, `models/model_card.json`.
- **Impact / how to reverse:** Configured in `fit_and_persist_pipelines()` in `src/carprice/train.py`.

---

### D-018: Hyperparameter Tuning Budget, Search Spaces, and Fold Precomputation

- **Date / Phase:** 2026-11-08 / Phase 4
- **Context:** Tuning top candidate algorithms across 5-fold cross-validation without risking out-of-fold data leakage or excessive latency.
- **Options considered:**
  - (a) Grid search (computationally intractable).
  - (b) RandomizedSearchCV.
  - (c) Optuna Tree-structured Parzen Estimator (TPE) Bayesian optimization with 35 trials per candidate, fixed seed (42), and precomputed leak-free fold features.
- **Decision:** Adopt option (c) across the top candidate models (`xgboost`, `lightgbm`, `hist_gradient_boosting`).
- **Reason:** TPE explores complex non-linear parameter interactions far more efficiently than random guessing. Precomputing fold transforms ($X_{tr, k}, X_{val, k}$) strictly on each training fold once reduces tuning latency by $> 85\%$ while remaining 100% leak-free and mathematically identical to in-loop pipeline fits.
- **Evidence:** `reports/tables/tuning_trials_*.csv`, `models/best_params.json`, `reports/figures/tuning_history.png`, `reports/figures/tuning_param_importances.png`.
- **Impact / how to reverse:** Configured in `configs/config.yaml` (`models.tuning_trials: 35`, `models.timeout_seconds`, and the per-model ranges under `models.search_spaces`). Key order in each search space is the order Optuna suggests parameters, so it must not be re-sorted if trials are to be reproduced.

---

### D-019: Final Model Selection Rule and Winning Estimator

- **Date / Phase:** 2026-11-08 / Phase 4
- **Context:** Identifying the winning production model from tuned candidates strictly using training-set CV metrics before touching the locked test set.
- **Decision Rule:** Select the candidate achieving lowest mean 5-fold CV log RMSE; tie-breaker: smaller fold standard deviation, lower model complexity, and faster inference latency.
- **Outcome:** Tuned `xgboost` achieved the lowest 5-fold CV log RMSE of **0.1654** (OOF $R^2 = 0.9426$), edging out `lightgbm` (0.1660) and `hist_gradient_boosting` (0.1660) (`models/best_params.json`). The margin (0.0006) is well inside one fold standard deviation (about 0.005 in `cv_results.csv`), so the three boosters are effectively tied.
- **Superiority over baselines:**
  - 76.1% lower RMSE than DummyRegressor (0.6921 $\rightarrow$ 0.1654), `model_card.json` → `improvement_vs_dummy_pct`.
  - 13.06% lower RMSE than the Ridge linear baseline (0.1903 $\rightarrow$ 0.1654), `model_card.json` → `improvement_vs_ridge_pct`. (An earlier version said 13.08%, computed from rounded figures; the card uses the unrounded OOF RMSE.)
  - Untuned out-of-fold INR MAE: Ridge ₹1,20,518 vs untuned XGBoost ₹97,642 (`cv_results.csv`).
- **Evidence:** `reports/tables/cv_results.csv`, `models/best_params.json`, `models/model_card.json`.
- **Impact / how to reverse:** Governed by `fit_and_persist_pipelines()` in `src/carprice/train.py`.

---

### D-020: Empirical Out-of-Fold Residual Quantile Method for 80% Prediction Intervals

- **Date / Phase:** 2026-11-08 / Phase 4
- **Context:** Academic requirement 4 requires providing a plausible prediction interval for estimated used car values.
- **Options considered:**
  - (a) Gaussian parametric assumption ($\hat{y} \pm 1.28 \times \text{RMSE}$).
  - (b) Empirical out-of-fold residual quantiles (10th and 90th percentiles) on log1p scale: $[\text{expm1}(\hat{y}_{\log} + q_{10}), \text{expm1}(\hat{y}_{\log} + q_{90})]$.
- **Decision:** Adopt option (b) with $q_{10} = -0.1903$ and $q_{90} = +0.1915$.
- **Reason:** Residuals on used car valuations exhibit slight asymmetry due to seller negotiation bands. Empirical out-of-fold quantiles make zero normality assumptions and directly reflect real-world validation coverage.
- **Evidence:** `models/model_card.json`.
- **Impact / how to reverse:** Documented in model card and consumed by `predict.py` and GUI in Phases 5 & 6.

---

### D-021: Phase 4 Consistency Review and Corrections

- **Date / Phase:** 2026-11-08 / Phase 4 review
- **Context:** A review of Phases 0 to 4 against AGENTS.md found places where the decision log, config and code disagreed.
- **Problems found and fixes:**
  1. *Typed-in evidence.* `evaluate_feature_candidates()` measured only the baseline; every candidate row of `feature_checks.csv` was a typed-in number, against the A4 rule "do not invent numbers". **Fix:** the function now measures every row. Four of the six original numbers were reproduced exactly; `is_luxury_brand` (true Δ −0.0011, was −0.0006) and the combined set (0.1903, was 0.1910) were wrong, and the encoder figures were not measurements at all (D-013, D-014 corrected).
  2. *Untested proposal.* `power_x_automatic` (D-010) was never evaluated. **Fix:** now measured and rejected (D-013).
  3. *Config that disagreed with the code.* `splitting.stratify_by` was `null` while the code stratified on price quintiles; `rare_brand_threshold: 30` and `rare_fuel_threshold: 40` were unused while the code hard-coded 20 and 40; `min_price`, `max_price`, `min_year`, `max_year` were unused (the data has no year column, and the real maximum price, ₹3.95 crore, exceeded `max_price`). **Fix:** every threshold the code uses is now read from config with the same value; unused keys were removed.
  4. *Hard-coded model grids.* Candidate hyperparameters and Optuna search spaces were literals in `train.py` (A2 requires them in config), with three copy-pasted tuning functions. **Fix:** moved to `models.candidate_params` and `models.search_spaces`; one `tune_model_study()` replaces the three copies.
  5. *Wrong figure.* `tuning_param_importances.png` showed LightGBM's study although XGBoost won. **Fix:** the figure now follows the winning study. The tuning figures and the CV comparison now have titles that state the finding (A7).
  6. *Provenance.* `model_card.json` recorded the Phase 3 commit because training ran before Phase 4 was committed. **Fix:** the hash gets a `-dirty` suffix when `src/` or `configs/` have uncommitted changes, and the model was refitted from a clean commit.
  7. *Fuel-label check.* The LPG exemption in `fix_fuel_labels()` compared against `"LPG"`, but labels are title-cased to `"Lpg"`, so it never matched (harmless: LPG has 44 rows, above the 40 threshold). **Fix:** exemption removed.
  8. *Reproduction.* `python tasks.py train` was a placeholder, and Phases 2 and 3 had no command. **Fix:** `tasks.py eda`, `split` and `train` now run the real pipelines; the README lists them in order.
- **Verification ("without affecting previous work"):** after the changes, Phases 1, 3 and 4 were re-run from scratch and compared with copies taken beforehand. `cars_clean.csv`, all audit tables, `cleaning_log.csv`, `train.csv`, `test.csv` and `split_indices.json` are byte-identical. `best_params.json`, all scores in `cv_results.csv`, every Optuna trial's parameters and value, and every metric in `model_card.json` are identical; only timings and timestamps differ. The refitted `final_pipeline.joblib` gives the same predictions as the pre-review pipeline on all 12,195 training rows.
- **Evidence:** commit history, `reports/tables/feature_checks.csv`, `reports/tables/encoder_comparison.csv`, new tests in `tests/test_config.py`, `tests/test_features.py`, `tests/test_train.py`.
- **Impact / how to reverse:** `git revert` the review commits; no model or data changes are involved.

---

### D-022: Pre-Split Steps That Use the Whole Dataset (Leakage Risk Assessment)

- **Date / Phase:** 2026-11-08 / Phase 4 review
- **Context:** A4 forbids computing statistics on the full dataset before splitting. Three earlier choices touch the full dataset, so each was checked.
- **Findings:**
  1. *Seat imputation (D-006):* the brand/model modal seat count is computed on all 15,244 rows before the split. It affects 2 rows, uses no price information, and the imputed value (5) is a published manufacturer spec for both cars. **Risk: negligible.** Left as is.
  2. *Brand-tier mapping (D-010/D-015):* derived from full-dataset brand median prices, so it **would** leak test prices, but it is not used by the model pipeline (D-015 correction). **Risk: none for the model**; reporting/GUI only.
  3. *`luxury_brands` list (D-013):* a fixed list of marques chosen from domain knowledge (premium European/Japanese luxury makers). It is not fitted from prices. **Risk: low.** One could argue the EDA price ranking informed the list; this is noted for the viva.
  4. *Everything fitted from data inside the model* (imputer, scaler, rare-category grouping, one-hot, TargetEncoder) sits inside the Pipeline and is fitted per training fold. `load_training_data()` refuses to read `test.csv`, and `tests/test_train.py::test_train_module_never_reads_test_csv` checks it.
- **Decision:** No change to the trained model. Findings recorded here for transparency.
- **Evidence:** `src/carprice/data.py::fix_invalid_seats`, `src/carprice/features.py`, `configs/config.yaml`.
- **Impact / how to reverse:** To remove finding 1 entirely, move seat imputation into the Pipeline as a fitted transformer and retrain; expected effect on metrics is nil (2 rows).

---

### D-023: Evaluation Metrics and Headline Metric (written before opening test.csv)

- **Date / Phase:** 2026-11-08 / Phase 5, task 5.1
- **Context:** The test set (`data/processed/test.csv`, 3,049 rows) is scored once. The metric set must be fixed before any test number is seen, so results cannot steer which metrics get reported.
- **Metrics, and why each is reported (final model, Ridge baseline, dummy baseline):**
  1. *MAE and RMSE in rupees.* The error a buyer or seller actually feels. MAE is the average miss and is robust to the luxury tail. RMSE squares errors, so a few large misses on ₹50 lakh+ cars dominate it; reporting both shows how much of the error sits in that tail.
  2. *MAE and RMSE on log price.* `log1p(price)` is what the model was trained to minimise (D-009, D-017), so these are the scores directly comparable with the training CV figures in `model_card.json` and `cv_results.csv`. A log error of 0.10 is roughly a 10% miss at any price level.
  3. *R² on both scales, plus adjusted R².* Share of price variance explained. Log-scale R² matches training; rupee-scale R² is dominated by expensive cars. Adjusted R² penalises predictor count with p = number of columns entering the estimator after preprocessing (39 for both Ridge and XGBoost; 0 for the dummy). With n = 3,049, p = 39 the adjustment is small; it is reported because the brief asks for it, not because it changes conclusions.
  4. *MAPE and median absolute percentage error (MdAPE).* Relative error is how people judge car prices ("it was 8% off"). MAPE is the mean and is pulled up by cheap cars, where a ₹40,000 miss on a ₹1.5 lakh car is a 27% error. MdAPE is the typical car's error and is robust to that.
  5. *Share of predictions within 10% and within 20% of the true price.* The easiest accuracy figure to explain in the GUI and the viva ("N in 10 cars are priced within 10% of the true price").
  6. *80% interval coverage.* Share of test cars whose true price falls inside the interval from D-020, with a 95% binomial (Wilson) confidence interval. **Verdict rule:** "well calibrated" if the nominal 80% lies inside that confidence interval, otherwise "over-confident" (coverage too low) or "conservative" (too high).
- **Headline metric:** **MAE in rupees together with the within-10% share.** MAE answers "how many rupees off, on average", in the unit the user cares about. Within-10% answers "how often is it close", is independent of price level, and is not distorted by cheap cars the way MAPE is. Log RMSE stays the *model-selection* metric (D-019) and is not re-optimised on the test set.
- **Uncertainty:** 1,000 bootstrap resamples of the test rows (seed 42), percentile 95% intervals for MAE (₹), within-10%, within-20%, MdAPE, log RMSE and R² (log), and for the MAE difference between the final model and Ridge.
- **Segment ranking:** the "worst" segments are ranked by **lowest within-10% share** among segments with at least `evaluation.min_segment_rows` (30) test rows. Ranking by rupee MAE would simply list the most expensive segments, which is a property of price level, not model weakness; the row minimum stops tiny segments from topping the list by chance.
- **Dummy baseline:** `DummyRegressor(strategy="median")` on log price, fitted on `train.csv` only, matching the dummy in the Phase 4 benchmark. It is not saved as a file, because it has no learned parameters beyond one median.
- **Evidence:** `configs/config.yaml` → `evaluation`, `src/carprice/evaluate.py`.
- **Impact / how to reverse:** change `evaluation.*` and re-run `python tasks.py evaluate`; any change after results are seen must be logged as post-hoc (D-025).

---

### D-024: `km_was_capped` Flag Is Not Used by the Model (found before test evaluation)

- **Date / Phase:** 2026-11-08 / Phase 5, before opening test.csv
- **Context:** D-004 added `km_was_capped` so that "trees can learn if it matters". Inspecting the saved pipeline shows the `ColumnTransformer` lists no such column and uses `remainder="drop"`, so the flag is silently discarded. The 39 columns the estimator sees are listed by `final_pipeline.regressor_.named_steps['features'].named_steps['preprocess'].get_feature_names_out()`.
- **Options considered:** (a) add the flag to the numeric columns and retrain before Phase 5; (b) keep the model as is and record the gap.
- **Decision:** (b). The model is not changed.
- **Reason:** The flag is set on 12 of 15,244 rows (0.08%, `cleaning_log.csv`), and capped km already limits their leverage (D-004). Retraining would reopen Phase 4 for a negligible effect. Because this was found before the test set was opened, the choice is not influenced by test results. Permutation importance in task 5.3 will show it as exactly zero, which confirms the finding.
- **Correction to D-004:** the cap is applied; the flag is kept in the data but is not a model input.
- **Impact / how to reverse:** add `"km_was_capped"` to `NUMERIC_FEATURES` in `src/carprice/features.py`, then re-run `python tasks.py train`. Doing so after this phase would be a post-hoc change and would contaminate the test set (D-025).

---

### D-025: One-Time Test-Set Scoring and Post-Hoc Change Policy

- **Date / Phase:** 2026-11-08 / Phase 5, task 5.2
- **Context:** The brief requires the locked test set to be used once for the final numbers.
- **What happened:** `test.csv` was read for the first time by `run_test_evaluation()` at code commit `0001bf0`, after the metric set (D-023) and the `km_was_capped` finding (D-024) had been committed. All three models were scored in that single pass; every table under `reports/tables/` named `test_*`, `interval_coverage`, `residual_stats`, `segment_errors`, `top_error*`, `bootstrap_ci` and `train_vs_test` comes from it.
- **Post-hoc changes:** **none.** No model, feature, hyperparameter, interval quantile or metric definition was changed after the test results were seen. Re-running `python tasks.py evaluate` recomputes the same numbers from the same saved pipelines (seeded bootstrap and permutation), which is a reproduction, not a second look used for selection.
- **Rule going forward:** if any model change is made after this point (for example acting on D-024 or Open question 1), the current test results must be labelled contaminated in this log, and any new test numbers reported as optimistic.
- **Evidence:** git history (`0001bf0` → this commit), `reports/tables/test_metrics.csv`.
- **Impact / how to reverse:** not reversible by design; the test set is now spent for model selection purposes.

---

### D-026: Known Limitations of the Final Model (from the test evaluation)

- **Date / Phase:** 2026-11-08 / Phase 5, task 5.4
- **Context:** The brief asks for the model's known limitations to be recorded. Each item below is backed by `reports/tables/segment_errors.csv`, `top_errors.csv`, `top_error_patterns.csv` or `interval_coverage.csv`.
- **Limitations:**
  1. *Old cars (9+ years):* within-10% share 33.92% (9-11 yrs) and 32.74% (12+ yrs), against 50.97% overall. Condition is not recorded, and it matters most for old cars.
  2. *Luxury cars:* 41.18% within 10% above ₹20 lakh, under-priced by a median 5.76%. 17 of the 20 largest errors are in this band. Coarse `model` labels (e.g. "Rover", "C") hide trim and variant.
  3. *Cheapest cars:* the lowest training quintile is over-priced by a median 7.26%.
  4. *Interval calibrated only on average:* overall coverage is 79.96% (well calibrated), but 92.45% for 0-2 year-old cars and 52.38% for 12+ years.
  5. *Rare categories:* LPG (11 test cars) and brands outside the top 10 cannot be evaluated reliably.
  6. *Data scope:* listing prices from one platform and one period; no city, condition, accident or ownership history.
  7. *Unused input:* `km_was_capped` (D-024).
- **Decision:** No change to the model in this phase (D-025). These limitations will be shown as warnings in the GUI (Phase 6) whenever a user's input falls in a weak segment.
- **Evidence:** `reports/evaluation_summary.md`, `notebooks/03_model_evaluation.ipynb` sections 6, 7 and 9.
- **Impact / how to reverse:** addressing items 1 to 4 needs richer data or age-dependent intervals, followed by a fresh hold-out set.

---

### D-027: Phase 5 Implementation Choices

- **Date / Phase:** 2026-11-08 / Phase 5
- **Choices:**
  1. *New module `src/carprice/eval_plots.py`* (not a new folder). It holds the evaluation and SHAP figures so `evaluate.py` stays focused on numbers. It reuses the palette and `save_figure()` from `viz.py`. `format_inr_indian()` and `format_lakh()` were added to `viz.py` for the A7 number formats and will be reused by the GUI.
  2. *Price bands use training-set quintile edges*, so test statistics never define segment boundaries. The luxury band uses the fixed ₹20 lakh threshold from the brief (`evaluation.luxury_threshold_inr`).
  3. *Adjusted R²* uses p = columns entering the estimator after preprocessing (`count_estimator_inputs()`), 0 for the dummy.
  4. *Permutation importance* is computed on raw input columns with two scorers: rupee MAE (headline) and log RMSE (training objective). The figure and ranking use log RMSE, because rupee MAE is dominated by expensive cars.
  5. *SHAP* uses `TreeExplainer` on the fitted XGBoost with a seeded sample of 2,000 test rows. One-hot columns are summed back to their input feature so `brand`, `fuel_type`, `transmission_type` and `seller_type` each get one score. Dependence plots show the effect as a % change in predicted price, `expm1(SHAP) × 100`, which is easier to read than log units.
  6. *SHAP colour map:* slate → teal for low → high feature values, instead of SHAP's default red-blue, to follow A7 (no rainbow or off-palette colours).
  7. *`reports/tables/test_predictions.csv`* stores every test prediction with its segment labels, so the GUI's Model Performance page can plot results without loading the test set or the model.
  8. *Notebook tooling pinned:* `ipykernel`, `nbclient`, `nbformat` were already installed (used to run notebooks 01 to 03). They are now pinned in `requirements.txt`; nothing new was installed.
  9. *Number provenance is tested:* `tests/test_evaluate.py` extracts every number from `reports/evaluation_summary.md` and from notebook 03's markdown, and fails if any is absent from `reports/tables/`, `model_card.json` or `config.yaml`. A companion test proves it flags an invented figure.
- **Impact / how to reverse:** settings under `evaluation:` in `config.yaml`; re-run `python tasks.py evaluate`.

---

### D-028: Resolution of the Phase 4 and Phase 5 Open Questions

- **Date / Phase:** 2026-11-08 / after Phase 5 (student answers)
- **Rule given by the student:** choose the option with less work unless it significantly affects final accuracy.
- **Resolutions:**
  1. *Ridge baseline encoding (D-014):* **keep as is (TargetEncoder).** The Ridge model is only a baseline; switching it to one-hot would make Ridge look somewhat better but cannot change the final model's accuracy. Changing it now would also be a post-hoc change after test scoring (D-025).
  2. *Brand-tier mapping (D-015):* **keep** in `config.yaml` as a reporting and GUI grouping (for example the Market Explorer page). It stays out of the model pipeline.
  3. *Seat imputation before the split (D-022, finding 1):* **keep as is.** It touches 2 of 15,244 rows and uses no price information, so the effect on accuracy is nil. Recorded as an accepted minor deviation from A4.
  4. *`km_was_capped` not used by the model (D-024):* **keep the model unchanged.** The flag affects 12 rows; retraining would contaminate the test set (D-025) for no meaningful accuracy gain. Permutation importance already shows the flag carries no information to the model.
  5. *Phase 2 numbers that did not match their tables:* **corrected, text only.** On checking, the correlation figures in `reports/eda_findings.md` and the notebook 02 markdown did not match `eda_spearman_correlation.csv` / `eda_pearson_correlation.csv` on any version of the data (raw, cleaned or train). They are now: Spearman ρ with price 0.72 for `max_power` (was 0.69), 0.64 for `engine` (was 0.58), −0.48 for `vehicle_age` (was −0.60), −0.17 for `km_driven` (was −0.37, now described as weak); Pearson r with log price 0.76 for `max_power` (was 0.75); `engine`–`max_power` r 0.81 and ρ 0.83 (were 0.78 and 0.79). The same edit repaired five `\rho` symbols in notebook 02 that had been saved as a carriage return plus "ho" and rendered as broken text. No code, figure, table or model output changed. Fixing was chosen over leaving it because it took minutes and the final report will quote these figures. A full audit of every other number in Phase 2 was not done.
  6. *GUI warnings (D-026):* **yes.** Phase 6 will show a plain-language caution beside the estimate when the input car is 9+ years old, is a luxury brand or would sell for more than ₹20 lakh, or uses a brand, model or fuel with little data.
- **Effect on accuracy:** none. The final model, its test results (D-025) and all Phase 3 to 5 tables are unchanged.

---

### D-029: Prediction Input Validation Rules

- **Date / Phase:** 2026-11-08 / Phase 6, tasks 6.1 to 6.3
- **Context:** The predictor must refuse inputs it cannot price sensibly and explain why in words a GUI user understands. Tree models cannot extrapolate: outside the training range they just repeat the nearest leaf, so a price for a 40-year-old car would look precise but mean nothing.
- **Where the allowed values come from:** `models/input_schema.json`, built by `build_input_schema()` from `data/processed/train.csv` **only** (12,195 rows). It records the brand → model pairs with row counts, the allowed levels of seller, fuel and transmission with counts, numeric min and max, and the allowed seat counts. Test data never shapes the rules. Rebuild with `python -m carprice.predict build-schema`.
- **Options considered:** (a) hard-coded "physically plausible" bounds in config; (b) the range actually seen in training; (c) no range checks, warnings only.
- **Decision:** (b), with these rules:
  1. *Normalisation first:* labels are trimmed and title-cased exactly as Phase 1 cleaning did, so `" bmw "` → `Bmw`. Numbers may include thousands separators (`"60,000"`).
  2. *Missing or unreadable fields:* every one of the 11 CarSpec fields is required. Text where a number is expected, NaN or infinity is refused.
  3. *Whole numbers:* `vehicle_age`, `km_driven`, `engine` and `seats` must be integers (4.5 years is refused, not rounded).
  4. *Numeric ranges:* `vehicle_age` 0 to 29, `km_driven` 100 to 5,00,000, `mileage` 4 to 33.54 kmpl, `engine` 793 to 6,592 cc, `max_power` 38.4 to 626 bhp: the minimum and maximum in train.csv. `seats` must be one of 2, 4, 5, 6, 7, 8, 9.
  5. *Categories:* seller `Dealer`, `Individual`, `Trustmark Dealer`; fuel `Petrol`, `Diesel`, `Cng`, `Lpg`; transmission `Manual`, `Automatic`. Anything else (for example `Electric`) is refused with the list of valid choices.
  6. *Brand and model:* see D-030.
  7. *All problems at once:* parsing and rule checks run on every field that could be read, so a user sees every problem in a single message instead of fixing them one at a time.
- **How errors surface:** `InputError`, a subclass of `ValueError`, carries `.problems` (a list of sentences) and a combined message starting "Please check the car details:". The CLI prints these sentences and exits with code 2, never a traceback. The GUI can show `.problems` as a bullet list.
- **Cautions are not errors:** a valid car in a weak segment (D-026) is priced but gets plain-language cautions in `PredictionResult.cautions`. They fire for age ≥ 9 years (`prediction.old_car_age_years`), a luxury brand or a predicted price above ₹20 lakh, a brand/model with fewer than 50 training rows (`prediction.rare_training_rows`), and a fuel with fewer than 50 training rows (LPG has 33). This implements D-028 item 6 at the logic level; the GUI phase will display them. (D-028 said "Phase 6 will show"; the display belongs to the GUI phase, and the logic is in place now.)
- **Evidence:** `src/carprice/predict.py` (`parse_fields`, `validate_values`, `caution_messages`), `models/input_schema.json`, `tests/test_predict.py`.
- **Impact / how to reverse:** edit `build_input_schema()` or the `prediction:` block of `config.yaml`, then rebuild the schema.

---

### D-030: Enforcing Brand/Model Consistency

- **Date / Phase:** 2026-11-08 / Phase 6, task 6.3
- **Context:** In the cleaned data every one of the 119 model names belongs to exactly one of 30 brands (`eda_model_cross_brand.csv` is empty), so a model name determines its brand. The pipeline encodes `model` with a TargetEncoder that silently maps unknown names to the global mean, which would hide a typo behind a plausible-looking price.
- **Rule:** the brand must be a training brand, and the model must appear under **that** brand in `input_schema.json`. Three distinct messages:
  - unknown brand: "Brand 'Tesla' is not in the training data. Known brands: …";
  - model of another brand: "Model 'Swift' is made by Maruti, not Hyundai." (detected by looking the model up across all brands);
  - unknown model: "Model 'Zen Estilo X' is not a known Maruti model. Known Maruti models: …" (up to 12 listed, then "and N more").
- **Options considered:** (a) accept unknown models and let the encoder fall back to the average; (b) fuzzy-match close spellings; (c) strict lookup against training pairs.
- **Decision:** (c). (a) produces confident nonsense for typos; (b) adds a dependency and guesswork. The GUI phase will offer the model list as a dropdown filtered by brand, so strict lookup costs users nothing.
- **Known consequence:** one test car, a Mercedes-AMG C (brand absent from training), would be refused by the GUI. It is still scored in `reports/predictions/test_predictions.csv` with `predict_batch(validate=False)` and flagged `passes_validation = False`, so the file covers every test row (D-031).
- **Evidence:** `brand_model_problems()` in `src/carprice/predict.py`; tests in `tests/test_predict.py`.
- **Impact / how to reverse:** relax `brand_model_problems()`; not recommended.

---

## Open questions for the student

- None open. Questions raised in Phases 4 and 5 are resolved in D-028.
