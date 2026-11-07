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
- **Impact / how to reverse:** Configured in `configs/config.yaml` (`splitting.test_size: 0.2`, `project.random_seed: 42`) and managed by `create_train_test_split()`.

---

### D-013: Candidate Feature Engineering Evaluations and Selection Decisions

- **Date / Phase:** 2026-11-07 / Phase 3
- **Context:** EDA proposed candidate domain features. Each candidate was independently evaluated via 5-fold cross-validation on `train.csv` only against a baseline Ridge regression model (Baseline CV RMSE: 0.2020):
  - *Candidate 1 (`km_per_year`):* $\text{km\_driven} / \max(\text{vehicle\_age}, 1)$ $\rightarrow$ RMSE: 0.2018 ($\Delta = -0.0002$) $\rightarrow$ **KEEP**.
  - *Candidate 2 (`power_per_litre`):* $\text{max\_power} / (\text{engine} / 1000)$ $\rightarrow$ RMSE: 0.2016 ($\Delta = -0.0005$) $\rightarrow$ **KEEP**.
  - *Candidate 3 (`log_km_driven`):* $\log(1 + \text{km\_driven})$ $\rightarrow$ RMSE: 0.2025 ($\Delta = +0.0004$) $\rightarrow$ **REJECT**. Degrades linear scores when km is already capped at 500,000 km.
  - *Candidate 4 (`is_luxury_brand`):* Binary indicator for luxury/ultra-luxury marques $\rightarrow$ RMSE: 0.2014 ($\Delta = -0.0006$) $\rightarrow$ **KEEP**.
  - *Candidate 5 (`age_x_power`):* Interaction between vehicle age and engine power $\rightarrow$ RMSE: 0.1925 ($\Delta = -0.0095$) $\rightarrow$ **KEEP**. Substantially benefits linear baselines by capturing accelerated power depreciation.
  - *Combined Kept Features:* Drops overall Ridge CV RMSE from 0.2020 down to **0.1910** ($\Delta = -0.0110$).
- **Decision:** Retain `km_per_year`, `power_per_litre`, `is_luxury_brand`, and `age_x_power`; reject `log_km_driven`.
- **Reason:** Every retained feature demonstrably improves cross-validation accuracy on the training partition and reflects physical domain validity.
- **Evidence:** `reports/tables/feature_checks.csv`.
- **Impact / how to reverse:** Implemented in `DomainFeatureEngineer` inside `src/carprice/features.py`.

---

### D-014: High-Cardinality Model Feature Encoding (Cross-Validated TargetEncoder)

- **Date / Phase:** 2026-11-07 / Phase 3
- **Context:** The `model` feature contains ~120 unique categorical levels, posing a risk of dimensionality explosion and sparse splits if one-hot encoded.
- **Options considered:**
  - (a) One-Hot Encoding with minimum frequency grouping (`min_frequency=10`).
  - (b) Scikit-learn's `TargetEncoder` with internal 5-fold cross-fitting (`smooth="auto"`).
- **Decision:** Adopt option (b) `TargetEncoder`.
- **Reason:** 5-fold cross-validation on `train.csv` demonstrated that TargetEncoder achieved superior predictive accuracy (Ridge CV RMSE: 0.3732) compared to One-Hot Encoding (RMSE: 0.3785) while compressing 120 sparse dummy columns into a single informative continuous feature. Target encoding is fit strictly within training folds, preventing target leakage.
- **Evidence:** `reports/tables/feature_checks.csv`, `tests/test_features.py`.
- **Impact / how to reverse:** Configured in `build_preprocessor()` inside `src/carprice/features.py`.

---

### D-015: Brand-Tier Mapping and Equity Hierarchy

- **Date / Phase:** 2026-11-07 / Phase 3
- **Context:** The cleaned dataset features 31 automotive brands, 14 of which have $< 30$ listings.
- **Options considered:**
  - (a) Rely entirely on unregularized one-hot dummy variables.
  - (b) Group all rare brands under a generic "Other" category.
  - (c) Establish an auditable 5-tier brand hierarchy (`Budget`, `Mid`, `Premium`, `Luxury`, `Ultra-Luxury`) in `configs/config.yaml` combined with a `RareCategoryEncoder(min_frequency=20)`.
- **Decision:** Adopt option (c).
- **Reason:** Phase 2 EDA confirmed that brand medians stratify into well-defined valuation bands. Grouping infrequent brands into tiers preserves high-level equity context without creating high-variance sparse columns.
- **Evidence:** `configs/config.yaml`, `reports/tables/eda_brand_medians.csv`, `reports/tables/eda_rare_brands.csv`.
- **Impact / how to reverse:** Configured in `configs/config.yaml` under `features.brand_tier_mapping` and `features.luxury_brands`.

---

## Open questions for the student

- None currently for Phase 0.

