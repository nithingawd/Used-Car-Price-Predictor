# Decision Log (decisions.md)

This log records every non-trivial design, modeling, cleaning, and architectural choice made during the development of the Used Car Price Prediction System. It serves as auditable evidence for academic grading, viva defence, and reproducibility.

## Contents

Entries are numbered D-001 to D-044 in the order they were made; each heading is `### D-NNN: title`. Corrections to earlier entries are added as dated "Revision" or "Correction" lines inside the original entry, so the history stays visible.

| Id | Phase | One-line summary |
|---|---|---|
| [D-001](#d-001) | 0 | Fixed tech stack and pinned environment |
| [D-002](#d-002) | 0 | Modular repository; raw data read-only |
| [D-003](#d-003) | 1 | Drop 167 exact duplicates; keep same-spec rows with different prices |
| [D-004](#d-004) | 1 | Cap km_driven at 5,00,000 with a flag |
| [D-005](#d-005) | 1 | Recode four 'Electric' Camry hybrids to Petrol |
| [D-006](#d-006) | 1 | Impute two zero-seat rows from the brand/model mode |
| [D-007](#d-007) | 1 | Keep genuine luxury cars; evaluate by segment |
| [D-008](#d-008) | 1 | Drop identifier columns; validate schema at each stage |
| [D-009](#d-009) | 2 | Model log1p(price) |
| [D-010](#d-010) | 2 | Feature proposals from the EDA (measured later in D-013) |
| [D-011](#d-011) | 2 | Non-parametric hypothesis tests with effect sizes |
| [D-012](#d-012) | 3 | Stratified 80/20 split on price quintiles, seed 42 |
| [D-013](#d-013) | 3 | Keep four engineered features by measured CV gain |
| [D-014](#d-014) | 3 | Target-encode `model` (better for XGBoost) |
| [D-015](#d-015) | 3 | Group rare brands per fold; brand tiers for reporting only |
| [D-016](#d-016) | 4 | Ten-algorithm shortlist |
| [D-017](#d-017) | 4 | TransformedTargetRegressor around the whole pipeline |
| [D-018](#d-018) | 4 | Optuna TPE, 35 trials per booster, precomputed folds |
| [D-019](#d-019) | 4 | Lowest CV log RMSE wins: tuned XGBoost |
| [D-020](#d-020) | 4 | 80% interval from out-of-fold residual quantiles |
| [D-021](#d-021) | 4 | Phase 4 consistency review and corrections |
| [D-022](#d-022) | 4 | Pre-split steps assessed for leakage |
| [D-023](#d-023) | 5 | Metrics and headline fixed before test scoring |
| [D-024](#d-024) | 5 | `km_was_capped` is not a model input; left unchanged |
| [D-025](#d-025) | 5 | Test set scored once; no post-hoc changes |
| [D-026](#d-026) | 5 | Known limitations from the test evaluation |
| [D-027](#d-027) | 5 | Phase 5 implementation choices |
| [D-028](#d-028) | 5 | Resolution of the Phase 4 and 5 open questions |
| [D-029](#d-029) | 6 | Prediction input validation rules |
| [D-030](#d-030) | 6 | Strict brand/model consistency |
| [D-031](#d-031) | 6 | Prediction outputs, worked-example rules, timing method |
| [D-032](#d-032) | 7 | GUI page structure and architecture |
| [D-033](#d-033) | 7 | Native TreeSHAP in the app; response time |
| [D-034](#d-034) | 7 | Rare and luxury cars in the UI |
| [D-035](#d-035) | 7 | Design system implementation and deviations |
| [D-036](#d-036) | 7 | Dark mode |
| [D-037](#d-037) | 8 | Phase 8 reproducibility audit and build fixes |
| [D-038](#d-038) | 8 | EDA notebook display fix; audit of Phase 1 and 2 figures |
| [D-039](#d-039) | 8 | Number provenance checks and derived-figure table |
| [D-040](#d-040) | 8 | Report export toolchain |
| [D-041](#d-041) | 8 | Final model and why |
| [D-042](#d-042) | 8 | Final list of known limitations |
| [D-043](#d-043) | 8 | Scope not delivered, and why |
| [D-044](#d-044) | 8 | Resolution of the remaining open questions |

---

## Assumptions

1. The target prediction task is regression: estimating continuous used car selling prices in Indian Rupees (INR) alongside plausible prediction intervals.
2. The raw dataset (`data/raw/cardekho_dataset.csv`, 15,411 rows, 14 columns) is treated as an immutable, read-only source of truth. All data cleaning transforms produce distinct files under `data/interim/` and `data/processed/`.
3. To eliminate data leakage, the train/test split (80/20) must be executed before fitting any data-dependent transformations, encoders, scalers, or imputation models.
4. All reusable computation, validation, and pipeline definitions reside strictly in `src/carprice/`, with Jupyter notebooks restricted to visual EDA and narrative reports (no standalone functions exceeding 10 lines).
5. All execution is governed by a fixed global random seed (42) loaded via `configs/config.yaml`.

---

## Decisions

<a id="d-001"></a>

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

<a id="d-002"></a>

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

<a id="d-003"></a>

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

<a id="d-004"></a>

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

<a id="d-005"></a>

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

<a id="d-006"></a>

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

<a id="d-007"></a>

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

<a id="d-008"></a>

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

<a id="d-009"></a>

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

<a id="d-010"></a>

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
- **Correction (2026-11-09, see D-038):** two figures above did not match their tables. The cleaned data have 31 brands, 13 of them with fewer than 30 rows (`report_key_figures.csv`, `eda_rare_brands.csv`), not 32 and 14. The raw median automatic/manual price ratio is 2.10 (`eda_transmission_medians.csv`), not 2.74; the premium within power quartiles is confirmed by `eda_hypothesis_tests.csv`. The decision is unaffected.
- **Revision (2026-11-08, see D-021):** these were *proposals*. Phase 3 screening (D-013) decided the final set. `power_x_automatic` was measured and rejected (ΔRMSE 0.0000). `brand_tier` was never added to the pipeline; brand information reaches the model through one-hot `brand` (rare levels grouped, D-015) and the `is_luxury_brand` flag.

---

<a id="d-011"></a>

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

<a id="d-012"></a>

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

<a id="d-013"></a>

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

<a id="d-014"></a>

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

<a id="d-015"></a>

### D-015: Rare-Brand Handling and the Brand-Tier Mapping

- **Date / Phase:** 2026-11-07 / Phase 3 (corrected 2026-11-08, see D-021)
- **Context:** The cleaned dataset features 31 automotive brands, 13 of which have $< 30$ listings (corrected from 14 in Phase 8, D-038; `eda_rare_brands.csv`).
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

<a id="d-016"></a>

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

<a id="d-017"></a>

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

<a id="d-018"></a>

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

<a id="d-019"></a>

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

<a id="d-020"></a>

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

<a id="d-021"></a>

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

<a id="d-022"></a>

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

<a id="d-023"></a>

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

<a id="d-024"></a>

### D-024: `km_was_capped` Flag Is Not Used by the Model (found before test evaluation)

- **Date / Phase:** 2026-11-08 / Phase 5, before opening test.csv
- **Context:** D-004 added `km_was_capped` so that "trees can learn if it matters". Inspecting the saved pipeline shows the `ColumnTransformer` lists no such column and uses `remainder="drop"`, so the flag is silently discarded. The 39 columns the estimator sees are listed by `final_pipeline.regressor_.named_steps['features'].named_steps['preprocess'].get_feature_names_out()`.
- **Options considered:** (a) add the flag to the numeric columns and retrain before Phase 5; (b) keep the model as is and record the gap.
- **Decision:** (b). The model is not changed.
- **Reason:** The flag is set on 12 of 15,244 rows (0.08%, `cleaning_log.csv`), and capped km already limits their leverage (D-004). Retraining would reopen Phase 4 for a negligible effect. Because this was found before the test set was opened, the choice is not influenced by test results. Permutation importance in task 5.3 will show it as exactly zero, which confirms the finding.
- **Correction to D-004:** the cap is applied; the flag is kept in the data but is not a model input.
- **Impact / how to reverse:** add `"km_was_capped"` to `NUMERIC_FEATURES` in `src/carprice/features.py`, then re-run `python tasks.py train`. Doing so after this phase would be a post-hoc change and would contaminate the test set (D-025).

---

<a id="d-025"></a>

### D-025: One-Time Test-Set Scoring and Post-Hoc Change Policy

- **Date / Phase:** 2026-11-08 / Phase 5, task 5.2
- **Context:** The brief requires the locked test set to be used once for the final numbers.
- **What happened:** `test.csv` was read for the first time by `run_test_evaluation()` at code commit `0001bf0`, after the metric set (D-023) and the `km_was_capped` finding (D-024) had been committed. All three models were scored in that single pass; every table under `reports/tables/` named `test_*`, `interval_coverage`, `residual_stats`, `segment_errors`, `top_error*`, `bootstrap_ci` and `train_vs_test` comes from it.
- **Post-hoc changes:** **none.** No model, feature, hyperparameter, interval quantile or metric definition was changed after the test results were seen. Re-running `python tasks.py evaluate` recomputes the same numbers from the same saved pipelines (seeded bootstrap and permutation), which is a reproduction, not a second look used for selection.
- **Rule going forward:** if any model change is made after this point (for example acting on D-024 or Open question 1), the current test results must be labelled contaminated in this log, and any new test numbers reported as optimistic.
- **Evidence:** git history (`0001bf0` → this commit), `reports/tables/test_metrics.csv`.
- **Impact / how to reverse:** not reversible by design; the test set is now spent for model selection purposes.

---

<a id="d-026"></a>

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

<a id="d-027"></a>

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

<a id="d-028"></a>

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

<a id="d-029"></a>

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

<a id="d-030"></a>

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

<a id="d-031"></a>

### D-031: Phase 6 Prediction Outputs, Example Selection and Timing Method

- **Date / Phase:** 2026-11-08 / Phase 6, tasks 6.5 to 6.8
- **Choices:**
  1. *New modules:* `src/carprice/insights.py` holds the per-car explanation, what-if, model-spec lookup and similar-listings helpers, shared by the reports and the GUI. `src/carprice/prediction_reports.py` writes the Phase 6 files. `src/carprice/formatting.py` holds the rupee formatters, moved out of `viz.py` so the predictor does not import seaborn (this cut import time by about 3 seconds). `explanation_frame`, `original_feature_of` and `group_shap_by_feature` moved from `evaluate.py` to `insights.py`, and `prediction_interval` moved to `predict.py`; the old modules import them, so nothing is duplicated. New folder `reports/predictions/`, as named in the brief.
  2. *Test predictions cover every row:* `reports/predictions/test_predictions.csv` is scored with `predict_batch(validate=False)`. One test car (Mercedes-AMG C, a brand absent from training) would be refused by validation (D-030); it is kept and flagged `passes_validation = False`. Predictions equal the Phase 5 `reports/tables/test_predictions.csv` to the rupee (tested), so Phase 5 numbers stand.
  3. *Worked-example selection is rule-based, not hand-picked:* eight categories (cheap petrol hatchback, family diesel SUV or MPV, 12+ year-old car, luxury car above ₹20 lakh, CNG car, automatic petrol city car, 150,000+ km car, 0-1 year-old car). In each, the chosen car is the one whose absolute % error is closest to that category's median, so the example is typical rather than flattering. Then the single best and worst predictions in the test set. Each explanation sentence is generated from the car's own TreeSHAP contributions plus the D-026 weak-segment rule that applies. Accurate predictions are explained by data coverage; close calls and misses by the weakness. Verdict thresholds: within 10% "Accurate", above 20% "Miss", otherwise "Close".
  4. *What-if reference car:* the most common model in train.csv (Maruti Swift Dzire), with its most common specs and median age and km. The grid is 12 ages × 15 km levels. The chart compares the two sweeps at the car's typical age and km rather than at the extremes, because a 1-year-old car with 150,000 km is implausible usage and the model prices it down sharply (stated in the caption).
  5. *Per-car explanation uses XGBoost's native `pred_contribs`:* this is the exact TreeSHAP algorithm. On 200 rows it matched the shap library's `TreeExplainer` with a maximum difference of 0.0, and contributions sum to the log prediction within 1e-5 (tested). It avoids importing the shap library in the GUI.
  6. *Timing method:* `reports/tables/predictor_timing.csv` is measured in 5 fresh Python processes. "Load" is the `PricePredictor()` call after the libraries it needs are imported (reading and unpickling the pipeline, card and schema, plus one warm-up prediction); library import time is reported separately, because the GUI pays it once at server start. "Single prediction" is the median of 50 calls after 10 warm-up calls (the GUI keeps one predictor alive). Pass/fail uses the fastest of the 5 runs, following the `timeit` convention that system noise can only slow a run; the slowest run is recorded beside it.
- **Timing results and an honest caveat:** on an unthrottled run during development, load was 0.53 s and a single prediction had a 30 ms median (44.6 ms p95), both passing. At the time of the final measurement the laptop was **on battery with the Windows "Silent" power plan**. The identical in-process benchmark then ran about 5× slower (140 ms against 28 ms), and a pure-Python CPU probe confirmed the throttling. Under those conditions load still passes (0.53 s best, 1.75 s worst), but the single-prediction median (133 ms best) misses the 100 ms target, and `test_timing_meets_targets` fails. The thresholds were **not** relaxed. Re-running `python tasks.py predict` on mains power with a normal power plan is expected to pass; until then this acceptance item is open.
- **Evidence:** `reports/predictions/`, `reports/figures/whatif_age_km.png`, `reports/tables/predictor_timing.csv`, `models/model_specs.csv`, `tests/test_predict.py`, `tests/test_insights.py`.
- **Impact / how to reverse:** settings in the `prediction:` block of `config.yaml`; re-run `python tasks.py predict`.

---

<a id="d-032"></a>

### D-032: GUI Page Structure and Architecture

- **Date / Phase:** 2026-11-08 / Phase 7
- **Pages (in navigation order):** Price estimator (default, `app/Home.py`, with *Single car* and *Batch estimate* tabs), Market explorer, Model performance, About and data (`app/pages/`).
- **Why this order:** the task a non-technical visitor came for (pricing a car) comes first and needs no scrolling. The market context follows for people comparing options, then the evidence for people who want to trust the number, then background.
- **Navigation:** `st.navigation(position="top")` gives a top navigation bar instead of the default sidebar. It reads like a product, leaves the full width for the 40/60 form-and-result layout, and stacks cleanly on narrow screens. `Home.py` is the entry point and router; the estimator is defined in it as the default page.
- **Code layout:** no modelling logic in the GUI. Pages call `PricePredictor` and shared helpers only:
  - `src/carprice/insights.py`: explanations, what-if, spec lookup, similar listings.
  - `src/carprice/market.py`: new; filtering and summaries for Market explorer and the compare panel, all tested.
  - `src/carprice/palette.py`: new; the single source of A7 hex values for both matplotlib and the GUI.
  - `app/components/`: `palette.py` (the only GUI colour source, plus the Plotly style), `ui.py` (`load_theme`, `page_header`, `section_header`, `metric_card`, `info_note`, `empty_state`, `chart_card`, `range_bar`, `format_inr`, `format_lakh`, `safely`), `charts.py` (all Plotly figures), `data.py` (cached loaders), `batch.py` (upload checks, testable without Streamlit) and `theme.css`.
- **Caching:** `st.cache_resource` holds the `PricePredictor`, loaded once per server process and never on interaction. `st.cache_data` holds the spec lookup, training listings, cleaned listings, report tables, the model card and library versions.
- **Pre-fill:** `models/model_specs.csv` (D-031) gives each brand/model's most common engine, power, mileage, seats, fuel and transmission. Changing the brand selects its most common model, and changing the model pre-fills those fields through `on_change` callbacks. Every field stays editable.
- **When predictions run:** on first visit the most common car in the training data is priced once, so the page is never blank. After that, prices only update when **Estimate price** is pressed. If the form changes afterwards, an "Inputs changed" pill marks the result as stale instead of silently re-pricing. The what-if panel is an `st.fragment`, so switching between Age and Kilometres redraws only that panel.
- **Layout:** the form is in the left 40%. The result card and the "why this price" and "compare" panels share the right 60%; the what-if chart spans the full width below. This balances the column heights (no ragged columns), while still placing "three small panels below the result".
- **Data sources:** Market explorer uses `data/interim/cars_clean.csv` only, as the brief says. The "similar listings" message and the compare panel use `train.csv`, the data the model learned from. The Model performance and About pages read only `reports/tables/`, `models/model_card.json` and library metadata. No metric is typed into the code; Market explorer figures are computed live from the filtered data by `market.py`.
- **Evidence:** `app/`, `tests/test_app_helpers.py` (AppTest smoke tests for all four pages, brand filtering, pre-fill and the estimate button).

---

<a id="d-033"></a>

### D-033: SHAP in the App and Response Time

- **Date / Phase:** 2026-11-08 / Phase 7
- **Method:** "Why this price" shows the top five factors for the single car, from XGBoost's native `pred_contribs`. This is exact TreeSHAP; it equals the shap library to 0.0 and its contributions add up to the log prediction (D-031, `tests/test_insights.py`). One-hot columns are summed back to their input. Effects are shown as % change against an average car (`expm1(contribution)`), teal for raising the price and slate for lowering it. Values are written into the bar labels so nothing clips on narrow panels.
- **Why not the shap library:** importing shap would add seconds to server start, and TreeExplainer adds nothing over the native call for one row.
- **Latency:** the app shows the measured time under every estimate. Validation, prediction, explanation, similar listings and the peer comparison took 0.12 to 0.49 s in AppTest and in the browser, on the throttled laptop described in D-031. `tests/test_app_helpers.py::test_estimate_with_explanation_under_two_seconds` enforces the 2-second budget (`gui.max_response_seconds`). The what-if curve (21 predictions in one batch call) renders in its own fragment.
- **Evidence:** `app/Home.py` (`run_estimate`, `why_panel`), `src/carprice/insights.py`.

---

<a id="d-034"></a>

### D-034: Treatment of Rare and Luxury Cars in the UI

- **Date / Phase:** 2026-11-08 / Phase 7
- **Rules:**
  1. *Cannot pick an unknown car:* brand and model are dropdowns from the training data (D-030), so users cannot enter a brand or model the model has never seen. Numeric inputs are bounded by the training range (D-029).
  2. *Lower-confidence notice:* when `PredictionResult.cautions` is non-empty (age ≥ 9 years, a luxury brand or an estimate above ₹20 lakh, a model or fuel with fewer than 50 training rows), the result card shows a "Lower confidence" note with a red left edge and one plain sentence per reason, beside the estimate. The estimate is still shown: it remains the best available guess, and the 80% range already widens in rupees for expensive cars.
  3. *Similar listings:* for rare models with fewer than 10 comparable listings, the message falls back to the brand ("Similar 6-year-old Bmw cars…") or says plainly that there are too few listings to quote a market range.
  4. *Model performance page:* the weakest segment is the page's single amber highlight, and the caption names old, luxury and the cheapest cars as weaker.
- **Why:** D-026 showed these groups are where the model is least reliable. Warning beside the number, in words, follows A7's rule never to rely on colour alone and avoids silently over-trusting the estimate.

---

<a id="d-035"></a>

### D-035: Design System Implementation and Deviations from A7 and the Guideline

- **Date / Phase:** 2026-11-08 / Phase 7
- **Typography:** Fraunces for headings, Inter for the UI and JetBrains Mono for numbers, loaded through Streamlit's native theme options (`font`, `headingFont`, `codeFont` with Google Fonts URLs) and the `@import` in `theme.css`. **Deviation from the guideline's `config.toml`:** it lists `font = "sans serif"`; that line is replaced by the A7 fonts, because A7 requires Inter, Fraunces and JetBrains Mono, and the colour values are kept exactly. **Three type sizes:** 44 px (the estimate only, so it is always the largest element), 28 px (page title and metric values) and 14 px (everything else, including chart text); hierarchy comes from weight and colour.
- **Colour discipline:** only A7 colours, from `carprice/palette.py` and `theme.css`. The guideline config also sets `redColor`, `greenColor`, `borderColor` and the chart palettes to A7 values, so no default Streamlit red or blue appears, and the default red-orange decoration bar is hidden. **One amber highlight per view:** the estimate marker (estimator), the top brand (market explorer) and the weakest segment (model performance). **Contrast:** muted grey #6B7280 measures 4.44:1 on the warm background #F7F5F0, just under 4.5:1, so muted text is used only inside white cards (4.83:1); text on the page background is ink. Amber is never used for text (2.05:1). The red A7 "negative" colour marks only errors and low-confidence warnings.
- **Spacing and surfaces:** the 4 px scale (4/8/12/16/24/32/48) as CSS variables; cards are white with a 1 px #E3DED3 border, 12 px radius, 24 px padding and no shadows. Widgets use an 8 px radius.
- **Motion (the student asked for clean, smooth animations):** cards and the estimate rise 6 px and fade in (320 to 420 ms, ease-out); charts and tables fade in; buttons lift 1 px on hover; inputs get a soft teal focus ring; the tab underline slides. All motion is disabled under `prefers-reduced-motion`. Because Streamlit only re-mounts changed elements, animations play when content changes (a new estimate), not on every keystroke.
- **Robustness:** `client.showErrorDetails = "none"`, so no traceback ever reaches the browser. Every page body runs inside `safely()`, which turns a missing file or bad data into a plain-language error note. Each page has an empty state (no estimate yet, no upload yet, filters match nothing), a loading state (spinners and cached-loader messages) and an error state.
- **Responsive:** below 960 px wide, all column rows stack (CSS on `stHorizontalBlock`), so the 800 px screenshots show a single readable column with nothing cut off.
- **Controls:** fuel and transmission use horizontal radio buttons rather than segmented controls, because a segmented control can be de-selected to "nothing", which would produce a needless "missing" error.
- **Known deviation:** the SHAP summary on Model performance is the Phase 5 static PNG, whose internal labels use matplotlib's fonts, sizes and raw feature names. It is the evaluated report figure, so it is shown as is, under a sentence-case heading with a readable plain-language paragraph beside it.
- **Screenshots:** `scripts/capture_screenshots.py` (new folder `scripts/`) drives headless Chrome over the DevTools protocol, using the `websockets` package already installed with Streamlit. It waits until every Plotly chart has drawn and the page is stable, then saves `reports/figures/gui_<page>_<desktop|narrow>.png` at 1440 and 800 px. Chrome's own `--screenshot` flag captured only Streamlit's loading skeleton. When the script closes Chrome, Windows asyncio may log a `ConnectionResetError` to the *server* console. That noise comes from the tool, not the app, and the script now leaves the page before closing.

---

<a id="d-036"></a>

### D-036: Dark Mode for the GUI

- **Date / Phase:** 2026-11-08 / Phase 7 (student request)
- **Context:** A7 defines only a light palette. The student asked for a dark mode.
- **Options considered:** (a) a custom in-app toggle that injects dark CSS, (b) a custom toggle that writes Streamlit's stored theme and reloads the page, (c) Streamlit's native `[theme.light]` / `[theme.dark]` themes with its built-in System / Light / Dark picker.
- **Decision:** (c). The app follows the visitor's system setting by default. The ⋮ menu (top right) now offers System / Light / Dark: `client.toolbarMode` changed from `minimal` to `viewer`, and the menu is no longer hidden in `theme.css`.
- **Reason:** (a) cannot restyle the canvas-drawn `st.dataframe`, so tables would stay light. (b) depends on Streamlit internals and the reload wipes the form. (c) is supported, keeps the session, and switches every element instantly without a rerun.
- **How each layer follows the theme:**
  - Native widgets and tables use the two config sections.
  - Custom cards and notes use CSS `light-dark()` tokens. They resolve against the `color-scheme` Streamlit sets on the app; browsers without `light-dark()` keep the light values.
  - Plotly charts are drawn with colour placeholders `#000001` to `#000010`. In the browser, Streamlit swaps these for the active theme's `chartCategoricalColors`. That list has ten role slots: six A7 series, then muted, border, grid and surface.
- **Dark palette (derived from A7, in `src/carprice/palette.py`):**

  | Role | Value |
  |---|---|
  | Page | `#0F1626` |
  | Cards | `#18233A` |
  | Text | `#EEEAE1` |
  | Muted text | `#A4ACBA` |
  | Lines, links, selected text | `#3FB1AF` (brightened teal) |
  | Borders | `#2C3852` |
  | Gridlines | `#243049` |
  | Positive | `#5DBB63` |
  | Negative | `#E8695B` |
  | Ochre series | `#D08A45` |

  - Amber, soft teal and slate are unchanged. The ink series flips to the light text colour.
  - Buttons keep brand teal `#0E7C7B` (white text 5.0:1).
  - Brand teal is only 3.6:1 as text on navy. Selected tabs, toggles and slider values therefore use the brightened teal via CSS.
  - All dark text tokens are at least 4.5:1 and chart series at least 3:1 (WCAG 1.4.11), enforced by `tests/test_theme.py`.
  - The dark sequential ramp is a linear blend from card to teal, built the same way as the light ramp.
- **Related fixes needed for dark mode:**
  - In Streamlit 1.67 the old `stVerticalBlockBorderWrapper` selector matched nothing, so cards had no surface colour in either theme. Bordered containers now come from `ui.panel(name)`, keyed `card-<name>`, and CSS targets `[class*="st-key-card-"]`.
  - The what-if caption said "the dark dot", which is wrong in dark mode. It now says "the ringed dot".
  - The static SHAP PNG stays light and sits on a white frame.
- **Light mode:** unchanged. `[theme.light]` holds the exact A7 values; a test checks this.
- **Evidence:** `.streamlit/config.toml`, `app/components/theme.css`, `app/components/palette.py`, `tests/test_theme.py`. `scripts/capture_screenshots.py` now also saves `gui_*_dark.png`.
- **How to reverse:** delete `[theme.dark]` and add `base = "light"` under `[theme]`.

---

<a id="d-037"></a>

### D-037: Phase 8 Reproducibility Audit and Build Fixes

- **Date / Phase:** 2026-11-09 / Phase 8, task 8.1
- **Context:** The brief asks for `make setup clean train evaluate test` in a fresh clone with a fresh environment, a `make all` and `make app` target, a comparison with the saved results, coverage of `src/` (aim 80%), and a sweep for absolute paths, prints, TODOs and unused imports.
- **Environment:** Windows 11, `Python 3.11.9` from `py -3.11`, a new `.venv` built by `make setup`. Windows has no native `make`; the audit used `GNU Make 4.4.1` from MSYS2 (`C:\msys64\usr\bin\make.exe`). `python tasks.py <target>` is the equivalent everywhere and is what the Makefile calls.
- **"clean" in the brief's command:** the Makefile's `clean` deletes caches and generated data (never the raw CSV). With the prerequisites added below, `make train` then rebuilds the cleaned data and split first, so the brief's command works whichever meaning of "clean" was intended.
- **Failures found and fixed (evidence: `reports/tables/reproducibility_runs.csv`, `reports/reproducibility_check.md`):**
  1. *`make train` failed on a fresh clone* (`FileNotFoundError: data/processed/train.csv`): nothing built the cleaned data or split first. **Fix:** file rules in the Makefile (`cars_clean.csv` → `train.csv` → `final_pipeline.joblib`) so `train`, `evaluate`, `predict`, `test` and `app` build what is missing; new targets `all`, `facts`, `notebooks`, `coverage`, `report`.
  2. *`make test` raised 3 errors in Streamlit's AppTest* ("Could not determine home directory"): MSYS2's make drops `USERPROFILE` and `HOMEDRIVE` when it starts Windows programs. **Fix:** `tasks.py` passes child processes an environment with `USERPROFILE` restored from `HOME` when it is missing.
  3. *`make` retrained the model before `test` and again before `predict`:* `test_run_cleaning_end_to_end` rewrote `data/interim/cars_clean.csv`, making it newer than the split. **Fix:** the test writes to `tmp_path`; `write_csv_if_changed()` (new, in `data.py`) leaves cleaned data and split files untouched when their content is unchanged, so re-running notebook 01 no longer triggers a retrain either.
  4. *`eda_26_luxury_segment_outliers.png` differed between runs:* seaborn's strip-plot jitter used numpy's unseeded global generator. **Fix:** seeded with the project seed for that call only, then the generator state is restored.
  5. *`worked_examples.csv` text differed:* the Phase 7 label change ("Car model") had never been regenerated into the Phase 6 files. **Fix:** regenerated; every number is unchanged.
  6. *Final round: three number checks failed after `make all`:* the pipeline re-measures predictor timing and refreshes fit times and trial timestamps, which the report had quoted or could match by coincidence. **Fix:** run-time measurements are excluded from the number checks (D-039) and the report no longer quotes timings as results.
  7. *Final round: a `pip/` cache folder appeared in the clone:* MSYS2's make also drops `LOCALAPPDATA` and `APPDATA`. **Fix:** `tasks.py` restores them for child processes.
- **Final verification:** a second fresh clone of the release candidate ran `make setup`, `make all`, `make test` (187 passed) and `make lint` successfully, and all 113 compared files matched the committed project exactly (`reports/tables/reproducibility_comparison_final.csv`).
- **Tolerance and result:** numeric columns must agree within relative 1e-6 / absolute 1e-9, text columns exactly, PNG figures byte for byte; columns that record run time, timestamps or the git commit are skipped (`scripts/compare_runs.py`). The first clean clone matched the committed results in 99 of 102 files with a largest numeric difference of 0.0; the three differences were items 4 and 5 above and `report_key_figures.csv`, which is new in this phase (`reports/tables/reproducibility_comparison.csv`). The retrained model selected the same parameters and CV score (0.1654).
- **Tuning time-out risk:** each Optuna study also has `models.timeout_seconds` (3600). If a much slower machine reached it, fewer than 35 trials would run and the chosen parameters could differ. Training took about 20 minutes in total on the audit laptop, far from the limit, so the setting was left unchanged and is recorded here.
- **Coverage:** 53.8% of `src/carprice` before this phase, 90.4% after (`reports/tables/coverage_summary.csv`). The new `tests/test_pipeline_runs.py` runs the audit, EDA, benchmark, tuning, final fit, evaluation, explainability and prediction-report code on real data with every output folder redirected to a temporary directory, and checks that the audit, EDA and what-if tables regenerate identically. Evaluation code runs on training rows, never on the locked test set (D-025).
- **Lint:** `ruff check` and `ruff format --check` now run on the whole repository, notebooks included (previously only `src` and `tests`). Notebooks may import after `sys.path` is set (`E402` ignored for `notebooks/*.ipynb` only).
- **Sweep:** no TODO or FIXME comments; ruff reports no unused imports anywhere. `print` was replaced by logging in `tasks.py` and `scripts/capture_screenshots.py`; one `print` remains deliberately in `prediction_reports.py`, where a timing subprocess returns its result as JSON on standard output. The only absolute paths are standard browser install folders in `capture_screenshots.py`, now tried after `CHROME_PATH` and a `PATH` lookup. Notebook outputs no longer contain the author's absolute paths (they came from warnings).
- **New dependencies (logged as A2 requires):** `pytest-cov` and `coverage` for the coverage requirement; `pypandoc_binary` for the pandoc export (D-040).
- **Impact / how to reverse:** `git revert` the Phase 8 commits; no model, metric or table value changed.

---

<a id="d-038"></a>

### D-038: EDA Notebook Display Fix and Audit of Phase 1 and 2 Figures

- **Date / Phase:** 2026-11-09 / Phase 8, tasks 8.1 and 8.3
- **Context:** `viz.py` selects matplotlib's non-interactive `Agg` backend on import (right for scripts), so every `plt.show()` in `notebooks/02_eda.ipynb` printed a warning and **the EDA notebook displayed no charts**. Writing the final report also required checking every EDA figure against its table, which D-028 noted had not been done.
- **Decision:** (a) notebook 02 switches to the inline backend after importing `viz` and was re-executed (all three notebooks now run cleanly through `python tasks.py notebooks`); (b) every figure in `reports/eda_findings.md` and the notebook 01 and 02 markdown was checked against its table and corrected where it differed. Text only: no code, table or model output changed.
- **Corrections (all values now from the named tables):**
  - Transmission medians ₹ 10,50,000 automatic vs ₹ 5,00,000 manual, raw ratio 2.10 (was ₹ 13.00 vs ₹ 4.75 lakh and "2.74x").
  - Fuel: petrol median ₹ 4,60,000 (was ₹ 4.50 lakh), diesel mean ₹ 10.00 lakh (was ₹ 9.28 lakh), petrol mean ₹ 5.73 lakh (was ₹ 6.29 lakh), CNG ₹ 3,70,000 (was ₹ 3.60 lakh), LPG ₹ 1,82,500 (was ₹ 2.12 lakh), Kruskal-Wallis H 2527.5 (was 1756.2).
  - Seller medians ₹ 5,90,000 dealer, ₹ 5,10,000 individual, effect size 0.18 (were ₹ 6.50 and ₹ 4.50 lakh, 0.26).
  - Brands: Maruti 4,933 (32.36%) and Hyundai 2,952 (19.36%), together 51.73%, in 31 brands of which 13 have fewer than 30 rows (were 4,948, 2,982, 52.0%, 32 and 14); brand medians for Toyota, MG, Isuzu, Audi, Mercedes-Benz, BMW and Porsche corrected.
  - Seats: 7-seater median ₹ 8,75,000 (was ₹ 8.80 lakh); 4-seaters are ₹ 27,50,000, the second dearest group (was "₹ 3.90 lakh"), so the finding now says seat count mostly reflects body type.
  - Depreciation: 9.39% (Mahindra) to 12.89% (Hyundai) for Maruti, Hyundai, Honda, Mahindra and Toyota (was "8% to 12%" and listed Ford instead of Mahindra; Toyota is not the slowest to depreciate).
  - Shares: manual 79.34% (was 79.5%), five-seaters 83.76% (was 84.4%), mileage median 19.67 kmpl (was "centred at 19.3").
  - Notebook 01: raw seller counts 9,539 / 5,699 / 173 (were 5,917 / 5,607 / 2,331), raw Maruti count 4,992 (was 4,948), raw price median ₹ 5.56 lakh and 95th percentile ₹ 20.50 lakh (were ₹ 4.5 and ₹ 18 lakh), the memory footprint (the markdown said about 4.6 MB, the notebook printed 2.30 MB), and mileage outliers above 30 identified as CNG cars quoted in km/kg (were "ultra-efficient diesels").
- **Why the conclusions did not change:** every corrected finding points the same way (automatics, diesels and dealers list higher; age and power drive price; brand concentration is high). The modelling decisions rest on measured CV results (D-013 to D-019), not on these descriptive figures.
- **Evidence:** `reports/eda_findings.md` (correction note at the top), notebook diffs in git, `reports/tables/report_key_figures.csv`.
- **Impact / how to reverse:** `git revert` the commit; not recommended.

---

<a id="d-039"></a>

### D-039: Number Provenance Checks and Derived-Figure Table

- **Date / Phase:** 2026-11-09 / Phase 8, task 8.3
- **Context:** A4 forbids invented numbers, and the brief requires every number in the final report to be traceable to a file, with an appendix naming each number's source.
- **Decision:**
  1. `src/carprice/provenance.py` extracts every number from prose (ignoring code, file names, link targets, decision ids, dates, version tags and whole numbers up to 12) and matches it, at the precision written, against the numbers in `reports/tables/*.csv`, `reports/predictions/*.csv`, `models/*.json` and `configs/config.yaml`. Percent (x100), lakh and crore forms of a stored value are accepted.
  2. Attribution prefers a file cited in backticks in the same paragraph or table caption ("cited"); a value found only by searching all files is marked. The report's Appendix D is generated by `python -m carprice.provenance reports/final_report.md`, and `tests/test_provenance.py` fails if the appendix is stale, if any number lacks a source, or if any number is matched only by search.
  3. Shares, ratios, relative improvements, skewness and counts that no earlier phase stored are computed by `src/carprice/report_facts.py` into `reports/tables/report_key_figures.csv`, with the formula and source file for each, instead of being worked out by hand.
  4. The same check runs on `eda_findings.md`, `evaluation_summary.md`, the README, the viva notes, the demo script, the traceability table, the reproducibility check and the markdown of all three notebooks.
- **Deviation from the brief:** the brief says "a file in `reports/tables/`". Worked-example and what-if figures live in `reports/predictions/`, the folder the Phase 6 brief named, and model settings in `models/` and `configs/config.yaml`; these are accepted as sources rather than copying them into `reports/tables/`.
- **Volatile measurements excluded:** `predictor_timing.csv`, fit and prediction time columns, Optuna trial timestamps and the model card's timestamp and commit change on every run. They are not sources, so a document cannot quote them as results and they cannot accidentally "support" an unrelated figure (found in the final clean-clone round, D-037 item 6).
- **Known limit:** a bag-of-numbers check can be fooled by a wrong figure that happens to equal some other stored value; the citation preference narrows this, but it is why D-038 also needed a manual audit.
- **Impact / how to reverse:** delete the module and its tests; the documents are unaffected.

---

<a id="d-040"></a>

### D-040: Report Export Toolchain

- **Date / Phase:** 2026-11-09 / Phase 8, task 8.3
- **Context:** The brief asks for `.docx` and `.pdf` exports with pandoc. Pandoc was not installed on the machine, and there is no LaTeX engine for pandoc's default PDF route.
- **Options considered:** (a) install pandoc and a LaTeX distribution system-wide; (b) `pypandoc_binary`, which ships the pandoc executable inside the virtual environment, with the PDF printed from pandoc's HTML by headless Chrome or Edge, already required for the screenshots; (c) skip the PDF.
- **Decision:** (b). `scripts/export_report.py` (`make report`) writes the Word file directly with pandoc (table of contents, numbered sections, embedded figures) and the PDF from a self-contained HTML page styled by `scripts/report_print.css` in the A7 palette and fonts. The equivalent plain pandoc commands are in the README.
- **Reason:** reproducible from `requirements.txt` with no system installs beyond a browser; the PDF keeps the report's typography.
- **Length:** the PDF has 36 A4 pages: title and contents (3), body (26), appendices (about 8, including the generated number-source table). The body is within the brief's 20 to 30 pages.
- **Impact / how to reverse:** any pandoc 3 on `PATH` with a LaTeX engine can produce the PDF instead (`pandoc final_report.md -o final_report.pdf`).

---

<a id="d-041"></a>

### D-041: Final Model and Why (Phase 8 Confirmation)

- **Date / Phase:** 2026-11-09 / Phase 8
- **Final model:** the tuned XGBoost pipeline in `models/final_pipeline.joblib`: `TransformedTargetRegressor` on `log1p(price)` around the four engineered features (D-013), median imputation, per-fold rare-brand grouping and one-hot encoding (D-015), a cross-fitted target encoder for `model` (D-014), and `XGBRegressor` with the parameters in `models/best_params.json`, plus the 80% residual-quantile interval (D-020).
- **Why:** it had the lowest five-fold CV log RMSE under the selection rule fixed before tuning (0.1654, D-019). LightGBM and histogram gradient boosting scored 0.1660, inside one fold standard deviation, so the three are effectively tied and the rule decided. On the locked test set it scored log RMSE 0.1648 (matching CV), MAE ₹ 94,648 and 50.97% within 10%, beating Ridge by ₹ 21,250 of MAE with a bootstrap interval that excludes zero (`test_metrics.csv`, `bootstrap_ci.csv`).
- **Phase 8 check:** retraining from a clean clone reproduced the same parameters, CV scores and model-card metrics (D-037). No model change was made in Phase 8, so the test results remain uncontaminated (D-025).

---

<a id="d-042"></a>

### D-042: Final List of Known Limitations

- **Date / Phase:** 2026-11-09 / Phase 8 (extends D-026)
- **Data:**
  1. No condition, accident or service history, owners, city, colour, trim or variant; coarse model labels (for example "Rover", "C") hide variants worth lakhs.
  2. Listing prices from one platform and one period, not transaction prices; prices will drift.
- **Model accuracy (`segment_errors.csv`, `top_error_patterns.csv`):**
  3. Cars 9 years or older: 33.92% (9-11 years) and 32.74% (12+ years) within 10%, against 50.97% overall.
  4. Luxury cars above ₹ 20 lakh under-priced by a median 5.76%; cheapest quintile over-priced by 7.26%.
  5. Interval calibrated overall (79.96%) but 92.45% for 0-2 year-old cars and 52.38% for 12+ years.
  6. Rare fuels and brands cannot be evaluated reliably; brands absent from training are refused by the predictor and cannot be chosen in the GUI.
- **Accepted deviations:** `km_was_capped` is not a model input (D-024); seat imputation runs before the split for 2 rows (D-022); Ridge uses target encoding (D-028).
- **Engineering:**
  7. Optuna's per-study time-out could shorten tuning on a much slower machine (D-037).
  8. The SHAP summary in the GUI is the static report PNG (D-035), and at desktop width the axis caption under "Why this price" is clipped ("Effect vs an averag…"); both are cosmetic and were left unchanged in Phase 8.
  9. `make` is not native on Windows; `python tasks.py` is the portable route (D-037).

---

<a id="d-043"></a>

### D-043: Scope Not Delivered, and Why

- **Date / Phase:** 2026-11-09 / Phase 8
- **Not delivered or delivered differently:**
  1. *Report author line:* delivered after review. The student supplied the authors, Nithin Gowda R S (1RV24CS174) and Nikhilesh (1RV24CS176), who replaced the placeholder; the number checker now skips the report's YAML title block so USNs are not read as quoted results.
  2. *PDF route:* produced through pandoc HTML and Chrome rather than pandoc with LaTeX (D-040).
  3. *Number sources:* include `reports/predictions/`, `models/` and the config as well as `reports/tables/` (D-039).
  4. *pre-commit hooks* (optional in A2): not set up; `make lint` and `make test` are the documented gates.
  5. *Notebooks in `make all`:* not included, because they only display saved outputs; `make notebooks` re-executes them.
  6. *Clipped GUI caption and static SHAP image:* left as they are (D-042 item 8); fixing them would change Phase 7 code outside this phase's scope.
  7. *Publishing:* the v1.0 tag and commits are local; nothing was pushed to the GitHub remote (that is the student's decision).
- **Delivered as asked:** clean-clone reproduction with a log, `make all` and `make app`, lint clean, tests passing, coverage reported (above 80%), README, decision log tidy, CHANGELOG, final report in three formats with a source appendix, viva notes, demo script and traceability table.

---

<a id="d-044"></a>

### D-044: Resolution of the Remaining Open Questions

- **Date / Phase:** 2026-11-09 / Phase 8
- **Question 1, predictor timing under a throttled CPU (D-031):** **resolved.** Re-measured on mains power with the Performance plan: in all five fresh processes the predictor loaded in 0.335 s to 0.419 s and a single prediction took a median of 20.508 ms to 24.887 ms, so every run passes both targets (2.0 s and 100 ms) and `test_timing_meets_targets` passes (`reports/tables/predictor_timing.csv`).
- **Question 2, dark-mode screenshots:** **resolved.** All eight light and eight dark screenshots were regenerated (`reports/figures/gui_*.png`, `gui_*_dark.png`).

---

## Open questions for the student

1. **Author line.** Resolved: Nithin Gowda R S (1RV24CS174) and Nikhilesh (1RV24CS176) are now the report's authors.
2. **Publishing.** Push the commits and the `v1.0` tag to GitHub (`git push origin main --tags`) when you are ready; nothing has been pushed.
3. **Cosmetic GUI fixes (optional).** Shorten the "Why this price" axis caption so it is not clipped at desktop width, and consider an interactive SHAP chart instead of the static image (D-042 item 8).
