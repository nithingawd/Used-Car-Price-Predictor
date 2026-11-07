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

## Open questions for the student

- None currently for Phase 0.

