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

## Open questions for the student

- None currently for Phase 0.

