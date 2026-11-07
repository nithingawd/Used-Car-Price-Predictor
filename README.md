# Used Car Price Prediction System

An end-to-end, reproducible machine learning system designed to estimate used car selling prices in Indian Rupees (INR) alongside plausible prediction intervals. Built with rigorous data hygiene, complete exploratory analysis, leak-free scikit-learn pipelines, gradient boosting models, and an interactive Streamlit multi-page web application.

---

## Problem Statement

Valuing used cars accurately requires accounting for multifaceted vehicular attributes—including brand equity, model hierarchy, age, cumulative odometer mileage, powertrain specifications (fuel, engine displacement, rated power), transmission technology, seating capacity, and seller channel. This project applies statistical data cleansing, exploratory data analysis, and advanced regression algorithms (Linear Regression, Ridge, Random Forest, XGBoost, LightGBM) to build a trustworthy pricing engine with explainability and intuitive user interfaces.

---

## Tech Stack & Environment

- **Python Version:** 3.11 (`3.11.9`)
- **Environment Management:** Python `venv` + pinned `requirements.txt`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn` (static EDA), `plotly` (interactive app)
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Hyperparameter Optimization:** `optuna`
- **Model Explainability:** `shap` (TreeExplainer) and permutation importance
- **Model Persistence:** `joblib` (end-to-end Pipeline serialization)
- **Application GUI:** `Streamlit`
- **Testing & Quality:** `pytest`, `ruff`
- **Configuration:** YAML via `configs/config.yaml` loaded into typed dataclasses

---

## Repository Structure

```text
.
├── AGENTS.md                  # Master project rules and technical specifications
├── decisions.md               # Auditable log of all technical and modeling decisions
├── README.md                  # Project overview and setup documentation
├── requirements.txt           # Pinned production and development dependencies
├── pyproject.toml             # Ruff linter/formatter & pytest configurations
├── tasks.py                   # Cross-platform CLI task runner (setup, test, lint, app)
├── Makefile                   # Standard Makefile automating workflows
├── configs/
│   └── config.yaml            # Centralized project, model, and path configurations
├── data/
│   ├── raw/                   # Read-only original dataset (cardekho_dataset.csv)
│   ├── interim/               # Cleaned data checkpoints
│   └── processed/             # Train and test splits
├── notebooks/                 # Narrative Jupyter notebooks for EDA and evaluation
├── src/carprice/              # Core reusable Python package
│   ├── __init__.py
│   ├── config.py              # Strongly-typed frozen config dataclasses
│   ├── data.py                # Load, validate, audit, clean (Phase 1)
│   ├── viz.py                 # Shared plot style and EDA figures (Phase 2)
│   ├── features.py            # Split, feature engineering, ColumnTransformer (Phase 3)
│   ├── train.py               # CV benchmark, Optuna tuning, final pipeline (Phase 4)
│   ├── evaluate.py            # Test metrics, segments, bootstrap, SHAP (Phase 5)
│   └── eval_plots.py          # Evaluation and SHAP figures (Phase 5)
├── app/                       # Streamlit web application
├── models/                    # Saved pipeline artifacts (.joblib) and metadata JSON
├── reports/                   # Figures, summary tables, and final report
└── tests/                     # Unit and integration test suite
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11 installed on your system.

### 2. Setup Environment
Clone the repository and run setup:

```bash
# Using Python task runner (Windows / macOS / Linux)
python tasks.py setup

# Or using Make (if installed)
make setup
```

Alternatively, manually create and activate a virtual environment:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Quality Checks and Tests

```bash
# Run test suite
python tasks.py test

# Run lint and formatting checks
python tasks.py lint
```

### 4. Reproduce the Pipeline (Phases 1 to 5)

Run these in order from the project root. Every step reads its settings from
`configs/config.yaml` and uses the global seed, so re-running gives the same numbers.

```bash
# Phase 1: audit tables + cleaned data (data/interim/cars_clean.csv)
python tasks.py clean-data

# Phase 2: EDA figures and tables (narrative lives in notebooks/02_eda.ipynb)
python tasks.py eda

# Phase 3: stratified train/test split + measured feature checks
python tasks.py split

# Phase 4: CV benchmark, Optuna tuning, final pipeline, encoder comparison
#          (writes models/*.joblib, models/model_card.json, reports/tables/*.csv)
python tasks.py train

# Phase 5: score the locked test set once, segments, bootstrap, permutation + SHAP
#          (narrative: notebooks/03_model_evaluation.ipynb, summary: reports/evaluation_summary.md)
python tasks.py evaluate
```

The saved `.joblib` pipelines are not committed to git, so a fresh clone must run
`python tasks.py train` before the tests or the app can load the model. Run-time
columns (`fit_time_sec`, `pred_time_sec`) and trial timestamps vary between machines;
every score and parameter does not.

### 5. Launch the Web Application

```bash
python tasks.py app
```

