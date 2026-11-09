# Requirements traceability

Each academic requirement mapped to the files, notebook sections, tests and decisions that demonstrate it. Requirement 4 is split into **4a** (evaluation with proper metrics and prediction results) and **4b** (a GUI that demonstrates the results). Paths are relative to the project root; "§" refers to a numbered section heading inside a notebook; chapter numbers refer to `reports/final_report.md`.

## Summary

| Requirement | Primary evidence | Command that regenerates it |
|---|---|---|
| 1. Data cleansed and pre-processed, every decision justified | `src/carprice/data.py`, `reports/tables/cleaning_log.csv`, `notebooks/01_data_audit.ipynb` | `python tasks.py clean-data` |
| 2. Complete EDA with written interpretation | `notebooks/02_eda.ipynb`, `reports/eda_findings.md`, `reports/figures/eda_*.png` | `python tasks.py eda` |
| 3. Suitable algorithms, built and tuned leak-free | `src/carprice/features.py`, `src/carprice/train.py`, `reports/tables/cv_results.csv`, `models/best_params.json` | `python tasks.py split`, `python tasks.py train` |
| 4a. Proper metrics, evaluation and prediction results | `src/carprice/evaluate.py`, `reports/tables/test_metrics.csv`, `notebooks/03_model_evaluation.ipynb`, `reports/predictions/` | `python tasks.py evaluate`, `python tasks.py predict` |
| 4b. GUI demonstrating the results | `app/Home.py`, `app/pages/`, `reports/figures/gui_*.png` | `python tasks.py app` |

## 1. Data cleansing and pre-processing

| What is shown | Where |
|---|---|
| Raw data never modified | `data/raw/cardekho_dataset.csv` (read only), `tests/test_data.py::test_cleaning_rules_do_not_mutate_their_input`, D-002 |
| Schema, dtype and range validation | `validate_schema()` in `src/carprice/data.py`; `tests/test_data.py` |
| Audit of the raw data | `notebooks/01_data_audit.ipynb` §1 to §8; `reports/tables/audit_*.csv` |
| Every cleaning rule with before/after counts | `reports/tables/cleaning_log.csv`; notebook 01 §9; final report Chapter 3 |
| Justification of each rule | `decisions.md` D-003 (duplicates), D-004 (odometer cap), D-005 (fuel labels), D-006 (seats), D-007 (luxury cars kept), D-008 (identifiers) |
| 2% row-loss guard | `cleaning.max_drop_row_fraction` in `configs/config.yaml`; `src/carprice/data.py` |
| Pre-processing inside a pipeline (imputation, scaling, encoding) | `build_preprocessor()` and `build_feature_pipeline()` in `src/carprice/features.py`; `tests/test_features.py` (output shape and column names) |
| Leakage prevention | Split before fitting (D-012); `load_training_data()` refuses `test.csv` (`tests/test_train.py::test_train_module_never_reads_test_csv`); pre-split steps assessed in D-022 |
| Cleaned output | `data/interim/cars_clean.csv` (generated) |

## 2. Exploratory data analysis

| What is shown | Where |
|---|---|
| Univariate: target and numeric distributions | Notebook 02 §2; `reports/figures/eda_01` to `eda_07`; `reports/tables/eda_numeric_summary.csv` |
| Univariate: categorical levels and imbalance | Notebook 02 §3; `reports/figures/eda_08` to `eda_12` |
| Bivariate: price against each input | Notebook 02 §4 and §5; `reports/figures/eda_13` to `eda_22`; `reports/tables/eda_*_medians.csv`, `eda_depreciation_rates.csv` |
| Multivariate: correlations, interactions, collinearity | Notebook 02 §6 and §7; `reports/figures/eda_23` to `eda_25`; `reports/tables/eda_pearson_correlation.csv`, `eda_spearman_correlation.csv`, `eda_vif_summary.csv` |
| Outliers and data quality | Notebook 02 §8 and §9; `reports/figures/eda_26_luxury_segment_outliers.png`; `reports/tables/eda_rare_brands.csv`, `eda_model_cross_brand.csv` |
| Hypothesis tests with effect sizes | Notebook 02 §10; `reports/tables/eda_hypothesis_tests.csv`; D-011 |
| Written interpretation | Markdown takeaway under every notebook 02 chart; `reports/eda_findings.md`; final report Chapter 4 |
| Implications carried into modelling | Notebook 02 §11; D-009, D-010; `reports/tables/feature_checks.csv` |
| Shared plotting style | `src/carprice/viz.py`, `src/carprice/palette.py`; `tests/test_viz.py` |

## 3. Algorithm selection, model building and tuning

| What is shown | Where |
|---|---|
| Target transformation | `TransformedTargetRegressor` in `src/carprice/train.py`; D-009, D-017 |
| Engineered features measured, not assumed | `evaluate_feature_candidates()` in `src/carprice/features.py`; `reports/tables/feature_checks.csv`; D-013 |
| Encoding choice for the 120-level `model` column | `reports/tables/encoder_comparison.csv`; D-014 |
| Ten-algorithm shortlist with reasons | `reports/tables/model_candidates.csv`; `models.candidates` in config; D-016 |
| Cross-validated comparison | `reports/tables/cv_results.csv`; `reports/figures/model_cv_comparison.png` |
| Tuning with fixed seed and logged trials | `tune_model_study()` in `src/carprice/train.py`; `reports/tables/tuning_trials_*.csv`; `reports/figures/tuning_history.png`, `tuning_param_importances.png`; D-018 |
| Selection rule and final choice | `models/best_params.json`; `models/model_card.json`; D-019, D-041 |
| Saved full pipeline and reload test | `models/final_pipeline.joblib` (generated); `tests/test_train.py`, `tests/test_predict.py` |

## 4a. Evaluation, metrics and prediction results

| What is shown | Where |
|---|---|
| Metric choice fixed before testing | D-023; `evaluation` block in `configs/config.yaml` |
| One-time test scoring | D-025; `run_test_evaluation()` in `src/carprice/evaluate.py` |
| Metrics on both scales for model and baselines | `reports/tables/test_metrics.csv`; notebook 03 §1 |
| Uncertainty of the metrics | `reports/tables/bootstrap_ci.csv`; notebook 03 §2 |
| Overfitting check | `reports/tables/train_vs_test.csv`; notebook 03 §3 |
| Predicted against actual; interval calibration | `reports/figures/eval_01_predicted_vs_actual.png`; `reports/tables/interval_coverage.csv`; notebook 03 §4 |
| Residual diagnostics | `reports/tables/residual_stats.csv`; `reports/figures/eval_02`, `eval_03`, `eval_04`; notebook 03 §5 |
| Segment analysis and weakest segments | `reports/tables/segment_errors.csv`; `reports/figures/eval_05_segment_within10.png`; notebook 03 §6 |
| Largest errors | `reports/tables/top_errors.csv`, `top_error_patterns.csv`; notebook 03 §7 |
| Explainability | `reports/tables/permutation_importance.csv`, `shap_importance.csv`; `reports/figures/eval_07_permutation_importance.png`, `shap_*.png`; notebook 03 §8 |
| Metric functions verified by hand examples | `tests/test_evaluate.py` |
| Test-set predictions | `reports/predictions/test_predictions.csv` |
| Worked examples | `reports/predictions/worked_examples.csv`, `worked_examples.md`; D-031 |
| What-if analysis | `reports/predictions/whatif_age_km.csv`; `reports/figures/whatif_age_km.png` |
| Predictor on a known row; validation of bad input | `tests/test_predict.py`; `src/carprice/predict.py`; D-029, D-030 |
| Written verdict and limitations | `reports/evaluation_summary.md`; notebook 03 §9; D-026, D-042; final report Chapters 6, 7 and 9 |

## 4b. Graphical user interface

| What is shown | Where |
|---|---|
| Price estimator with range, warnings, explanation, compare, what-if, batch upload | `app/Home.py`; `app/components/`; `reports/figures/gui_estimator_desktop.png` |
| Market explorer | `app/pages/1_market_explorer.py`; `src/carprice/market.py`; `reports/figures/gui_market_explorer_desktop.png` |
| Model performance (all numbers read from tables) | `app/pages/2_model_performance.py`; `reports/figures/gui_model_performance_desktop.png` |
| About and data, model card | `app/pages/3_about_and_data.py`; `reports/figures/gui_about_and_data_desktop.png` |
| Design system and theme (light and dark) | `.streamlit/config.toml`; `app/components/theme.css`; `tests/test_theme.py`; D-035, D-036 |
| Page smoke tests and 2-second response budget | `tests/test_app_helpers.py`; `gui.max_response_seconds` in config; D-033 |
| Narrow-screen and dark-mode screenshots | `reports/figures/gui_*_narrow.png`, `gui_*_dark.png`; `scripts/capture_screenshots.py` |
| Demo walkthrough | `reports/demo_script.md` |

## Project-level evidence

| What is shown | Where |
|---|---|
| Decision log | `decisions.md` (D-001 to D-044) |
| Clean-clone reproduction | `reports/reproducibility_check.md` |
| Every quoted number has a source | `src/carprice/provenance.py`; `tests/test_provenance.py`; final report Appendix D |
| Final report | `reports/final_report.md`, `.docx`, `.pdf` |
