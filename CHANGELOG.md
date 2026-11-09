# Changelog

All notable changes to this project. Dates are ISO dates; decision ids refer to `decisions.md`.

## [1.0.0] - 2026-11-09 (git tag `v1.0`)

Final submission snapshot.

### Phase 8: verification, documentation and release
- Clean-clone reproducibility audit with commands, run times and a file-by-file comparison against the committed results (`reports/reproducibility_check.md`, `scripts/compare_runs.py`, D-037). A final fresh clone of the release reproduced all 113 compared outputs exactly.
- Fixed: `make train`, `make evaluate` and `make test` failed on a fresh clone because nothing built the cleaned data and the split first. The Makefile now has file prerequisites; new targets `all`, `facts`, `notebooks`, `coverage` and `report`.
- Fixed: under MSYS2 `make` on Windows, child processes lost `USERPROFILE`, `LOCALAPPDATA` and `APPDATA` (Streamlit tests failed, pip cached into the project); `tasks.py` restores them.
- Fixed: a test rewrote `data/interim/cars_clean.csv`, which made `make` retrain; tests now write to temporary folders and unchanged data files are no longer rewritten.
- Fixed: the luxury-outlier EDA figure used unseeded jitter and differed between runs.
- Test coverage of `src/carprice` raised from 53.8% to 90.4% with sandboxed runs of every pipeline stage.
- All transitive dependencies frozen in `requirements.txt` (notebook tooling was previously unpinned).
- Fixed: the EDA notebook displayed no charts (the non-interactive plotting backend was forced on import); it now renders inline and was re-executed. Figures in the Phase 1 and Phase 2 narratives that did not match their tables were corrected (D-038).
- Added `src/carprice/provenance.py` (every quoted number must exist in a saved table, model file or the config) and `src/carprice/report_facts.py` (derived shares, ratios and improvements in `reports/tables/report_key_figures.csv`), with tests (D-039).
- Added the final report (`reports/final_report.md`, `.docx`, `.pdf`) with a generated number-source appendix, the viva notes, the demo script and the requirements traceability table.
- Report export with the pandoc binary from `pypandoc_binary` and headless Chrome for the PDF (`scripts/export_report.py`, D-040).
- README rewritten; `decisions.md` given a table of contents, corrections and the Phase 8 entries D-037 to D-044.
- Lint covers the whole repository including notebooks; `tasks.py` and the scripts use logging instead of `print`; missing docstrings added; run-time measurements are excluded from number checks.
- GUI screenshots regenerated in light and dark mode.

### Phase 7: GUI (D-032 to D-036)
- Four-page Streamlit app: price estimator with range, cautions, per-car TreeSHAP explanation, comparison, what-if and batch upload; market explorer; model performance; about and data.
- A7 design system, light and dark themes, GUI tests, screenshot tool.

### Phase 6: prediction (D-029 to D-031)
- Validated `PricePredictor` with typed `CarSpec` and `PredictionResult`, plain-language input errors, brand/model consistency, cautions for weak segments, command-line interface.
- Test-set predictions, rule-based worked examples, what-if sweep, timing measurement.

### Phase 5: evaluation (D-023 to D-028)
- Metrics fixed before the locked test set was scored once; bootstrap intervals, calibration, segment analysis, largest errors, permutation importance and SHAP; evaluation notebook and summary.

### Phase 4: modelling (D-016 to D-022)
- Ten-algorithm cross-validated benchmark, Optuna tuning of three boosters, tuned XGBoost selected and saved with a model card and 80% interval quantiles; consistency review.

### Phase 3: features and split (D-012 to D-015)
- Stratified 80/20 split, measured feature screening, ColumnTransformer pipeline with rare-category grouping and target encoding.

### Phase 2: exploratory analysis (D-009 to D-011)
- Univariate, bivariate and multivariate analysis, hypothesis tests, EDA figures and findings.

### Phase 1: cleaning (D-003 to D-008)
- Data audit and seven logged cleaning rules producing `data/interim/cars_clean.csv`.

### Phase 0: setup (D-001, D-002)
- Repository layout, pinned environment, configuration and theme.
