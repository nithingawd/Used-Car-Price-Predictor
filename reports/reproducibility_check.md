# Reproducibility check (Phase 8)

Can a fresh clone, a fresh environment and the documented commands reproduce the cleaned data, the model, the report figures and the GUI? This log records the commands run, how long they took, what failed, what was fixed and how the results compare with the committed ones. Decisions are in `decisions.md` D-037; every number below comes from `reports/tables/reproducibility_runs.csv`, `reproducibility_comparison.csv`, `reproducibility_comparison_final.csv` or `coverage_summary.csv`.

## Verdict

**Yes.** A brand-new clone of the release candidate, with a new environment, ran `make setup`, `make all`, `make test` and `make lint` successfully. All 113 compared output files (cleaned data, split, every table, every model metadata file and every figure) matched the committed project exactly, with a largest numeric difference of 0.0 [`reproducibility_comparison_final.csv`]. Getting there took three audit rounds and one final round, which found and fixed seven problems (failures 1 to 5 and two differing outputs); they are described below in the order they were found.

## Environment

| Item | Value |
|---|---|
| Machine | Windows 11 laptop, on mains power, Performance power plan |
| Python | `Python 3.11.9` (`py -3.11`), new `.venv` created by `make setup` |
| make | `GNU Make 4.4.1` from MSYS2 (`C:\msys64\usr\bin\make.exe`); Windows has no native make |
| Clone location | a new temporary folder outside the project, cloned with `git clone` from the project repository |
| Isolation | the project's own `.venv` was removed from `PATH`; the clone used only its own environment |

## Commands

The brief's command, run one target at a time so each could be timed:

```bash
git clone <project> clone1 && cd clone1
make setup PYTHON="py -3.11"     # first run only: choose the Python 3.11 interpreter
make clean
make train
make evaluate
make test
make eda predict                 # figures for the report and the GUI outputs
```

Each `make` target runs the same function as `python tasks.py <target>`.

## Audit 1: the project as it was at the end of Phase 7 (commit `6968954`)

[`reproducibility_runs.csv`, rows for run `audit-1`]

| Target | Exit code | Seconds | Result |
|---|---|---|---|
| `setup` | 0 | 438 | New environment created, all pinned requirements installed |
| `clean` | 0 | 2 | Caches and generated data removed |
| `train` | 2 | 31 | **Failed:** `FileNotFoundError: data/processed/train.csv` |

**Failure 1.** On a fresh clone nothing had produced the cleaned data or the split, and the Makefile had no prerequisites, so `make train` stopped. **Fix (commit `f1195ee`):** file rules in the Makefile (`data/interim/cars_clean.csv` → `data/processed/train.csv` → `models/final_pipeline.joblib`), so `train`, `evaluate`, `predict`, `test` and `app` build whatever is missing; a `make all` target runs the whole pipeline.

## Audit 2: after the Makefile fix (commit `f1195ee`)

[`reproducibility_runs.csv`, rows for run `audit-2`]

| Target | Exit code | Seconds | Result |
|---|---|---|---|
| `clean` | 0 | 2 | |
| `train` | 0 | 1170 | Built the cleaned data and split first, then benchmark, tuning, final fit; selected tuned XGBoost with CV log RMSE 0.1654, as committed |
| `evaluate` | 0 | 36 | Test tables, figures, permutation importance, SHAP |
| `test` | 2 | 35 | 146 passed, **3 errors** |

**Failure 2.** The three errors came from Streamlit's AppTest: "Could not determine home directory". MSYS2's make removes `USERPROFILE` and `HOMEDRIVE` when it starts Windows programs, and Python then has no home folder. It does not happen when `python tasks.py test` is run directly. **Fix (commit `675fb7d`):** `tasks.py` restores `USERPROFILE` from `HOME` for its child processes.

## Audit 3: after the environment fix (commit `675fb7d`)

[`reproducibility_runs.csv`, rows for run `audit-3`]

| Target | Exit code | Seconds | Result |
|---|---|---|---|
| `test` | 0 | 637 | 149 passed, but make **retrained the model first** |
| `eda` | 0 | 28 | All EDA figures and tables |
| `predict` | 0 | 1251 | Phase 6 outputs, but make **retrained the model first** again |

**Failure 3.** One test, `test_run_cleaning_end_to_end`, rewrote `data/interim/cars_clean.csv` in the project. Its content was identical, but its newer timestamp made make treat the split and the model as out of date, so the next target retrained (which is why `test` and `predict` took so long). **Fix (commit `26c6d75` and `ebb46ac`):** the test writes to a temporary folder, and `write_csv_if_changed()` leaves the cleaned data and split files untouched when their content has not changed.

## Comparison with the committed results

`scripts/compare_runs.py` compared every CSV in `reports/tables/` and `reports/predictions/`, every JSON file in `models/`, the cleaned data, the split and every PNG in `reports/figures/` between the clone and the committed project [`reproducibility_comparison.csv`].

- **Tolerance:** numeric columns must agree within a relative 1e-6 and absolute 1e-9; text columns exactly; figures byte for byte. Columns recording run time, timestamps or the git commit are skipped, and `predictor_timing.csv` is skipped because it is a timing measurement.
- **Result:** 99 of 102 files matched; the largest numeric difference in any matching table was 0.0, so every reproduced number is identical, not merely close. The retrained model chose the same hyperparameters with the same CV score, and the test metrics, segment table, bootstrap intervals and SHAP importances are identical.
- **The three differences:**
  1. `report_key_figures.csv` is new in Phase 8 and did not exist at that commit (expected).
  2. `eda_26_luxury_segment_outliers.png` showed the same data with the points jittered differently: seaborn drew the jitter from numpy's unseeded generator. **Fixed:** seeded with the project seed (commit `26c6d75`); two consecutive runs now produce byte-identical files.
  3. `worked_examples.csv` and `.md` differed only in wording ("Car model" against "Model"): a Phase 7 label change had never been regenerated into the Phase 6 files. **Fixed:** regenerated (commit `fe4ea23`); every number is unchanged.

## Final verification: a second fresh clone (commits `dd394c3` and `0d904a5`)

After all Phase 8 changes and the dependency freeze, a new clone was made in a new folder with a new environment and the full pipeline was run with `make all` [`reproducibility_runs.csv`, rows `final-1` and `final-2`].

| Run | Target | Exit code | Seconds | Result |
|---|---|---|---|---|
| final-1 | `setup` | 0 | 450 | New environment from the frozen `requirements.txt` |
| final-1 | `all` | 0 | 985 | Clean data, EDA, split, train, evaluate, predict, key figures; no target retrained twice |
| final-1 | `test` | 2 | 49 | 183 passed, **3 failed** |
| final-2 | `test` | 0 | 52 | 187 passed |
| final-2 | `lint` | 0 | 1 | `ruff check` and `ruff format --check` clean |

**Failure 4.** The three failures were number checks. `make all` re-measures predictor timing and refreshes fit times and Optuna trial timestamps. The report quoted the exact timings, which then no longer matched, and the changing run-time columns could even make an invented test figure look "supported" by coincidence. **Fix (commit `0d904a5`):** the number checker ignores run-time measurements (`predictor_timing.csv`, fit and prediction time columns, trial timestamps, the model card's timestamp and commit), and the report states the timing result without quoting milliseconds.

**Failure 5.** The setup step left a `pip/` cache folder inside the clone: MSYS2's make also drops `LOCALAPPDATA` and `APPDATA`. **Fix (same commit):** `tasks.py` restores them for its child processes; confirmed by starting Python through MSYS2 make.

After pulling the fix, `make test` and `make lint` passed in the same clone without retraining (run `final-2`).

**Final comparison.** `scripts/compare_runs.py` compared the second clone's outputs with the committed project: 113 of 113 files matched, with a largest numeric difference of 0.0 [`reproducibility_comparison_final.csv`]. This includes `eda_26_luxury_segment_outliers.png` and the worked examples, the two files that differed in the first audit. The only files that differ on disk are the run-time measurements the comparison deliberately skips (`predictor_timing.csv`, fit times in `cv_results.csv`, trial timestamps, and the model card's timestamp and commit).

**Time** for the final round from an empty folder: 450 seconds for setup, 985 for the whole pipeline (mostly training) and 49 for the tests.

## Tests, lint and coverage

- **Tests:** the suite passes in a clean clone (149 tests in audit 3, 187 in the final round). Phase 8 adds tests for the provenance checks, the key figures, cleaning-rule immutability and sandboxed runs of every pipeline stage; the final count is printed by `make test`.
- **Lint:** `ruff check .` and `ruff format --check .` pass on the whole repository, notebooks included (previously only `src/` and `tests/` were checked, and the notebooks had unused imports and unformatted cells).
- **Coverage of `src/carprice`:** 53.8% before this phase and **90.4%** after, above the 80% aim [`coverage_summary.csv`]. The lowest modules are `prediction_reports.py` (73.9%, mostly the multi-process timing harness) and `train.py` (81.8%, mostly the command-line entry point).

```bash
make lint
make coverage        # pytest --cov=src/carprice
```

## Code sweep

| Check | Finding | Action |
|---|---|---|
| Hard-coded absolute paths | Browser install folders in `scripts/capture_screenshots.py` | Kept as fallbacks after `CHROME_PATH` and a `PATH` lookup |
| Absolute paths in notebook outputs | Warning messages showed the author's home folder | Causes removed (inline backend, filtered widget warning); outputs regenerated |
| Leftover `print` | `tasks.py`, `scripts/capture_screenshots.py` | Replaced by logging |
| `print` kept deliberately | `src/carprice/prediction_reports.py` | A timing subprocess returns JSON on standard output |
| TODO / FIXME comments | None | |
| Unused imports | Notebooks 01 to 03 | Removed; ruff now checks notebooks |
| Missing docstrings | Package `__init__` files, one nested function, nine tests | Added |

## Reproduce this check

```bash
git clone <repository-url> fresh && cd fresh
make setup                               # Windows: make setup PYTHON="py -3.11"
make all test lint
python scripts/compare_runs.py <path-to-reference-checkout> . --out comparison.csv
```

Run times differ between machines; numbers must not.
