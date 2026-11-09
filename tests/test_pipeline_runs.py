"""Integration tests that run the Phase 1 to 6 pipeline code on real data, in a sandbox.

The orchestration and plotting code in data.py, viz.py, train.py, evaluate.py, eval_plots.py
and prediction_reports.py writes figures, tables and models. These tests point every output
folder at a temporary directory (copying in the saved tables and models the code reads), so
the project's own outputs are never touched. Where a stage is cheap enough, its regenerated
tables are compared with the committed ones, which checks reproducibility as well as coverage.

The locked test set is never scored here: evaluation code runs on a sample of training rows.
"""

from collections.abc import Iterator
from dataclasses import replace
import json
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import carprice.config
from carprice.config import AppConfig, get_config
import carprice.data
import carprice.eval_plots
import carprice.evaluate
import carprice.features
import carprice.insights
import carprice.predict
import carprice.prediction_reports
import carprice.report_facts
import carprice.train

PATCHED_MODULES = [
    carprice.config,
    carprice.data,
    carprice.eval_plots,
    carprice.evaluate,
    carprice.features,
    carprice.insights,
    carprice.predict,
    carprice.prediction_reports,
    carprice.report_facts,
    carprice.train,
]
SAMPLE_ROWS = 600
MODEL_FILES = ("final_pipeline.joblib", "baseline_ridge.joblib", "model_card.json")


def _sandbox_config(real: AppConfig, root: Path) -> AppConfig:
    """Copy of the config whose output folders live under ``root`` (inputs stay real)."""
    paths = replace(
        real.paths,
        figures_dir=root / "reports" / "figures",
        tables_dir=root / "reports" / "tables",
        models_dir=root / "models",
        reports_dir=root / "reports",
    )
    return replace(real, paths=paths)


def _copy_inputs(real: AppConfig, sandbox: AppConfig) -> None:
    """Copy the saved tables, model files and prediction outputs the code reads."""
    shutil.copytree(real.paths.tables_dir, sandbox.paths.tables_dir)
    shutil.copytree(
        real.paths.reports_dir / "predictions", sandbox.paths.reports_dir / "predictions"
    )
    sandbox.paths.figures_dir.mkdir(parents=True)
    sandbox.paths.models_dir.mkdir(parents=True)
    for path in real.paths.models_dir.iterdir():
        if path.suffix in {".json", ".csv"} or path.name in MODEL_FILES:
            shutil.copy2(path, sandbox.paths.models_dir / path.name)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AppConfig]:
    """A config with outputs redirected to a temporary folder, patched into every module."""
    real = get_config()
    config = _sandbox_config(real, tmp_path_factory.mktemp("project"))
    _copy_inputs(real, config)
    with pytest.MonkeyPatch.context() as patch:
        for module in PATCHED_MODULES:
            patch.setattr(module, "get_config", lambda: config)
        yield config
    plt.close("all")


@pytest.fixture(scope="module")
def train_sample(sandbox: AppConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A seeded sample of training rows (features, rupee target, log target)."""
    X, y_inr, y_log = carprice.train.load_training_data()
    rows = X.sample(n=SAMPLE_ROWS, random_state=sandbox.project.random_seed).index
    return X.loc[rows], y_inr.loc[rows], y_log.loc[rows]


def _assert_same_table(produced: Path, committed: Path) -> None:
    """A regenerated table equals the committed one (numbers within float tolerance)."""
    pd.testing.assert_frame_equal(
        pd.read_csv(produced), pd.read_csv(committed), check_dtype=False, atol=1e-9
    )


# ---------------------------------------------------------------------------
# Phase 1 and 2: audit and EDA reproduce the committed tables
# ---------------------------------------------------------------------------


def test_data_audit_reproduces_committed_tables(sandbox: AppConfig) -> None:
    """run_data_audit() rewrites every audit table identically."""
    tables = carprice.data.run_data_audit(tables_dir=sandbox.paths.tables_dir)
    assert tables
    real_tables = carprice.config.find_project_root() / "reports" / "tables"
    for produced in sorted(sandbox.paths.tables_dir.glob("audit_*.csv")):
        _assert_same_table(produced, real_tables / produced.name)


def test_eda_pipeline_reproduces_tables_and_figures(sandbox: AppConfig) -> None:
    """run_eda_pipeline() writes all EDA figures and the same EDA tables as committed."""
    for figure in sandbox.paths.figures_dir.glob("eda_*.png"):
        figure.unlink()
    from carprice.viz import run_eda_pipeline

    run_eda_pipeline()
    real_root = carprice.config.find_project_root() / "reports"
    produced = sorted(p.name for p in sandbox.paths.figures_dir.glob("eda_*.png"))
    assert produced == sorted(p.name for p in (real_root / "figures").glob("eda_*.png"))
    for table in sorted(sandbox.paths.tables_dir.glob("eda_*.csv")):
        _assert_same_table(table, real_root / "tables" / table.name)


# ---------------------------------------------------------------------------
# Phase 4: benchmark, tuning, final fit and encoder comparison on a sample
# ---------------------------------------------------------------------------


def test_cv_benchmark_and_plot(sandbox: AppConfig, train_sample: tuple) -> None:
    """Every candidate is cross-validated and the comparison figure is drawn."""
    X, y_inr, y_log = train_sample
    results, oof = carprice.train.run_cv_benchmark(X, y_inr, y_log, sandbox.paths.tables_dir)
    assert set(results["model_id"]) == set(sandbox.models.candidates)
    assert results["log_rmse_mean"].is_monotonic_increasing
    assert set(oof) == set(sandbox.models.candidates)
    assert carprice.train.plot_cv_comparison(results, sandbox.paths.figures_dir).exists()


def test_tuning_study_is_seeded_and_plotted(sandbox: AppConfig, train_sample: tuple) -> None:
    """Two runs with the same seed give the same trials; diagnostics are drawn."""
    X, _, y_log = train_sample
    cv = carprice.features.make_cv_splitter()
    folds = carprice.train.prepare_cv_folds(X, y_log, cv)
    seed = sandbox.project.random_seed
    first = carprice.train.tune_model_study("xgboost", folds, n_trials=3, seed=seed)
    second = carprice.train.tune_model_study("xgboost", folds, n_trials=3, seed=seed)
    assert [t.value for t in first.trials] == [t.value for t in second.trials]
    paths = carprice.train.plot_tuning_diagnostics({"xgboost": first}, sandbox.paths.figures_dir)
    assert all(path.exists() for path in paths)


def test_final_fit_persists_pipelines_and_card(sandbox: AppConfig, train_sample: tuple) -> None:
    """The tuned pipeline and Ridge baseline are saved, reload and predict in rupees."""
    X, y_inr, y_log = train_sample
    models_dir = sandbox.paths.models_dir.parent / "fitted"
    models_dir.mkdir(exist_ok=True)
    final_path, ridge_path, card = carprice.train.fit_and_persist_pipelines(
        X, y_inr, y_log, _best_params(), models_dir=models_dir
    )
    model = joblib.load(final_path)
    assert (model.predict(X.head(5)) > 0).all()
    assert joblib.load(ridge_path) is not None
    assert card["training_sample_count"] == len(X)
    interval = card["prediction_interval_80pct"]
    assert interval["residual_quantile_10"] < 0 < interval["residual_quantile_90"]


def _best_params() -> dict:
    """The committed tuned parameters (models/best_params.json in the real project)."""
    path = carprice.config.find_project_root() / "models" / "best_params.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_encoder_comparison_table(sandbox: AppConfig, train_sample: tuple) -> None:
    """Both encodings are scored for both model families."""
    X, _, y_log = train_sample
    table = carprice.train.run_encoder_comparison(
        X, y_log, _best_params(), tables_dir=sandbox.paths.tables_dir
    )
    assert len(table) == 4
    assert set(table["model_encoding"]) == {"onehot", "target"}
    assert carprice.train.get_git_commit_hash()


# ---------------------------------------------------------------------------
# Phase 5: evaluation code on training rows (never the locked test set)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_sample(train_sample: tuple) -> tuple[dict, pd.DataFrame]:
    """Saved models applied to the training sample, with segment columns."""
    X, y_inr, _ = train_sample
    models = carprice.evaluate.load_models()
    frame = carprice.evaluate.build_prediction_frame(models, X, y_inr)
    _, all_train_prices, _ = carprice.train.load_training_data()
    return models, carprice.evaluate.assign_segments(frame, all_train_prices)


def test_evaluation_tables_and_figures(scored_sample: tuple) -> None:
    """Every Phase 5 table is built and every test figure is drawn."""
    models, segmented = scored_sample
    tables = carprice.evaluate.build_test_tables(models, segmented)
    assert {"test_metrics", "interval_coverage", "segment_errors"} <= set(tables)
    assert set(tables["test_metrics"]["model_id"]) == set(models)
    paths = carprice.eval_plots.plot_test_figures(segmented, tables)
    assert paths and all(path.exists() for path in paths)
    bootstrap = carprice.evaluate.bootstrap_confidence_intervals(segmented)
    assert (bootstrap["ci_low"] <= bootstrap["ci_high"]).all()


def test_explainability_tables_and_figures(sandbox: AppConfig, train_sample: tuple) -> None:
    """Permutation importance, SHAP values and their figures on a small sample."""
    X, y_inr, _ = train_sample
    final_model = joblib.load(sandbox.paths.models_dir / "final_pipeline.joblib")
    small_x, small_y = X.head(150), y_inr.head(150)
    permutation = carprice.evaluate.compute_permutation_importance(final_model, small_x, small_y)
    assert (
        permutation.loc[permutation["feature"] == "km_was_capped", "rmse_log_increase"].eq(0).all()
    )
    sample = carprice.evaluate.sample_for_shap(small_x)
    shap_values, transformed = carprice.evaluate.compute_shap_values(final_model, sample)
    grouped = carprice.insights.group_shap_by_feature(shap_values, list(transformed.columns))
    tables = {
        "permutation_importance": permutation,
        "shap_importance": carprice.evaluate.shap_importance_table(grouped),
    }
    assert np.isclose(tables["shap_importance"]["share_pct"].sum(), 100, atol=0.1)
    paths = carprice.eval_plots.plot_explainability_figures(
        shap_values, transformed, grouped, sample, tables
    )
    assert all(path.exists() for path in paths)


# ---------------------------------------------------------------------------
# Phase 6: prediction reports from the saved, already-published test predictions
# ---------------------------------------------------------------------------


def test_worked_examples_are_reproduced(sandbox: AppConfig) -> None:
    """Selecting examples from the saved predictions picks the committed cars again."""
    scored = pd.read_csv(sandbox.paths.reports_dir / "predictions" / "test_predictions.csv")
    committed = pd.read_csv(sandbox.paths.reports_dir / "predictions" / "worked_examples.csv")
    examples = carprice.prediction_reports.select_worked_examples(scored)
    assert list(examples["model"]) == list(committed["model"])
    predictor = carprice.predict.PricePredictor()
    described = carprice.prediction_reports.describe_examples(predictor, examples.head(3))
    assert described["explanation"].str.len().gt(0).all()
    path = carprice.prediction_reports.write_worked_examples(described)
    assert path.exists() and "###" in path.read_text(encoding="utf-8")


def test_whatif_grid_and_figure(sandbox: AppConfig) -> None:
    """The what-if grid covers every configured age and km and its chart is drawn."""
    specs = carprice.prediction_reports.write_model_specs()
    car = carprice.prediction_reports.reference_car(specs)
    predictor = carprice.predict.PricePredictor()
    grid = carprice.prediction_reports.whatif_grid(predictor, car)
    settings = sandbox.prediction
    assert len(grid) == len(settings.whatif_ages) * len(settings.whatif_km)
    committed = pd.read_csv(sandbox.paths.reports_dir / "predictions" / "whatif_age_km.csv")
    assert np.allclose(grid["predicted_inr"].to_numpy(), committed["predicted_inr"].to_numpy())
    figure = carprice.prediction_reports.plot_whatif(grid, car)
    assert figure.axes
