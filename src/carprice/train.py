"""Model training, CV benchmarking, tuning, and pipeline persistence."""

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from carprice.config import AppConfig, get_config
from carprice.features import (
    BASE_NUMERIC_FEATURES,
    build_feature_pipeline,
    build_ridge_probe,
    cv_log_rmse,
    make_cv_splitter,
)
from carprice.viz import (
    ACCENT_AMBER,
    BORDER_COLOR,
    CATEGORICAL_PALETTE,
    INK_COLOR,
    MUTED_TEXT,
    PRIMARY_TEAL,
    SURFACE_COLOR,
    apply_style,
    save_figure,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_git_commit_hash() -> str:
    """Retrieve the current short git commit hash for model provenance.

    A "-dirty" suffix means the code or config that defines the model (src/, configs/)
    had uncommitted changes at training time. Generated outputs that training itself
    rewrites (reports/, models/) are deliberately ignored.

    Returns:
        str: 7-character commit SHA (optionally suffixed "-dirty") or 'unknown'.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "configs"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def load_training_data(
    train_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load training partition and extract feature matrix and target series.

    Args:
        train_path: Optional explicit path to train.csv.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: Features (X), rupee price (y), log price (y_log).

    Raises:
        FileNotFoundError: If the train data file is missing.
        ValueError: If an attempt is made to load the locked test set.
    """
    config: AppConfig = get_config()
    target_path = train_path or (config.paths.processed_data_dir / "train.csv")
    if "test" in target_path.name.lower():
        raise ValueError(f"Access violation: strictly forbidden to load '{target_path.name}'.")
    if not target_path.is_file():
        raise FileNotFoundError(f"Training dataset not found at: {target_path}")

    df = pd.read_csv(target_path)
    X = df.drop(columns=["selling_price", "log_selling_price"], errors="ignore")
    y_inr = df["selling_price"]
    y_log = np.log1p(y_inr)
    logger.info("Loaded training partition: X=%s, y=%d rows", X.shape, len(y_inr))
    return X, y_inr, y_log


ESTIMATOR_CLASSES: dict[str, tuple[type, str, bool]] = {
    # model_id: (estimator class, preprocessor kind, accepts random_state)
    "dummy": (DummyRegressor, "tree", False),
    "linear_regression": (LinearRegression, "linear", False),
    "ridge": (Ridge, "linear", True),
    "lasso": (Lasso, "linear", True),
    "k_nearest_neighbors": (KNeighborsRegressor, "linear", False),
    "decision_tree": (DecisionTreeRegressor, "tree", True),
    "random_forest": (RandomForestRegressor, "tree", True),
    "hist_gradient_boosting": (HistGradientBoostingRegressor, "tree", True),
    "xgboost": (XGBRegressor, "tree", True),
    "lightgbm": (LGBMRegressor, "tree", True),
}
# Runtime-only arguments that do not affect results; kept out of the config grids.
RUNTIME_KWARGS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_jobs": -1},
    "xgboost": {"n_jobs": -1},
    "lightgbm": {"n_jobs": -1, "verbose": -1},
}


def get_candidate_models(seed: int) -> dict[str, tuple[Any, str]]:
    """Instantiate the configured candidate regressors paired with preprocessor kind.

    Hyperparameters come from ``models.candidate_params`` in config.yaml (D-016).

    Args:
        seed: Random seed for deterministic reproducibility.

    Returns:
        dict[str, tuple[Any, str]]: Mapping of model_id to (estimator, preprocessor_kind).
    """
    models_cfg = get_config().models
    candidates: dict[str, tuple[Any, str]] = {}
    for model_id in models_cfg.candidates:
        estimator_cls, kind, seeded = ESTIMATOR_CLASSES[model_id]
        kwargs = {
            **models_cfg.candidate_params.get(model_id, {}),
            **RUNTIME_KWARGS.get(model_id, {}),
        }
        if seeded:
            kwargs["random_state"] = seed
        candidates[model_id] = (estimator_cls(**kwargs), kind)
    return candidates


def evaluate_single_candidate(
    model_id: str,
    estimator: Any,
    kind: str,
    X: pd.DataFrame,
    y_log: pd.Series,
    y_inr: pd.Series,
    cv: KFold,
) -> dict[str, Any]:
    """Execute 5-fold cross-validation for a candidate pipeline on training data.

    Args:
        model_id: Model identifier string.
        estimator: Raw scikit-learn compatible estimator.
        kind: 'linear' or 'tree' preprocessor kind.
        X: Feature matrix.
        y_log: Target series on log1p scale.
        y_inr: Target series on raw INR scale.
        cv: Cross-validation splitter.

    Returns:
        dict[str, Any]: Aggregated evaluation metrics and out-of-fold predictions.
    """
    oof_preds_log = np.zeros(len(X), dtype=float)
    fold_rmses: list[float] = []
    fold_maes: list[float] = []
    fold_r2s: list[float] = []
    fit_times: list[float] = []
    pred_times: list[float] = []

    for train_idx, val_idx in cv.split(X, y_log):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        pipe = Pipeline(
            [
                ("features", build_feature_pipeline(kind=kind)),
                ("regressor", estimator),
            ]
        )

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_pred = pipe.predict(X_val)
        t_pred = time.perf_counter() - t1

        oof_preds_log[val_idx] = val_pred
        fit_times.append(t_fit)
        pred_times.append(t_pred)

        fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, val_pred))))
        fold_maes.append(float(mean_absolute_error(y_val, val_pred)))
        fold_r2s.append(float(r2_score(y_val, val_pred)))

    oof_preds_inr = np.expm1(oof_preds_log)
    inr_rmse = float(np.sqrt(mean_squared_error(y_inr, oof_preds_inr)))
    inr_mae = float(mean_absolute_error(y_inr, oof_preds_inr))

    return {
        "model_id": model_id,
        "log_rmse_mean": round(float(np.mean(fold_rmses)), 4),
        "log_rmse_std": round(float(np.std(fold_rmses)), 4),
        "log_mae_mean": round(float(np.mean(fold_maes)), 4),
        "log_mae_std": round(float(np.std(fold_maes)), 4),
        "r2_mean": round(float(np.mean(fold_r2s)), 4),
        "r2_std": round(float(np.std(fold_r2s)), 4),
        "inr_rmse": round(inr_rmse, 1),
        "inr_mae": round(inr_mae, 1),
        "fit_time_sec": round(float(np.mean(fit_times)), 3),
        "pred_time_sec": round(float(np.mean(pred_times)), 4),
        "oof_preds_log": oof_preds_log,
    }


def run_cv_benchmark(
    X: pd.DataFrame,
    y_inr: pd.Series,
    y_log: pd.Series,
    tables_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run full 5-fold cross-validation benchmark across all candidate models.

    Args:
        X: Training features.
        y_inr: Raw INR targets.
        y_log: Log1p transformed targets.
        tables_dir: Output directory for cv_results.csv.

    Returns:
        tuple[pd.DataFrame, dict[str, np.ndarray]]: Metrics table and out-of-fold predictions.
    """
    config = get_config()
    out_dir = tables_dir or config.paths.tables_dir
    seed = config.project.random_seed
    cv = make_cv_splitter()
    candidates = get_candidate_models(seed=seed)

    records: list[dict[str, Any]] = []
    oof_dict: dict[str, np.ndarray] = {}

    for model_id, (estimator, kind) in candidates.items():
        logger.info("Evaluating candidate: %s (kind=%s)", model_id, kind)
        res = evaluate_single_candidate(model_id, estimator, kind, X, y_log, y_inr, cv)
        oof_dict[model_id] = res.pop("oof_preds_log")
        records.append(res)

    results_df = pd.DataFrame(records).sort_values("log_rmse_mean").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "cv_results.csv", index=False)
    logger.info("Saved CV comparison table to %s", out_dir / "cv_results.csv")
    return results_df, oof_dict


def plot_cv_comparison(
    cv_results: pd.DataFrame,
    figures_dir: Path | None = None,
) -> Path:
    """Plot horizontal bar chart of 5-fold CV log RMSE with standard deviation error bars.

    Args:
        cv_results: Benchmark metrics DataFrame.
        figures_dir: Output directory for chart PNG.

    Returns:
        Path: Path to saved figure.
    """
    apply_style()
    config = get_config()
    out_dir = figures_dir or config.paths.figures_dir
    sorted_df = cv_results.sort_values("log_rmse_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(sorted_df))
    best_idx = int(sorted_df["log_rmse_mean"].idxmin())

    colors = [ACCENT_AMBER if i == best_idx else PRIMARY_TEAL for i in range(len(sorted_df))]

    ax.barh(
        y_pos,
        sorted_df["log_rmse_mean"],
        xerr=sorted_df["log_rmse_std"],
        color=colors,
        edgecolor=BORDER_COLOR,
        height=0.65,
        capsize=4,
        error_kw={"elinewidth": 1.5, "ecolor": INK_COLOR, "capthick": 1.5},
    )

    for i, row in sorted_df.iterrows():
        val = row["log_rmse_mean"]
        err = row["log_rmse_std"]
        ax.text(
            val + err + 0.015,
            i,
            f"{val:.4f} (±{err:.4f})",
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK_COLOR,
            weight="semibold" if i == best_idx else "normal",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["model_id"], fontsize=10.5)
    best_row = sorted_df.loc[best_idx]
    ax.set_xlabel("Mean 5-fold CV RMSE (log1p price)")
    ax.set_title(
        f"{best_row['model_id']} has the lowest untuned CV error "
        f"({best_row['log_rmse_mean']:.4f} log RMSE)"
    )
    ax.set_xlim(0, max(sorted_df["log_rmse_mean"]) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    fig.text(
        0.12,
        0.02,
        "Lower is better. Error bars represent std across 5 folds. Amber denotes top candidate.",
        fontsize=9,
        color=MUTED_TEXT,
    )

    return save_figure(fig, out_dir / "model_cv_comparison.png")


def prepare_cv_folds(
    X: pd.DataFrame,
    y_log: pd.Series,
    cv: KFold,
    kind: str = "tree",
) -> list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]]:
    """Precompute transformed fold feature matrices for high-throughput tuning.

    Args:
        X: Feature matrix.
        y_log: Target series on log1p scale.
        cv: Cross-validation splitter.
        kind: 'tree' or 'linear'.

    Returns:
        list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]]: Fold tuples.
    """
    prepared_folds: list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]] = []
    for tr_idx, val_idx in cv.split(X, y_log):
        feat_pipe = build_feature_pipeline(kind=kind)
        X_tr_prep = feat_pipe.fit_transform(X.iloc[tr_idx], y_log.iloc[tr_idx])
        X_val_prep = feat_pipe.transform(X.iloc[val_idx])
        prepared_folds.append((X_tr_prep, y_log.iloc[tr_idx], X_val_prep, y_log.iloc[val_idx]))
    return prepared_folds


def suggest_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Draw one hyperparameter set from a config search space, in key order.

    Args:
        trial: Active Optuna trial.
        space: Mapping of parameter name to {type, low, high, step?, log?}.

    Returns:
        dict[str, Any]: Suggested hyperparameters.

    Raises:
        ValueError: If a parameter type is not 'int' or 'float'.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            raise ValueError(f"Unsupported search-space type for '{name}': {spec['type']}")
    return params


def tune_model_study(
    model_name: str,
    prepared_folds: list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]],
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run a seeded Optuna TPE study minimising mean CV log RMSE for one model.

    Args:
        model_name: Key in ``models.search_spaces`` (and create_tuned_estimator).
        prepared_folds: Pre-transformed fold feature tuples.
        n_trials: Number of optimization trials.
        seed: Random seed for the sampler and the estimator.

    Returns:
        optuna.Study: Completed optimization study.
    """
    models_cfg = get_config().models
    space = models_cfg.search_spaces[model_name]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
        fold_rmses: list[float] = []
        for X_tr, y_tr, X_val, y_val in prepared_folds:
            model = create_tuned_estimator(model_name, params, seed)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        return float(np.mean(fold_rmses))

    study.optimize(objective, n_trials=n_trials, timeout=models_cfg.timeout_seconds)
    return study


def run_tuning(
    X: pd.DataFrame,
    y_log: pd.Series,
    tables_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute hyperparameter tuning for top candidates and persist parameters.

    Args:
        X: Training features.
        y_log: Log target series.
        tables_dir: Tables destination directory.
        models_dir: Models destination directory.

    Returns:
        dict[str, Any]: Best parameters dictionary across the top candidates.
    """
    config = get_config()
    tbl_dir = tables_dir or config.paths.tables_dir
    mod_dir = models_dir or config.paths.models_dir
    n_trials = config.models.tuning_trials
    seed = config.project.random_seed

    logger.info("Precomputing leak-free CV folds for fast tuning...")
    prep_folds = prepare_cv_folds(X, y_log, make_cv_splitter(), kind="tree")

    logger.info("Beginning Optuna hyperparameter optimization (%d trials each)", n_trials)
    studies: dict[str, optuna.Study] = {
        name: tune_model_study(name, prep_folds, n_trials, seed)
        for name in config.models.search_spaces
    }

    best_summary: dict[str, Any] = {}
    for name, study in studies.items():
        trials_df = study.trials_dataframe()
        trials_df.to_csv(tbl_dir / f"tuning_trials_{name}.csv", index=False)
        best_summary[name] = {
            "best_cv_rmse": round(study.best_value, 4),
            "best_params": study.best_params,
        }
        logger.info(
            "Tuned %s: Best CV RMSE = %.4f (params=%s)",
            name,
            study.best_value,
            study.best_params,
        )

    mod_dir.mkdir(parents=True, exist_ok=True)
    with (mod_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    plot_tuning_diagnostics(studies, config.paths.figures_dir)
    return best_summary


def plot_tuning_diagnostics(
    studies: dict[str, optuna.Study],
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Plot optimization history curves and parameter importance bar charts.

    Args:
        studies: Dictionary of completed Optuna studies.
        figures_dir: Output directory for diagnostic figures.

    Returns:
        tuple[Path, Path]: Saved paths for (tuning_history.png, tuning_param_importances.png).
    """
    apply_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimization History
    winner = min(studies, key=lambda name: studies[name].best_value)
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    for i, (name, study) in enumerate(studies.items()):
        trial_values = [t.value for t in study.trials if t.value is not None]
        running_min = np.minimum.accumulate(trial_values)
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        ax1.plot(
            range(1, len(running_min) + 1),
            running_min,
            marker="o",
            markersize=4,
            label=f"{name} (best: {min(trial_values):.4f})",
            color=color,
            linewidth=2,
        )
    ax1.set_xlabel("Optimization trial number")
    ax1.set_ylabel("Best 5-fold CV RMSE so far (log1p price)")
    ax1.set_title(
        f"Tuned {winner} reaches the lowest CV error ({studies[winner].best_value:.4f} log RMSE)"
    )
    ax1.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    ax1.grid(True, linestyle="--", alpha=0.6)
    hist_path = save_figure(fig1, figures_dir / "tuning_history.png")

    # 2. Parameter importances for the winning model's study
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    try:
        importances = optuna.importance.get_param_importances(studies[winner])
    except (RuntimeError, ValueError) as err:
        logger.warning("Could not compute fANOVA importances: %s", err)
        ax2.text(0.5, 0.5, "Importances not computed", ha="center")
    else:
        top_param = next(iter(importances))
        ax2.barh(
            list(importances)[::-1],
            list(importances.values())[::-1],
            color=PRIMARY_TEAL,
            edgecolor=BORDER_COLOR,
            height=0.6,
        )
        ax2.set_xlabel("Relative parameter importance (fANOVA, share of variance)")
        ax2.set_title(f"{top_param} matters most when tuning {winner}")
        ax2.grid(axis="x", linestyle="--", alpha=0.6)
    imp_path = save_figure(fig2, figures_dir / "tuning_param_importances.png")

    return hist_path, imp_path


def create_tuned_estimator(model_name: str, params: dict[str, Any], seed: int) -> Any:
    """Instantiate estimator with tuned hyperparameters and deterministic seed.

    Args:
        model_name: Name of model ('lightgbm', 'xgboost', 'hist_gradient_boosting').
        params: Optimal hyperparameter dictionary.
        seed: Random seed.

    Returns:
        Any: Configured scikit-learn compatible estimator.
    """
    if model_name == "lightgbm":
        return LGBMRegressor(**params, random_state=seed, n_jobs=-1, verbose=-1)
    if model_name == "xgboost":
        return XGBRegressor(**params, random_state=seed, n_jobs=-1)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params, random_state=seed)
    raise ValueError(f"Unsupported model name for tuned estimator: '{model_name}'")


def fit_and_persist_pipelines(
    X: pd.DataFrame,
    y_inr: pd.Series,
    y_log: pd.Series,
    best_params: dict[str, Any],
    models_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Fit winning tuned pipeline and baseline Ridge on full training data and save to disk.

    Args:
        X: Training feature matrix.
        y_inr: Rupee targets.
        y_log: Log transformed targets.
        best_params: Tuned parameters dictionary.
        models_dir: Destination models directory.

    Returns:
        tuple[Path, Path, dict[str, Any]]: Final pipeline path, baseline path, model card.
    """
    config = get_config()
    out_dir = models_dir or config.paths.models_dir
    seed = config.project.random_seed
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Ridge Pipeline
    ridge_inner = Pipeline(
        [
            ("features", build_feature_pipeline(kind="linear")),
            ("model", get_candidate_models(seed)["ridge"][0]),
        ]
    )
    ridge_pipeline = TransformedTargetRegressor(
        regressor=ridge_inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    ridge_pipeline.fit(X, y_inr)
    baseline_path = out_dir / "baseline_ridge.joblib"
    joblib.dump(ridge_pipeline, baseline_path)
    logger.info("Saved baseline Ridge pipeline to %s", baseline_path)

    # 2. Select Winning Model based on lowest CV RMSE
    winner_name = min(best_params.keys(), key=lambda k: best_params[k]["best_cv_rmse"])
    winner_params = best_params[winner_name]["best_params"]
    winner_score = best_params[winner_name]["best_cv_rmse"]
    logger.info("Winning model selected: '%s' (CV RMSE=%.4f)", winner_name, winner_score)

    winner_estimator = create_tuned_estimator(winner_name, winner_params, seed)
    final_inner = Pipeline(
        [
            ("features", build_feature_pipeline(kind="tree")),
            ("model", winner_estimator),
        ]
    )
    final_pipeline = TransformedTargetRegressor(
        regressor=final_inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    final_pipeline.fit(X, y_inr)
    final_path = out_dir / "final_pipeline.joblib"
    joblib.dump(final_pipeline, final_path)
    logger.info("Saved final tuned pipeline to %s", final_path)

    # 3. Compute Out-of-fold Residual Quantiles for 80% Prediction Interval
    cv = make_cv_splitter()
    oof_log_preds = np.zeros(len(X), dtype=float)
    for tr_idx, val_idx in cv.split(X, y_log):
        fold_est = create_tuned_estimator(winner_name, winner_params, seed)
        fold_pipe = Pipeline(
            [
                ("features", build_feature_pipeline(kind="tree")),
                ("model", fold_est),
            ]
        )
        fold_pipe.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
        oof_log_preds[val_idx] = fold_pipe.predict(X.iloc[val_idx])

    log_residuals = y_log.to_numpy() - oof_log_preds
    q10 = float(np.percentile(log_residuals, 10))
    q90 = float(np.percentile(log_residuals, 90))
    oof_rmse_log = float(np.sqrt(mean_squared_error(y_log, oof_log_preds)))
    oof_mae_log = float(mean_absolute_error(y_log, oof_log_preds))
    oof_r2 = float(r2_score(y_log, oof_log_preds))

    # Read actual baseline scores from CV results table
    cv_table = pd.read_csv(config.paths.tables_dir / "cv_results.csv").set_index("model_id")
    dummy_rmse = float(cv_table.loc["dummy", "log_rmse_mean"])
    ridge_rmse = float(cv_table.loc["ridge", "log_rmse_mean"])
    pct_better_dummy = round(((dummy_rmse - oof_rmse_log) / dummy_rmse) * 100, 2)
    pct_better_ridge = round(((ridge_rmse - oof_rmse_log) / ridge_rmse) * 100, 2)

    # 4. Generate and save model card
    card = {
        "model_name": f"{winner_name} (Tuned)",
        "model_architecture": (
            f"TransformedTargetRegressor(Pipeline[DomainFeatureEngineer, "
            f"ColumnTransformer, {winner_name}])"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "training_sample_count": len(X),
        "target_transform": "log1p(selling_price) -> expm1",
        "best_hyperparameters": winner_params,
        "cross_validation": {
            "n_splits": config.splitting.cv_folds,
            "random_seed": seed,
            "oof_log_rmse": round(oof_rmse_log, 4),
            "oof_log_mae": round(oof_mae_log, 4),
            "oof_r2": round(oof_r2, 4),
            "improvement_vs_dummy_pct": pct_better_dummy,
            "improvement_vs_ridge_pct": pct_better_ridge,
        },
        "prediction_interval_80pct": {
            "method": "Empirical out-of-fold log residual quantiles",
            "residual_quantile_10": round(q10, 4),
            "residual_quantile_90": round(q90, 4),
            "formula": "[expm1(log_pred + q10), expm1(log_pred + q90)]",
        },
        "input_features": list(X.columns),
    }

    card_path = out_dir / "model_card.json"
    with card_path.open("w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    logger.info("Saved model card to %s", card_path)

    return final_path, baseline_path, card


def run_encoder_comparison(
    X: pd.DataFrame,
    y_log: pd.Series,
    best_params: dict[str, Any],
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare TargetEncoder vs one-hot for ``model`` on train-only CV (evidence for D-014).

    Two model families are checked: the Ridge feature probe and the tuned winner.

    Args:
        X: Training features.
        y_log: log1p(selling_price).
        best_params: Tuned parameters (from best_params.json).
        tables_dir: Output directory for encoder_comparison.csv.

    Returns:
        pd.DataFrame: One row per (model family, encoding).
    """
    out_dir = tables_dir or get_config().paths.tables_dir
    seed = get_config().project.random_seed
    winner = min(best_params, key=lambda k: best_params[k]["best_cv_rmse"])
    records = []
    for encoding in ["target", "onehot"]:
        ridge_rmse = cv_log_rmse(build_ridge_probe(BASE_NUMERIC_FEATURES, encoding), X, y_log)
        tuned_pipe = Pipeline(
            [
                ("features", build_feature_pipeline(kind="tree", model_encoding=encoding)),
                ("model", create_tuned_estimator(winner, best_params[winner]["best_params"], seed)),
            ]
        )
        tuned_rmse = cv_log_rmse(tuned_pipe, X, y_log)
        records.append(
            {
                "model_family": "ridge_probe",
                "model_encoding": encoding,
                "cv_log_rmse": round(ridge_rmse, 4),
            }
        )
        records.append(
            {
                "model_family": f"{winner}_tuned",
                "model_encoding": encoding,
                "cv_log_rmse": round(tuned_rmse, 4),
            }
        )
    table = pd.DataFrame(records).sort_values(["model_family", "model_encoding"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "encoder_comparison.csv", index=False)
    logger.info("Saved encoder comparison to %s", out_dir / "encoder_comparison.csv")
    return table


def main() -> None:
    """Command-line interface to execute training stages."""
    parser = argparse.ArgumentParser(description="Used Car Price Prediction Model Training CLI")
    parser.add_argument(
        "--stage",
        choices=["cv", "tune", "final", "checks", "all"],
        default="all",
        help="Pipeline execution stage (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting car price training runner stage: %s", args.stage)

    X, y_inr, y_log = load_training_data()
    config = get_config()

    if args.stage in ["cv", "all"]:
        run_cv_benchmark(X, y_inr, y_log)
        cv_df = pd.read_csv(config.paths.tables_dir / "cv_results.csv")
        plot_cv_comparison(cv_df)

    if args.stage in ["tune", "all"]:
        run_tuning(X, y_log)

    if args.stage in ["final", "all"]:
        best_params_path = config.paths.models_dir / "best_params.json"
        if not best_params_path.is_file():
            run_tuning(X, y_log)
        with best_params_path.open("r", encoding="utf-8") as f:
            best_params = json.load(f)
        fit_and_persist_pipelines(X, y_inr, y_log, best_params)

    if args.stage in ["checks", "all"]:
        with (config.paths.models_dir / "best_params.json").open("r", encoding="utf-8") as f:
            run_encoder_comparison(X, y_log, json.load(f))

    logger.info("Stage '%s' completed successfully.", args.stage)


if __name__ == "__main__":
    main()
