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
from carprice.features import build_feature_pipeline
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

    Returns:
        str: 7-character commit SHA or 'unknown'.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


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


def get_candidate_models(seed: int) -> dict[str, tuple[Any, str]]:
    """Instantiate the candidate regression algorithms paired with preprocessor kind.

    Args:
        seed: Random seed for deterministic reproducibility.

    Returns:
        dict[str, tuple[Any, str]]: Mapping of model_id to (estimator, preprocessor_kind).
    """
    return {
        "dummy": (DummyRegressor(strategy="median"), "tree"),
        "linear_regression": (LinearRegression(), "linear"),
        "ridge": (Ridge(alpha=1.0, random_state=seed), "linear"),
        "lasso": (Lasso(alpha=0.0005, random_state=seed, max_iter=3000), "linear"),
        "k_nearest_neighbors": (
            KNeighborsRegressor(n_neighbors=5, weights="distance"),
            "linear",
        ),
        "decision_tree": (
            DecisionTreeRegressor(max_depth=8, random_state=seed),
            "tree",
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            "tree",
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=seed),
            "tree",
        ),
        "xgboost": (
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=seed,
                n_jobs=-1,
            ),
            "tree",
        ),
        "lightgbm": (
            LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            "tree",
        ),
    }


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
    cv = KFold(n_splits=config.splitting.cv_folds, shuffle=True, random_state=seed)
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
    ax.set_xlabel("Mean 5-Fold Cross-Validation RMSE (Log1p Price Scale)")
    ax.set_title("Model Comparison: 5-Fold CV Log RMSE with Error Bars")
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


def tune_lightgbm_study(
    prepared_folds: list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]],
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run Optuna hyperparameter study for LightGBM on 5-fold CV log RMSE.

    Args:
        prepared_folds: Pre-transformed fold feature tuples.
        n_trials: Number of optimization trials.
        seed: Random seed.

    Returns:
        optuna.Study: Completed optimization study.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        fold_rmses: list[float] = []
        for X_tr, y_tr, X_val, y_val in prepared_folds:
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        return float(np.mean(fold_rmses))

    study.optimize(objective, n_trials=n_trials)
    return study


def tune_xgboost_study(
    prepared_folds: list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]],
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run Optuna hyperparameter study for XGBoost on 5-fold CV log RMSE.

    Args:
        prepared_folds: Pre-transformed fold feature tuples.
        n_trials: Number of optimization trials.
        seed: Random seed.

    Returns:
        optuna.Study: Completed optimization study.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
        }
        fold_rmses: list[float] = []
        for X_tr, y_tr, X_val, y_val in prepared_folds:
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        return float(np.mean(fold_rmses))

    study.optimize(objective, n_trials=n_trials)
    return study


def tune_hist_gb_study(
    prepared_folds: list[tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]],
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run Optuna hyperparameter study for HistGradientBoosting on 5-fold CV log RMSE.

    Args:
        prepared_folds: Pre-transformed fold feature tuples.
        n_trials: Number of optimization trials.
        seed: Random seed.

    Returns:
        optuna.Study: Completed optimization study.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "random_state": seed,
        }
        fold_rmses: list[float] = []
        for X_tr, y_tr, X_val, y_val in prepared_folds:
            model = HistGradientBoostingRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        return float(np.mean(fold_rmses))

    study.optimize(objective, n_trials=n_trials)
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
    cv = KFold(n_splits=config.splitting.cv_folds, shuffle=True, random_state=seed)

    logger.info("Precomputing leak-free CV folds for fast tuning...")
    prep_folds = prepare_cv_folds(X, y_log, cv, kind="tree")

    logger.info("Beginning Optuna hyperparameter optimization (%d trials each)", n_trials)
    studies: dict[str, optuna.Study] = {
        "hist_gradient_boosting": tune_hist_gb_study(prep_folds, n_trials, seed),
        "xgboost": tune_xgboost_study(prep_folds, n_trials, seed),
        "lightgbm": tune_lightgbm_study(prep_folds, n_trials, seed),
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
    ax1.set_xlabel("Optimization Trial Number")
    ax1.set_ylabel("Best Observed 5-Fold CV RMSE (Log Scale)")
    ax1.set_title("Hyperparameter Optimization Convergence History")
    ax1.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    ax1.grid(True, linestyle="--", alpha=0.6)
    hist_path = save_figure(fig1, figures_dir / "tuning_history.png")

    # 2. Parameter Importances for best model (LightGBM)
    lgbm_study = studies.get("lightgbm")
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    if lgbm_study:
        try:
            importances = optuna.importance.get_param_importances(lgbm_study)
            p_names = list(importances.keys())[::-1]
            p_vals = list(importances.values())[::-1]
            ax2.barh(p_names, p_vals, color=PRIMARY_TEAL, edgecolor=BORDER_COLOR, height=0.6)
            ax2.set_xlabel("Relative Parameter Importance (fANOVA)")
            ax2.set_title("Hyperparameter Importance for LightGBM Regressor")
            ax2.grid(axis="x", linestyle="--", alpha=0.6)
        except Exception as e:
            logger.warning("Could not compute fANOVA importances: %s", e)
            ax2.text(0.5, 0.5, "Importances not computed", ha="center")
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
            ("model", Ridge(alpha=1.0, random_state=seed)),
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
    cv = KFold(n_splits=config.splitting.cv_folds, shuffle=True, random_state=seed)
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


def main() -> None:
    """Command-line interface to execute training stages."""
    parser = argparse.ArgumentParser(description="Used Car Price Prediction Model Training CLI")
    parser.add_argument(
        "--stage",
        choices=["cv", "tune", "final", "all"],
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

    logger.info("Stage '%s' completed successfully.", args.stage)


if __name__ == "__main__":
    main()
