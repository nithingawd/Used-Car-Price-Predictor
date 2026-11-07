"""Phase 5: one-time test-set evaluation, segment diagnostics, bootstrap and explainability.

This is the only module allowed to read ``data/processed/test.csv``. Metric choices and the
headline metric are justified in decisions.md (D-023) and were fixed before scoring.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from carprice.config import get_config
from carprice.data import validate_schema
from carprice.features import CATEGORICAL_FEATURES
from carprice.train import load_training_data

logger = logging.getLogger(__name__)

TARGET_COLUMNS: list[str] = ["selling_price", "log_selling_price"]
MODEL_LABELS: dict[str, str] = {
    "final": "Final model (tuned XGBoost)",
    "ridge_baseline": "Ridge baseline",
    "dummy_baseline": "Dummy baseline (train median)",
}
SEGMENT_COLUMNS: dict[str, str] = {
    "seg_price_band": "Price band (train quintiles)",
    "seg_luxury": "Luxury band",
    "seg_brand": "Brand (top 10)",
    "seg_fuel": "Fuel type",
    "seg_transmission": "Transmission",
    "seg_age_band": "Vehicle age band",
    "seg_seller": "Seller type",
}

# ---------------------------------------------------------------------------
# Metric functions (pure, verified in tests/test_evaluate.py)
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. Inputs are same-length arrays; returns a float."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. Inputs are same-length arrays; returns a float."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(diff)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot.

    Raises:
        ValueError: If y_true has zero variance (R² undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        raise ValueError("R² is undefined when y_true is constant")
    ss_res = float(np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2))
    return 1.0 - ss_res / ss_tot


def adjusted_r2(r2_value: float, n_rows: int, n_predictors: int) -> float:
    """Adjusted R² = 1 - (1 - R²)(n - 1)/(n - p - 1).

    Raises:
        ValueError: If n_rows <= n_predictors + 1.
    """
    if n_rows <= n_predictors + 1:
        raise ValueError(f"Need n_rows > n_predictors + 1 (got n={n_rows}, p={n_predictors})")
    return 1.0 - (1.0 - r2_value) * (n_rows - 1) / (n_rows - n_predictors - 1)


def absolute_percentage_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row |pred - true| / true, as fractions.

    Raises:
        ValueError: If any true value is not strictly positive.
    """
    y_true = np.asarray(y_true, dtype=float)
    if (y_true <= 0).any():
        raise ValueError("Percentage errors need strictly positive true values")
    return np.abs(np.asarray(y_pred, dtype=float) - y_true) / y_true


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    return float(np.mean(absolute_percentage_errors(y_true, y_pred)) * 100)


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error, in percent."""
    return float(np.median(absolute_percentage_errors(y_true, y_pred)) * 100)


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> float:
    """Percent of predictions whose absolute percentage error is <= tolerance (a fraction)."""
    return float(np.mean(absolute_percentage_errors(y_true, y_pred) <= tolerance) * 100)


def wilson_interval(successes: int, n_trials: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; returns (low, high) as fractions.

    Raises:
        ValueError: If n_trials is not positive.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p_hat = successes / n_trials
    denom = 1 + z**2 / n_trials
    centre = (p_hat + z**2 / (2 * n_trials)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2)) / denom
    return float(centre - half), float(centre + half)


def calibration_verdict(ci_low: float, ci_high: float, nominal: float) -> str:
    """Classify interval coverage: nominal inside the CI is 'well calibrated'."""
    if ci_high < nominal:
        return "over-confident (covers too few cars)"
    if ci_low > nominal:
        return "conservative (covers more cars than promised)"
    return "well calibrated"


def regression_metrics(
    y_true_inr: np.ndarray, y_pred_inr: np.ndarray, n_predictors: int
) -> dict[str, float]:
    """Every D-023 metric on both scales for one model.

    Args:
        y_true_inr: True prices in rupees.
        y_pred_inr: Predicted prices in rupees.
        n_predictors: Columns entering the estimator (for adjusted R²).

    Returns:
        dict[str, float]: Rounded metrics (rupees 0 dp, log 4 dp, percentages 2 dp).
    """
    n = len(y_true_inr)
    log_true, log_pred = np.log1p(y_true_inr), np.log1p(y_pred_inr)
    r2_inr, r2_log = r2(y_true_inr, y_pred_inr), r2(log_true, log_pred)
    signed_pct = (np.asarray(y_pred_inr) - np.asarray(y_true_inr)) / np.asarray(y_true_inr)
    metrics = {
        "mae_inr": round(mae(y_true_inr, y_pred_inr)),
        "rmse_inr": round(rmse(y_true_inr, y_pred_inr)),
        "r2_inr": round(r2_inr, 4),
        "adj_r2_inr": round(adjusted_r2(r2_inr, n, n_predictors), 4),
        "mae_log": round(mae(log_true, log_pred), 4),
        "rmse_log": round(rmse(log_true, log_pred), 4),
        "r2_log": round(r2_log, 4),
        "adj_r2_log": round(adjusted_r2(r2_log, n, n_predictors), 4),
        "mape_pct": round(mape(y_true_inr, y_pred_inr), 2),
        "mdape_pct": round(median_ape(y_true_inr, y_pred_inr), 2),
        "median_signed_pct_error": round(float(np.median(signed_pct)) * 100, 2),
    }
    for tol in get_config().evaluation.within_tolerances:
        metrics[f"within_{round(tol * 100)}pct"] = round(
            within_tolerance(y_true_inr, y_pred_inr, tol), 2
        )
    return metrics


# ---------------------------------------------------------------------------
# Loading data and models
# ---------------------------------------------------------------------------


def load_test_data(test_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the locked test partition (Phase 5 only).

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features and true prices in rupees.

    Raises:
        FileNotFoundError: If test.csv is missing.
    """
    path = test_path or (get_config().paths.processed_data_dir / "test.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found at: {path}")
    df = pd.read_csv(path)
    validate_schema(df, is_cleaned=True)
    logger.info("Loaded locked test partition: %d rows", len(df))
    return df.drop(columns=TARGET_COLUMNS, errors="ignore"), df["selling_price"]


def load_model_card() -> dict[str, Any]:
    """Read models/model_card.json written in Phase 4."""
    path = get_config().paths.models_dir / "model_card.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fit_dummy_baseline(X_train: pd.DataFrame, y_train_inr: pd.Series) -> TransformedTargetRegressor:
    """Fit the median-of-log-price dummy on train.csv, as in the Phase 4 benchmark."""
    params = get_config().models.candidate_params["dummy"]
    dummy = TransformedTargetRegressor(
        regressor=DummyRegressor(**params), func=np.log1p, inverse_func=np.expm1
    )
    return dummy.fit(X_train, y_train_inr)


def count_estimator_inputs(model: TransformedTargetRegressor) -> int:
    """Number of columns entering the estimator after preprocessing (0 for the dummy)."""
    inner = model.regressor_
    if not hasattr(inner, "named_steps"):
        return 0
    preprocess = inner.named_steps["features"].named_steps["preprocess"]
    return len(preprocess.get_feature_names_out())


def load_models() -> dict[str, TransformedTargetRegressor]:
    """Load the saved final and Ridge pipelines and fit the dummy baseline on train only."""
    models_dir = get_config().paths.models_dir
    X_train, y_train, _ = load_training_data()
    return {
        "final": joblib.load(models_dir / "final_pipeline.joblib"),
        "ridge_baseline": joblib.load(models_dir / "baseline_ridge.joblib"),
        "dummy_baseline": fit_dummy_baseline(X_train, y_train),
    }


def prediction_interval(
    pred_inr: np.ndarray, q_low: float, q_high: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply D-020 log-residual quantiles: [expm1(log1p(pred)+q_low), expm1(log1p(pred)+q_high)]."""
    log_pred = np.log1p(np.asarray(pred_inr, dtype=float))
    return np.expm1(log_pred + q_low), np.expm1(log_pred + q_high)


def build_prediction_frame(
    models: dict[str, TransformedTargetRegressor], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Score every model once on the test set and add errors and the final model's interval.

    Returns:
        pd.DataFrame: Test features plus actual, per-model predictions, interval and errors.
    """
    card = load_model_card()["prediction_interval_80pct"]
    frame = X_test.copy()
    frame["actual_inr"] = y_test.to_numpy()
    for name, model in models.items():
        frame[f"pred_{name}"] = model.predict(X_test)
    low, high = prediction_interval(
        frame["pred_final"], card["residual_quantile_10"], card["residual_quantile_90"]
    )
    frame["interval_low"], frame["interval_high"] = low, high
    frame["inside_interval"] = (frame["actual_inr"] >= low) & (frame["actual_inr"] <= high)
    frame["abs_error_inr"] = (frame["pred_final"] - frame["actual_inr"]).abs()
    frame["signed_pct_error"] = (frame["pred_final"] / frame["actual_inr"] - 1) * 100
    frame["log_residual"] = np.log1p(frame["actual_inr"]) - np.log1p(frame["pred_final"])
    return frame


# ---------------------------------------------------------------------------
# Test metrics, calibration and residual diagnostics
# ---------------------------------------------------------------------------


def compute_test_metrics(
    pred_df: pd.DataFrame, models: dict[str, TransformedTargetRegressor]
) -> pd.DataFrame:
    """One row per model with every D-023 metric on both scales."""
    rows = []
    for name, model in models.items():
        metrics = regression_metrics(
            pred_df["actual_inr"].to_numpy(),
            pred_df[f"pred_{name}"].to_numpy(),
            count_estimator_inputs(model),
        )
        rows.append(
            {"model_id": name, "model": MODEL_LABELS[name], "n_test": len(pred_df), **metrics}
        )
    return pd.DataFrame(rows)


def compute_interval_coverage(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Coverage of the 80% interval with a Wilson CI and a calibration verdict (D-023)."""
    evaluation = get_config().evaluation
    n, covered = len(pred_df), int(pred_df["inside_interval"].sum())
    low, high = wilson_interval(covered, n, evaluation.confidence_level)
    width_pct = (pred_df["interval_high"] - pred_df["interval_low"]) / pred_df["pred_final"]
    record = {
        "n_test": n,
        "covered": covered,
        "nominal_pct": round(evaluation.interval_nominal_coverage * 100, 2),
        "coverage_pct": round(covered / n * 100, 2),
        "ci_low_pct": round(low * 100, 2),
        "ci_high_pct": round(high * 100, 2),
        "below_interval_pct": round(
            float((pred_df["actual_inr"] < pred_df["interval_low"]).mean()) * 100, 2
        ),
        "above_interval_pct": round(
            float((pred_df["actual_inr"] > pred_df["interval_high"]).mean()) * 100, 2
        ),
        "median_width_pct_of_pred": round(float(width_pct.median()) * 100, 2),
        "verdict": calibration_verdict(low, high, evaluation.interval_nominal_coverage),
    }
    return pd.DataFrame([record])


def compute_residual_stats(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Shape of the final model's log residuals (supports the QQ plot reading)."""
    resid = pred_df["log_residual"].to_numpy()
    std = float(np.std(resid, ddof=1))
    record = {
        "mean_log_residual": round(float(np.mean(resid)), 4),
        "median_log_residual": round(float(np.median(resid)), 4),
        "std_log_residual": round(std, 4),
        "skewness": round(float(stats.skew(resid)), 3),
        "excess_kurtosis": round(float(stats.kurtosis(resid)), 3),
        "share_beyond_2sd_pct": round(float(np.mean(np.abs(resid) > 2 * std)) * 100, 2),
        "normal_share_beyond_2sd_pct": round(float(2 * stats.norm.sf(2)) * 100, 2),
    }
    return pd.DataFrame([record])


def compare_train_and_test(
    test_metrics: pd.DataFrame, pred_train_log_rmse: dict[str, float]
) -> pd.DataFrame:
    """Train in-sample, train 5-fold OOF CV and test scores side by side (log scale).

    Args:
        test_metrics: Output of compute_test_metrics().
        pred_train_log_rmse: In-sample metrics of the final model on train.csv.

    Returns:
        pd.DataFrame: One row per evaluation split with the gap versus CV.
    """
    cv = load_model_card()["cross_validation"]
    final = test_metrics.set_index("model_id").loc["final"]
    rows = [
        {"split": "train (in-sample fit)", **pred_train_log_rmse},
        {
            "split": "train (5-fold out-of-fold CV)",
            "rmse_log": cv["oof_log_rmse"],
            "mae_log": cv["oof_log_mae"],
            "r2_log": cv["oof_r2"],
        },
        {
            "split": "test (held out, scored once)",
            "rmse_log": final["rmse_log"],
            "mae_log": final["mae_log"],
            "r2_log": final["r2_log"],
        },
    ]
    table = pd.DataFrame(rows)
    table["rmse_gap_vs_cv_pct"] = ((table["rmse_log"] / cv["oof_log_rmse"] - 1) * 100).round(2)
    return table


def train_in_sample_metrics(final_model: TransformedTargetRegressor) -> dict[str, float]:
    """Final model's fit on its own training data (log scale), to expose memorisation."""
    X_train, y_train, _ = load_training_data()
    log_true, log_pred = np.log1p(y_train), np.log1p(final_model.predict(X_train))
    return {
        "rmse_log": round(rmse(log_true, log_pred), 4),
        "mae_log": round(mae(log_true, log_pred), 4),
        "r2_log": round(r2(log_true, log_pred), 4),
    }


# ---------------------------------------------------------------------------
# Segments and largest errors
# ---------------------------------------------------------------------------


def price_band_labels(train_prices: pd.Series, n_bins: int) -> tuple[list[float], list[str]]:
    """Quantile edges of *train* prices and readable labels (edges open at both ends)."""
    quantiles = np.quantile(train_prices, np.linspace(0, 1, n_bins + 1))
    edges = [-np.inf, *quantiles[1:-1], np.inf]
    labels = [
        f"Q{i + 1}: ₹{quantiles[i] / 1e5:.1f}-{quantiles[i + 1] / 1e5:.1f} lakh"
        for i in range(n_bins)
    ]
    return edges, labels


def age_band_labels(edges: list[int]) -> tuple[list[float], list[str]]:
    """Left-closed age bands from config edges, last band open-ended (e.g. '12+ yrs')."""
    bounds = [*edges, np.inf]
    labels = [f"{lo}-{hi - 1} yrs" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return bounds, [*labels, f"{edges[-1]}+ yrs"]


def assign_segments(pred_df: pd.DataFrame, train_prices: pd.Series) -> pd.DataFrame:
    """Add one column per segmentation in SEGMENT_COLUMNS (bands use train-set edges)."""
    evaluation = get_config().evaluation
    out = pred_df.copy()
    price_edges, price_labels = price_band_labels(train_prices, evaluation.price_band_bins)
    out["seg_price_band"] = pd.cut(out["actual_inr"], price_edges, labels=price_labels)
    lakh = evaluation.luxury_threshold_inr / 1e5
    is_lux = out["actual_inr"] > evaluation.luxury_threshold_inr
    out["seg_luxury"] = np.where(is_lux, f"Above ₹{lakh:.0f} lakh", f"₹{lakh:.0f} lakh or below")
    top_brands = out["brand"].value_counts().head(evaluation.top_brands).index
    out["seg_brand"] = out["brand"].where(out["brand"].isin(top_brands), "Other brands")
    age_edges, age_labels = age_band_labels(evaluation.age_band_edges)
    out["seg_age_band"] = pd.cut(out["vehicle_age"], age_edges, labels=age_labels, right=False)
    out["seg_fuel"] = out["fuel_type"]
    out["seg_transmission"] = out["transmission_type"]
    out["seg_seller"] = out["seller_type"]
    return out


def segment_metrics(group: pd.DataFrame) -> dict[str, float]:
    """Error summary for one segment of the final model's test predictions."""
    actual, pred = group["actual_inr"].to_numpy(), group["pred_final"].to_numpy()
    tolerances = get_config().evaluation.within_tolerances
    record: dict[str, float] = {
        "n_test": len(group),
        "median_actual_inr": round(float(np.median(actual))),
        "mae_inr": round(mae(actual, pred)),
        "mape_pct": round(mape(actual, pred), 2),
        "mdape_pct": round(median_ape(actual, pred), 2),
        "median_signed_pct_error": round(float(group["signed_pct_error"].median()), 2),
        "interval_coverage_pct": round(float(group["inside_interval"].mean()) * 100, 2),
    }
    for tol in tolerances:
        record[f"within_{round(tol * 100)}pct"] = round(within_tolerance(actual, pred, tol), 2)
    return record


def rank_worst_segments(table: pd.DataFrame) -> pd.DataFrame:
    """Flag the worst segments: lowest within-10% share among segments with enough rows (D-023)."""
    evaluation = get_config().evaluation
    eligible = table[table["n_test"] >= evaluation.min_segment_rows]
    worst = eligible.sort_values(["within_10pct", "mdape_pct"], ascending=[True, False])
    worst_ids = worst.head(evaluation.worst_segments).index
    out = table.copy()
    out["eligible_for_ranking"] = out["n_test"] >= evaluation.min_segment_rows
    out["worst_rank"] = pd.Series(range(1, len(worst_ids) + 1), index=worst_ids)
    return out


def compute_segment_errors(segmented: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every segment of every segmentation, worst five flagged."""
    rows = []
    for column, title in SEGMENT_COLUMNS.items():
        for segment, group in segmented.groupby(column, observed=True, sort=True):
            rows.append({"segmentation": title, "segment": str(segment), **segment_metrics(group)})
    return rank_worst_segments(pd.DataFrame(rows))


def top_absolute_errors(pred_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """The n largest rupee errors of the final model, with every input feature."""
    feature_cols = [c for c in pred_df.columns if not c.startswith(("pred_", "seg_"))]
    columns = [*feature_cols, "pred_final"]
    top = pred_df.nlargest(n_rows, "abs_error_inr")[columns].copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top.round(
        {
            "pred_final": 0,
            "interval_low": 0,
            "interval_high": 0,
            "abs_error_inr": 0,
            "signed_pct_error": 2,
            "log_residual": 4,
        }
    )


def summarise_top_errors(top: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Counts behind the written explanation of the largest-error patterns.

    Args:
        top: Output of top_absolute_errors().
        pred_df: Full prediction frame (for the share of total error).

    Returns:
        pd.DataFrame: One row of counts and shares.
    """
    evaluation = get_config().evaluation
    luxury_brands = set(get_config().features.luxury_brands)
    facts = {
        "n_rows": len(top),
        "actual_above_luxury_threshold": int(
            (top["actual_inr"] > evaluation.luxury_threshold_inr).sum()
        ),
        "luxury_brand": int(top["brand"].isin(luxury_brands).sum()),
        "over_predicted": int((top["signed_pct_error"] > 0).sum()),
        "under_predicted": int((top["signed_pct_error"] < 0).sum()),
        "outside_interval": int((~top["inside_interval"]).sum()),
        "automatic": int((top["transmission_type"] == "Automatic").sum()),
        "dealer_sellers": int(top["seller_type"].str.contains("Dealer").sum()),
        "median_vehicle_age": float(top["vehicle_age"].median()),
        "median_actual_inr": round(float(top["actual_inr"].median())),
        "share_of_total_abs_error_pct": round(
            float(top["abs_error_inr"].sum() / pred_df["abs_error_inr"].sum()) * 100, 2
        ),
        "share_of_test_rows_pct": round(len(top) / len(pred_df) * 100, 2),
    }
    return pd.DataFrame([facts])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_samples(pred_df: pd.DataFrame, n_resamples: int, seed: int) -> dict[str, np.ndarray]:
    """Resampled headline metrics: each array has one value per bootstrap resample.

    Rows are drawn with replacement from the test set (same size as the test set).
    """
    rng = np.random.default_rng(seed)
    n = len(pred_df)
    idx = rng.integers(0, n, size=(n_resamples, n))
    actual = pred_df["actual_inr"].to_numpy()
    abs_final = np.abs(pred_df["pred_final"].to_numpy() - actual)
    abs_ridge = np.abs(pred_df["pred_ridge_baseline"].to_numpy() - actual)
    ape = abs_final / actual
    log_true = np.log1p(actual)
    sq = (log_true - np.log1p(pred_df["pred_final"].to_numpy())) ** 2
    log_sample = log_true[idx]
    ss_tot = ((log_sample - log_sample.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    samples = {
        "mae_inr": abs_final[idx].mean(axis=1),
        "mdape_pct": np.median(ape[idx], axis=1) * 100,
        "rmse_log": np.sqrt(sq[idx].mean(axis=1)),
        "r2_log": 1 - sq[idx].sum(axis=1) / ss_tot,
        "mae_inr_final_minus_ridge": (abs_final[idx] - abs_ridge[idx]).mean(axis=1),
    }
    for tol in get_config().evaluation.within_tolerances:
        samples[f"within_{round(tol * 100)}pct"] = (ape <= tol)[idx].mean(axis=1) * 100
    return samples


def bootstrap_confidence_intervals(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Percentile bootstrap CIs (D-023) for the final model's headline metrics."""
    config = get_config()
    evaluation = config.evaluation
    samples = bootstrap_samples(pred_df, evaluation.bootstrap_resamples, config.project.random_seed)
    alpha = (1 - evaluation.confidence_level) / 2
    decimals = {"mae_inr": 0, "mae_inr_final_minus_ridge": 0, "rmse_log": 4, "r2_log": 4}
    rows = []
    for metric, values in samples.items():
        dp = decimals.get(metric, 2)
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": round(float(values.mean()), dp),
                "ci_low": round(float(np.quantile(values, alpha)), dp),
                "ci_high": round(float(np.quantile(values, 1 - alpha)), dp),
                "n_resamples": evaluation.bootstrap_resamples,
                "confidence_level": evaluation.confidence_level,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def save_table(table: pd.DataFrame, name: str) -> Path:
    """Write a table to reports/tables/<name>.csv and return its path."""
    out_dir = get_config().paths.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    table.to_csv(path, index=False)
    logger.info("Saved %s", path)
    return path


def load_report_table(name: str) -> pd.DataFrame:
    """Read reports/tables/<name>.csv (used by the notebook and the GUI)."""
    return pd.read_csv(get_config().paths.tables_dir / f"{name}.csv")


# ---------------------------------------------------------------------------
# Explainability (task 5.3)
# ---------------------------------------------------------------------------


def log_rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative log-scale RMSE (higher is better), for permutation importance."""
    return -rmse(np.log1p(y_true), np.log1p(y_pred))


def compute_permutation_importance(
    final_model: TransformedTargetRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Permutation importance on the test set for every raw input column.

    Importance = how much the error grows when one column is shuffled: rupee MAE (headline)
    and log RMSE (training objective). Columns the pipeline ignores score exactly zero.
    """
    config = get_config()
    result = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring={"mae_inr": "neg_mean_absolute_error", "rmse_log": make_scorer(log_rmse_score)},
        n_repeats=config.evaluation.permutation_repeats,
        random_state=config.project.random_seed,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mae_inr_increase": result["mae_inr"].importances_mean.round(0),
            "mae_inr_increase_std": result["mae_inr"].importances_std.round(0),
            "rmse_log_increase": result["rmse_log"].importances_mean.round(4),
            "rmse_log_increase_std": result["rmse_log"].importances_std.round(4),
        }
    )
    return table.sort_values("rmse_log_increase", ascending=False).reset_index(drop=True)


def explanation_frame(final_model: TransformedTargetRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Rows as the estimator sees them: engineered, encoded, with output column names."""
    features = final_model.regressor_.named_steps["features"]
    names = features.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame(features.transform(X), columns=names, index=X.index)


def compute_shap_values(
    final_model: TransformedTargetRegressor, X_sample: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeExplainer SHAP values (log-price units) for a sample of rows."""
    import shap

    transformed = explanation_frame(final_model, X_sample)
    explainer = shap.TreeExplainer(final_model.regressor_.named_steps["model"])
    return np.asarray(explainer.shap_values(transformed)), transformed


def original_feature_of(column: str) -> str:
    """Map a one-hot output column such as 'brand_Bmw' back to its input feature."""
    for categorical in CATEGORICAL_FEATURES:
        if column.startswith(f"{categorical}_"):
            return categorical
    return column


def group_shap_by_feature(shap_values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns so each input feature has one value per row."""
    frame = pd.DataFrame(shap_values, columns=columns)
    groups = [original_feature_of(c) for c in columns]
    return frame.T.groupby(groups, sort=False).sum().T


def shap_importance_table(grouped: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per input feature, with its share of the total."""
    mean_abs = grouped.abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": mean_abs.index,
            "mean_abs_shap_log": mean_abs.to_numpy().round(4),
            "share_pct": (mean_abs / mean_abs.sum() * 100).to_numpy().round(2),
        }
    )


def sample_for_shap(X_test: pd.DataFrame) -> pd.DataFrame:
    """Seeded sample of test rows for SHAP (size from evaluation.shap_sample_rows)."""
    config = get_config()
    n_rows = min(config.evaluation.shap_sample_rows, len(X_test))
    return X_test.sample(n=n_rows, random_state=config.project.random_seed)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_test_tables(
    models: dict[str, TransformedTargetRegressor], segmented: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Every task 5.2 table, computed from the single scoring pass."""
    test_metrics = compute_test_metrics(segmented, models)
    top = top_absolute_errors(segmented, get_config().evaluation.top_errors)
    return {
        "test_metrics": test_metrics,
        "interval_coverage": compute_interval_coverage(segmented),
        "residual_stats": compute_residual_stats(segmented),
        "segment_errors": compute_segment_errors(segmented),
        "top_errors": top,
        "top_error_patterns": summarise_top_errors(top, segmented),
        "bootstrap_ci": bootstrap_confidence_intervals(segmented),
        "train_vs_test": compare_train_and_test(
            test_metrics, train_in_sample_metrics(models["final"])
        ),
        "test_predictions": segmented.round(
            {
                "pred_final": 0,
                "pred_ridge_baseline": 0,
                "pred_dummy_baseline": 0,
                "interval_low": 0,
                "interval_high": 0,
                "abs_error_inr": 0,
                "signed_pct_error": 2,
                "log_residual": 4,
            }
        ),
    }


def run_test_evaluation() -> dict[str, pd.DataFrame]:
    """Task 5.2: score the locked test set once, save every table and figure."""
    from carprice.eval_plots import plot_test_figures

    models = load_models()
    X_test, y_test = load_test_data()
    _, y_train, _ = load_training_data()
    segmented = assign_segments(build_prediction_frame(models, X_test, y_test), y_train)
    tables = build_test_tables(models, segmented)
    for name, table in tables.items():
        save_table(table, name)
    plot_test_figures(segmented, tables)
    return tables


def run_explainability() -> dict[str, pd.DataFrame]:
    """Task 5.3: permutation importance (full test set) and SHAP (seeded sample)."""
    from carprice.eval_plots import plot_explainability_figures

    final_model = joblib.load(get_config().paths.models_dir / "final_pipeline.joblib")
    X_test, y_test = load_test_data()
    sample = sample_for_shap(X_test)
    shap_values, transformed = compute_shap_values(final_model, sample)
    grouped = group_shap_by_feature(shap_values, list(transformed.columns))
    tables = {
        "permutation_importance": compute_permutation_importance(final_model, X_test, y_test),
        "shap_importance": shap_importance_table(grouped),
    }
    for name, table in tables.items():
        save_table(table, name)
    plot_explainability_figures(shap_values, transformed, grouped, sample, tables)
    return tables


def main() -> None:
    """Command-line entry point: python -m carprice.evaluate --stage {test,explain,all}."""
    parser = argparse.ArgumentParser(description="Phase 5 model evaluation")
    parser.add_argument("--stage", choices=["test", "explain", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.stage in ["test", "all"]:
        run_test_evaluation()
    if args.stage in ["explain", "all"]:
        run_explainability()
    logger.info("Evaluation stage '%s' completed.", args.stage)


if __name__ == "__main__":
    main()
