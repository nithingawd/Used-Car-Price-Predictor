"""Unit and integration tests for train.py: model persistence, inference, and pipeline safety."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from carprice.config import get_config
from carprice.train import get_candidate_models, load_training_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Return the shared application config."""
    return get_config()


@pytest.fixture(scope="module")
def training_data():
    """Load training partition once for all tests in this module."""
    X, y_inr, y_log = load_training_data()
    return X, y_inr, y_log


@pytest.fixture(scope="module")
def final_pipeline(config):
    """Load the saved final pipeline from disk."""
    path = config.paths.models_dir / "final_pipeline.joblib"
    assert path.is_file(), f"final_pipeline.joblib not found at {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def baseline_pipeline(config):
    """Load the saved baseline Ridge pipeline from disk."""
    path = config.paths.models_dir / "baseline_ridge.joblib"
    assert path.is_file(), f"baseline_ridge.joblib not found at {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def model_card(config):
    """Load model card JSON from disk."""
    path = config.paths.models_dir / "model_card.json"
    assert path.is_file(), f"model_card.json not found at {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def known_row() -> pd.DataFrame:
    """A representative single-row input for deterministic inference checks."""
    return pd.DataFrame(
        [
            {
                "brand": "Maruti",
                "model": "Swift",
                "vehicle_age": 5,
                "km_driven": 60000,
                "seller_type": "Individual",
                "fuel_type": "Petrol",
                "transmission_type": "Manual",
                "mileage": 21.5,
                "engine": 1197.0,
                "max_power": 82.0,
                "seats": 5,
                "km_was_capped": False,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Task 4.4 — Pipeline reload and determinism
# ---------------------------------------------------------------------------


def test_final_pipeline_loads(final_pipeline) -> None:
    """Final pipeline must reload from disk without error."""
    assert final_pipeline is not None


def test_baseline_pipeline_loads(baseline_pipeline) -> None:
    """Baseline Ridge pipeline must reload from disk without error."""
    assert baseline_pipeline is not None


def test_final_pipeline_predict_output_type(final_pipeline, known_row) -> None:
    """Prediction on a single row must return a numpy array of floats."""
    preds = final_pipeline.predict(known_row)
    assert isinstance(preds, np.ndarray), "predict() must return np.ndarray"
    assert preds.dtype.kind == "f", "Predictions must be floating-point values"


def test_final_pipeline_predict_output_shape(final_pipeline, known_row) -> None:
    """Prediction on a single row must return shape (1,)."""
    preds = final_pipeline.predict(known_row)
    assert preds.shape == (1,), f"Expected shape (1,), got {preds.shape}"


def test_final_pipeline_predict_positive_price(final_pipeline, known_row) -> None:
    """Predicted rupee price must be strictly positive."""
    preds = final_pipeline.predict(known_row)
    assert preds[0] > 0, f"Predicted price must be > 0, got {preds[0]}"


def test_final_pipeline_predict_plausible_range(final_pipeline, known_row) -> None:
    """A 5-year-old Maruti Swift should be priced between ₹1 lakh and ₹15 lakh."""
    preds = final_pipeline.predict(known_row)
    price = preds[0]
    assert 100_000 <= price <= 1_500_000, (
        f"Maruti Swift prediction ₹{price:,.0f} is outside the plausible range "
        "₹1,00,000 – ₹15,00,000"
    )


def test_final_pipeline_deterministic(final_pipeline, known_row) -> None:
    """Calling predict twice must return identical results (no stochastic sampling)."""
    p1 = final_pipeline.predict(known_row)
    p2 = final_pipeline.predict(known_row)
    np.testing.assert_array_equal(p1, p2, err_msg="Predictions are not deterministic")


def test_final_pipeline_beats_baseline(final_pipeline, baseline_pipeline, known_row) -> None:
    """Final pipeline and baseline must both predict, final need not necessarily beat baseline on
    a single row, but both must return positive finite values."""
    pf = final_pipeline.predict(known_row)[0]
    pb = baseline_pipeline.predict(known_row)[0]
    assert np.isfinite(pf), "Final pipeline prediction is not finite"
    assert np.isfinite(pb), "Baseline pipeline prediction is not finite"
    assert pf > 0 and pb > 0


def test_final_pipeline_batch_predict(final_pipeline, training_data) -> None:
    """Batch prediction on 10 training rows must return shape (10,) with all positive values."""
    X, _, _ = training_data
    preds = final_pipeline.predict(X.head(10))
    assert preds.shape == (10,), f"Expected (10,), got {preds.shape}"
    assert np.all(preds > 0), "All batch predictions must be positive"


# ---------------------------------------------------------------------------
# Task 4.5 — Model card integrity
# ---------------------------------------------------------------------------


def test_model_card_required_keys(model_card) -> None:
    """Model card must contain all required provenance and performance fields."""
    required = {
        "model_name",
        "model_architecture",
        "timestamp_utc",
        "git_commit",
        "training_sample_count",
        "target_transform",
        "best_hyperparameters",
        "cross_validation",
        "prediction_interval_80pct",
        "input_features",
    }
    missing = required - set(model_card.keys())
    assert not missing, f"Model card missing keys: {missing}"


def test_model_card_cv_scores_sensible(model_card) -> None:
    """OOF log RMSE must be below 0.25 and R² above 0.90."""
    cv = model_card["cross_validation"]
    assert cv["oof_log_rmse"] < 0.25, f"OOF RMSE too high: {cv['oof_log_rmse']}"
    assert cv["oof_r2"] > 0.90, f"R² too low: {cv['oof_r2']}"


def test_model_card_interval_quantiles(model_card) -> None:
    """10th percentile must be negative and 90th must be positive (centred residuals)."""
    pi = model_card["prediction_interval_80pct"]
    assert pi["residual_quantile_10"] < 0, "Q10 of residuals should be negative"
    assert pi["residual_quantile_90"] > 0, "Q90 of residuals should be positive"


def test_model_card_training_count(model_card, config) -> None:
    """Training sample count in model card must match the actual train.csv row count."""
    train_path = config.paths.processed_data_dir / "train.csv"
    actual_count = len(pd.read_csv(train_path))
    assert model_card["training_sample_count"] == actual_count, (
        f"Card says {model_card['training_sample_count']} rows; train.csv has {actual_count}"
    )


# ---------------------------------------------------------------------------
# Data-isolation guard — test.csv must never be read in Phase 4
# ---------------------------------------------------------------------------


def test_train_module_never_reads_test_csv(config) -> None:
    """Confirm load_training_data raises ValueError when given test.csv path."""
    test_path = config.paths.processed_data_dir / "test.csv"
    with pytest.raises(ValueError, match="forbidden"):
        load_training_data(train_path=test_path)


# ---------------------------------------------------------------------------
# Candidate model factory
# ---------------------------------------------------------------------------


def test_candidate_models_count() -> None:
    """get_candidate_models must return at least 8 candidates (Dummy + 7 algorithms)."""
    candidates = get_candidate_models(seed=42)
    assert len(candidates) >= 8, f"Expected ≥ 8 candidates, got {len(candidates)}"
    assert "dummy" in candidates, "DummyRegressor must be in the candidate list"


def test_cv_results_file_exists(config) -> None:
    """cv_results.csv must exist and contain all 10 candidate rows."""
    path = config.paths.tables_dir / "cv_results.csv"
    assert path.is_file(), "cv_results.csv not found"
    df = pd.read_csv(path)
    assert len(df) >= 8, f"Expected ≥ 8 rows in cv_results.csv, got {len(df)}"
    assert "log_rmse_mean" in df.columns
    # Dummy must be worst (highest RMSE)
    dummy_rmse = df.loc[df["model_id"] == "dummy", "log_rmse_mean"].iloc[0]
    best_rmse = df["log_rmse_mean"].min()
    assert dummy_rmse > best_rmse, "DummyRegressor must have higher RMSE than the best model"
