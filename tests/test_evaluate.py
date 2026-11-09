"""Tests for evaluate.py: metric functions on hand-computed examples, plus output checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carprice.config import get_config
from carprice.evaluate import (
    absolute_percentage_errors,
    adjusted_r2,
    age_band_labels,
    bootstrap_samples,
    calibration_verdict,
    mae,
    mape,
    median_ape,
    prediction_interval,
    price_band_labels,
    r2,
    rank_worst_segments,
    rmse,
    wilson_interval,
    within_tolerance,
)
from carprice.formatting import format_inr_indian, format_lakh
from carprice.provenance import load_sources, notebook_markdown, unsupported_numbers

# Hand-computed example: errors +10, -20, +30 on prices 100, 200, 300.
#   MAE = 60/3 = 20;  RMSE = sqrt((100 + 400 + 900)/3) = sqrt(466.67) = 21.6025
#   every |error|/price = 0.10  ->  MAPE = MdAPE = 10%
#   mean = 200, SS_tot = 20000, SS_res = 1400  ->  R² = 0.93; adj (n=3, p=1) = 0.86
Y_TRUE = np.array([100.0, 200.0, 300.0])
Y_PRED = np.array([110.0, 180.0, 330.0])


def test_mae_and_rmse_hand_example() -> None:
    """MAE and RMSE match the hand calculation."""
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(20.0)
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(np.sqrt(1400 / 3))


def test_r2_and_adjusted_r2_hand_example() -> None:
    """R² = 1 - 1400/20000 and its adjustment for one predictor."""
    value = r2(Y_TRUE, Y_PRED)
    assert value == pytest.approx(0.93)
    assert adjusted_r2(value, n_rows=3, n_predictors=1) == pytest.approx(0.86)


def test_r2_rejects_constant_target() -> None:
    """R² is undefined for a constant target."""
    with pytest.raises(ValueError, match="undefined"):
        r2(np.array([5.0, 5.0]), np.array([4.0, 6.0]))


def test_adjusted_r2_needs_enough_rows() -> None:
    """Adjusted R² requires n > p + 1."""
    with pytest.raises(ValueError):
        adjusted_r2(0.5, n_rows=3, n_predictors=2)


def test_percentage_metrics_hand_example() -> None:
    """All three relative errors are 10%."""
    assert mape(Y_TRUE, Y_PRED) == pytest.approx(10.0)
    assert median_ape(Y_TRUE, Y_PRED) == pytest.approx(10.0)


def test_median_ape_differs_from_mape_when_skewed() -> None:
    """One cheap car with a big relative miss moves MAPE but not MdAPE."""
    y_true = np.array([100.0, 100.0, 100.0, 10.0])
    y_pred = np.array([105.0, 95.0, 105.0, 20.0])
    assert mape(y_true, y_pred) == pytest.approx((5 + 5 + 5 + 100) / 4)
    assert median_ape(y_true, y_pred) == pytest.approx(5.0)


def test_within_tolerance_counts_share() -> None:
    """Within-tolerance is the percentage of rows with relative error <= tolerance."""
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([105.0, 115.0, 92.0, 130.0])
    assert within_tolerance(y_true, y_pred, 0.10) == pytest.approx(50.0)
    assert within_tolerance(y_true, y_pred, 0.20) == pytest.approx(75.0)


def test_percentage_errors_reject_non_positive_truth() -> None:
    """A zero true price would make percentage errors undefined."""
    with pytest.raises(ValueError, match="positive"):
        absolute_percentage_errors(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_wilson_interval_known_value() -> None:
    """8 successes of 10 at 95%: Wilson interval is about (0.4902, 0.9433)."""
    low, high = wilson_interval(8, 10, 0.95)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)


def test_calibration_verdict_cases() -> None:
    """Nominal inside the CI is well calibrated; outside is over-confident or conservative."""
    assert calibration_verdict(0.78, 0.82, 0.80) == "well calibrated"
    assert calibration_verdict(0.70, 0.75, 0.80).startswith("over-confident")
    assert calibration_verdict(0.85, 0.90, 0.80).startswith("conservative")


def test_prediction_interval_applies_log_quantiles() -> None:
    """Zero quantiles return the prediction; ±log(2) halves and doubles (1 + price)."""
    low, high = prediction_interval(np.array([99.0]), -np.log(2), np.log(2))
    assert low[0] == pytest.approx(49.0)
    assert high[0] == pytest.approx(199.0)


def test_price_band_labels_and_age_bands() -> None:
    """Band helpers return one edge more than labels and readable label text."""
    edges, labels = price_band_labels(pd.Series(np.arange(1, 101) * 1e5), n_bins=5)
    assert len(edges) == 6 and len(labels) == 5
    assert edges[0] == -np.inf and edges[-1] == np.inf
    bounds, age_labels = age_band_labels([0, 3, 6])
    assert age_labels == ["0-2 yrs", "3-5 yrs", "6+ yrs"]
    assert bounds[-1] == np.inf


def test_rank_worst_segments_skips_small_segments() -> None:
    """Segments below min_segment_rows are never ranked, however bad they look."""
    table = pd.DataFrame(
        {
            "segment": ["tiny", "a", "b", "c"],
            "n_test": [5, 100, 100, 100],
            "within_10pct": [0.0, 30.0, 50.0, 40.0],
            "mdape_pct": [90.0, 20.0, 8.0, 12.0],
        }
    )
    ranked = rank_worst_segments(table).set_index("segment")
    assert pd.isna(ranked.loc["tiny", "worst_rank"])
    assert ranked.loc["a", "worst_rank"] == 1
    assert ranked.loc["c", "worst_rank"] == 2


def test_bootstrap_is_seeded_and_brackets_point_estimate() -> None:
    """Same seed, same resamples; the resampled MAE centres on the sample MAE."""
    rng = np.random.default_rng(0)
    actual = rng.uniform(1e5, 1e6, 200)
    frame = pd.DataFrame(
        {
            "actual_inr": actual,
            "pred_final": actual * rng.uniform(0.8, 1.2, 200),
            "pred_ridge_baseline": actual * rng.uniform(0.7, 1.3, 200),
        }
    )
    first = bootstrap_samples(frame, n_resamples=200, seed=42)
    second = bootstrap_samples(frame, n_resamples=200, seed=42)
    np.testing.assert_array_equal(first["mae_inr"], second["mae_inr"])
    point = mae(frame["actual_inr"], frame["pred_final"])
    assert np.quantile(first["mae_inr"], 0.025) < point < np.quantile(first["mae_inr"], 0.975)


def test_indian_rupee_formatting() -> None:
    """A7 number format: Indian digit grouping and lakh."""
    assert format_inr_indian(550000) == "₹ 5,50,000"
    assert format_inr_indian(12345678) == "₹ 1,23,45,678"
    assert format_inr_indian(999) == "₹ 999"
    assert format_lakh(550000) == "₹ 5.50 lakh"


# ---------------------------------------------------------------------------
# Acceptance: no number in the written report may be absent from the tables
# ---------------------------------------------------------------------------


def _unsupported_numbers(text: str) -> list[str]:
    """Numbers in text that match no saved table, model file or config value."""
    return unsupported_numbers(text, load_sources(Path(".")))


def test_evaluation_summary_numbers_come_from_tables() -> None:
    """Every number in reports/evaluation_summary.md exists in a saved table or config."""
    text = (get_config().paths.reports_dir / "evaluation_summary.md").read_text(encoding="utf-8")
    assert _unsupported_numbers(text) == []


def test_notebook_03_markdown_numbers_come_from_tables() -> None:
    """Every number in notebook 03's markdown cells exists in a saved table or config."""
    text = notebook_markdown(Path("notebooks/03_model_evaluation.ipynb"))
    assert _unsupported_numbers(text) == []


def test_test_metrics_cover_all_models_and_scales() -> None:
    """Acceptance: every D-023 metric on both scales for all three models."""
    table = pd.read_csv(get_config().paths.tables_dir / "test_metrics.csv")
    assert set(table["model_id"]) == {"final", "ridge_baseline", "dummy_baseline"}
    required = {
        "mae_inr",
        "rmse_inr",
        "r2_inr",
        "adj_r2_inr",
        "mae_log",
        "rmse_log",
        "r2_log",
        "adj_r2_log",
        "mape_pct",
        "mdape_pct",
        "within_10pct",
        "within_20pct",
    }
    assert required <= set(table.columns)
    assert table[sorted(required)].notna().all().all()


def test_segment_table_has_six_segmentations_and_five_worst() -> None:
    """Acceptance: at least six segmentations and exactly five flagged worst segments."""
    table = pd.read_csv(get_config().paths.tables_dir / "segment_errors.csv")
    assert table["segmentation"].nunique() >= 6
    assert sorted(table["worst_rank"].dropna().astype(int)) == [1, 2, 3, 4, 5]


def test_number_checker_flags_invented_numbers() -> None:
    """The provenance check is not vacuous: an off-by-one rupee figure is caught."""
    assert _unsupported_numbers("Average miss ₹ 94,649 and 50.98% within 10%") == [
        "94,649",
        "50.98",
    ]
    assert _unsupported_numbers("Average miss ₹ 94,648 and 50.97% within 10%") == []
