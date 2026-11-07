"""Tests for predict.py (PricePredictor, validation, CLI) and the Phase 6 output files."""

from dataclasses import asdict

import pandas as pd
import pytest

from carprice.config import get_config
from carprice.predict import (
    CarSpec,
    InputError,
    PredictionResult,
    PricePredictor,
    build_input_schema,
    main,
)

SWIFT = {
    "brand": "Maruti",
    "model": "Swift",
    "vehicle_age": 5,
    "km_driven": 60000,
    "seller_type": "Individual",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 21.5,
    "engine": 1197,
    "max_power": 82.0,
    "seats": 5,
}
# Regression anchor recorded from the Phase 4 pipeline (commit 6e2e4c7). If the model is
# retrained this value changes and the test must be updated deliberately.
SWIFT_EXPECTED_INR = 431383
EXPECTED_TOLERANCE = 0.005


@pytest.fixture(scope="module")
def predictor() -> PricePredictor:
    """Load the predictor once for the module."""
    return PricePredictor()


def problems_for(predictor: PricePredictor, **changes: object) -> list[str]:
    """Problems raised for SWIFT with some fields changed."""
    with pytest.raises(InputError) as err:
        predictor.predict_one({**SWIFT, **changes})
    return err.value.problems


# ---------------------------------------------------------------------------
# Known input, typed output
# ---------------------------------------------------------------------------


def test_known_input_gives_expected_price(predictor: PricePredictor) -> None:
    """A fixed Maruti Swift prices at the recorded value (±0.5%)."""
    result = predictor.predict_one(SWIFT)
    assert isinstance(result, PredictionResult)
    assert result.price_inr == pytest.approx(SWIFT_EXPECTED_INR, rel=EXPECTED_TOLERANCE)


def test_prediction_matches_raw_pipeline(predictor: PricePredictor) -> None:
    """predict_one adds no hidden transformation on top of the saved pipeline."""
    frame = CarSpec.from_mapping(SWIFT).to_frame()
    assert predictor.predict_one(SWIFT).price_inr == pytest.approx(
        float(predictor.pipeline.predict(frame)[0])
    )


def test_interval_and_metadata(predictor: PricePredictor) -> None:
    """Bounds bracket the estimate; level and model version come from config and card."""
    result = predictor.predict_one(SWIFT)
    assert result.lower_inr < result.price_inr < result.upper_inr
    assert result.interval_level == get_config().evaluation.interval_nominal_coverage
    assert predictor.card["git_commit"] in result.model_version
    assert result.price_text.startswith("₹ ") and "lakh" in result.price_text


def test_labels_are_normalised(predictor: PricePredictor) -> None:
    """Lower-case, padded labels and '60,000' give the same price as clean input."""
    messy = {
        **SWIFT,
        "brand": "  maruti ",
        "model": "swift",
        "fuel_type": "petrol",
        "seller_type": "individual",
        "transmission_type": "MANUAL",
        "km_driven": "60,000",
    }
    assert predictor.predict_one(messy).price_inr == pytest.approx(
        predictor.predict_one(SWIFT).price_inr
    )


def test_cautions_for_weak_segments(predictor: PricePredictor) -> None:
    """An old luxury car carries both cautions; a common young car carries none."""
    old_bmw = {
        **SWIFT,
        "brand": "Bmw",
        "model": "5",
        "vehicle_age": 11,
        "fuel_type": "Diesel",
        "transmission_type": "Automatic",
        "engine": 1995,
        "max_power": 190.0,
    }
    cautions = predictor.predict_one(old_bmw).cautions
    assert any("years old" in c for c in cautions)
    assert any("Luxury" in c for c in cautions)
    assert predictor.predict_one(SWIFT).cautions == ()


# ---------------------------------------------------------------------------
# Invalid input raises friendly errors
# ---------------------------------------------------------------------------


def test_unknown_brand(predictor: PricePredictor) -> None:
    """Unknown brands are refused with the list of known brands."""
    (problem,) = problems_for(predictor, brand="Tesla")
    assert "Brand 'Tesla' is not in the training data" in problem


def test_model_from_another_brand(predictor: PricePredictor) -> None:
    """Brand/model consistency names the real maker (D-030)."""
    assert problems_for(predictor, brand="Hyundai") == [
        "Model 'Swift' is made by Maruti, not Hyundai."
    ]


def test_unknown_model(predictor: PricePredictor) -> None:
    """An unknown model lists that brand's models."""
    (problem,) = problems_for(predictor, model="Zen Estilo X")
    assert "not a known Maruti model" in problem and "Alto" in problem


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"vehicle_age": 40}, "Vehicle age (years) must be between"),
        ({"km_driven": "lots"}, "Kilometres driven must be a number"),
        ({"vehicle_age": 4.5}, "must be a whole number"),
        ({"fuel_type": "Electric"}, "Fuel type 'Electric' is not recognised"),
        ({"seats": 3}, "Seats must be one of"),
        ({"engine": None}, "Engine size (cc) is missing"),
        ({"mileage": float("inf")}, "finite"),
    ],
)
def test_invalid_values(predictor: PricePredictor, changes: dict, fragment: str) -> None:
    """Each rule in D-029 produces its own plain-language message."""
    assert any(fragment in p for p in problems_for(predictor, **changes))


def test_all_problems_reported_together(predictor: PricePredictor) -> None:
    """Several mistakes are reported at once, not one at a time."""
    problems = problems_for(predictor, vehicle_age=40, seats=3, km_driven="lots", brand="Hyundai")
    assert len(problems) == 4


def test_input_error_is_value_error() -> None:
    """InputError is a ValueError, as the brief requires."""
    assert issubclass(InputError, ValueError)


# ---------------------------------------------------------------------------
# Batch and single agree
# ---------------------------------------------------------------------------


def test_batch_and_single_agree(predictor: PricePredictor) -> None:
    """predict_batch and predict_one give the same numbers for the same cars."""
    rows = pd.read_csv(get_config().paths.processed_data_dir / "train.csv").head(25)
    batch = predictor.predict_batch(rows[list(SWIFT)])
    for i, record in enumerate(rows[list(SWIFT)].to_dict(orient="records")):
        single = predictor.predict_one(record)
        assert batch.loc[i, "predicted_inr"] == pytest.approx(single.price_inr)
        assert batch.loc[i, "lower_inr"] == pytest.approx(single.lower_inr)


def test_batch_validation_modes(predictor: PricePredictor) -> None:
    """validate=True names the bad row; validate=False flags it and keeps every row."""
    rows = pd.DataFrame([SWIFT, {**SWIFT, "brand": "Tesla"}])
    with pytest.raises(InputError, match="Row 2"):
        predictor.predict_batch(rows)
    flagged = predictor.predict_batch(rows, validate=False)
    assert flagged["passes_validation"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Schema and CLI
# ---------------------------------------------------------------------------


def test_schema_is_built_from_training_data_only(predictor: PricePredictor) -> None:
    """The saved schema matches a fresh build from train.csv."""
    train = pd.read_csv(get_config().paths.processed_data_dir / "train.csv")
    assert predictor.schema == build_input_schema(train)
    assert predictor.schema["n_training_rows"] == len(train)


def cli_args(spec: dict) -> list[str]:
    """Turn a spec into `one` sub-command arguments."""
    pairs = [[f"--{k.replace('_', '-')}", str(v)] for k, v in spec.items()]
    return ["one", *[item for pair in pairs for item in pair]]


def test_cli_success(capsys: pytest.CaptureFixture[str]) -> None:
    """`one` prints the estimate and exits 0."""
    assert main(cli_args(SWIFT)) == 0
    assert "Estimated price: ₹ 4" in capsys.readouterr().out


def test_cli_invalid_input_is_friendly(capsys: pytest.CaptureFixture[str]) -> None:
    """Bad input exits 2 with plain sentences and no traceback."""
    assert main(cli_args({**SWIFT, "brand": "Tesla"})) == 2
    err = capsys.readouterr().err
    assert "Could not predict" in err and "Traceback" not in err


def test_cli_missing_csv_is_friendly(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing batch file is reported, not raised."""
    assert (
        main(["batch", "--csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o.csv")]) == 2
    )
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Phase 6 output files (acceptance checks)
# ---------------------------------------------------------------------------


def predictions_file(name: str) -> pd.DataFrame:
    """Read reports/predictions/<name>.csv."""
    return pd.read_csv(get_config().paths.reports_dir / "predictions" / f"{name}.csv")


def test_test_predictions_cover_every_test_row() -> None:
    """Row count equals the test set and the required columns exist."""
    scored = predictions_file("test_predictions")
    test = pd.read_csv(get_config().paths.processed_data_dir / "test.csv")
    assert len(scored) == len(test)
    required = {
        "actual_inr",
        "predicted_inr",
        "lower_inr",
        "upper_inr",
        "abs_error_inr",
        "pct_error",
        "inside_interval",
        "brand",
        "model",
        "vehicle_age",
        "km_driven",
    }
    assert required <= set(scored.columns)


def test_worked_examples_include_good_and_poor() -> None:
    """Ten examples, at least one accurate and one miss, each with an explanation."""
    examples = predictions_file("worked_examples")
    settings = get_config().prediction
    assert len(examples) == 10
    assert (examples["abs_pct_error"] <= settings.worked_example_good_ape_pct).any()
    assert (examples["abs_pct_error"] > settings.worked_example_poor_ape_pct).any()
    assert examples["explanation"].str.startswith(("Accurate", "Close", "Miss")).all()


def test_whatif_grid_shape() -> None:
    """One row per (age, km) pair in the configured sweep."""
    grid = predictions_file("whatif_age_km")
    settings = get_config().prediction
    assert len(grid) == len(settings.whatif_ages) * len(settings.whatif_km)


def test_timing_meets_targets() -> None:
    """Measured load and single-prediction times meet the acceptance targets (D-031)."""
    timing = pd.read_csv(get_config().paths.tables_dir / "predictor_timing.csv").iloc[0]
    assert timing["load_passes"] and timing["single_passes"]


def test_car_spec_round_trip() -> None:
    """from_mapping and asdict are inverse for a clean spec."""
    assert asdict(CarSpec.from_mapping(SWIFT)) == SWIFT
