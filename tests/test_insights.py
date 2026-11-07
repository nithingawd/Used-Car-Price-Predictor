"""Tests for insights.py: TreeSHAP contributions, factors, what-if and lookups."""

import numpy as np
import pandas as pd
import pytest

from carprice.config import get_config
from carprice.insights import (
    build_model_specs,
    contributions,
    default_spec,
    explain_car,
    most_common,
    original_feature_of,
    similar_listing_range,
    top_factors,
    what_if,
)
from carprice.predict import CarSpec, InputError, PricePredictor
from tests.test_predict import SWIFT


@pytest.fixture(scope="module")
def predictor() -> PricePredictor:
    """Load the predictor once for the module."""
    return PricePredictor()


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    """Training partition."""
    return pd.read_csv(get_config().paths.processed_data_dir / "train.csv")


def test_contributions_add_up_to_prediction(predictor: PricePredictor, train: pd.DataFrame) -> None:
    """base + sum of contributions = log1p(prediction) for every row (TreeSHAP additivity)."""
    rows = train.head(50).drop(columns=["selling_price", "log_selling_price"])
    grouped, base = contributions(predictor.pipeline, rows)
    total = base + grouped.sum(axis=1).to_numpy()
    np.testing.assert_allclose(total, np.log1p(predictor.pipeline.predict(rows)), atol=1e-4)


def test_one_hot_columns_group_to_inputs() -> None:
    """One-hot output names map back to the input feature."""
    assert original_feature_of("brand_Bmw") == "brand"
    assert original_feature_of("fuel_type_Cng") == "fuel_type"
    assert original_feature_of("max_power") == "max_power"


def test_top_factors_sorted_by_size() -> None:
    """Largest absolute effects come first and carry readable labels."""
    row = pd.Series({"vehicle_age": -0.4, "max_power": 0.1, "model": 0.6})
    factors = top_factors(row, 2)
    assert factors["feature"].tolist() == ["model", "vehicle_age"]
    assert factors.loc[1, "label"] == "Vehicle age (years)"
    assert factors.loc[0, "effect_pct"] == pytest.approx(np.expm1(0.6) * 100)


def test_explain_car_returns_requested_count(predictor: PricePredictor) -> None:
    """explain_car gives exactly n factors for a valid car."""
    assert len(explain_car(predictor, CarSpec.from_mapping(SWIFT), 5)) == 5


def test_what_if_sweeps_one_field(predictor: PricePredictor) -> None:
    """One row per swept value; out-of-range values are refused."""
    car = CarSpec.from_mapping(SWIFT)
    curve = what_if(predictor, car, "vehicle_age", [1, 5, 10])
    assert curve["vehicle_age"].tolist() == [1, 5, 10]
    assert curve.loc[1, "predicted_inr"] == pytest.approx(predictor.predict_one(car).price_inr)
    with pytest.raises(InputError):
        what_if(predictor, car, "vehicle_age", [99])


def test_most_common_tie_break() -> None:
    """Ties resolve to the smallest value, deterministically."""
    assert most_common(pd.Series([5, 7, 7, 5, 9])) == 5


def test_model_specs_lookup(train: pd.DataFrame) -> None:
    """Every training brand/model has a spec row; the lookup returns its values."""
    specs = build_model_specs(train)
    assert len(specs) == train.groupby(["brand", "model"]).ngroups
    swift = default_spec(specs, "Maruti", "Swift")
    assert swift["engine"] == most_common(train.loc[train["model"] == "Swift", "engine"])
    with pytest.raises(KeyError):
        default_spec(specs, "Maruti", "Nope")


def test_saved_model_specs_match_fresh_build(train: pd.DataFrame) -> None:
    """models/model_specs.csv is reproducible from train.csv."""
    saved = pd.read_csv(get_config().paths.models_dir / "model_specs.csv")
    fresh = build_model_specs(train)
    pd.testing.assert_frame_equal(saved, fresh, check_dtype=False)


def test_similar_listing_range(train: pd.DataFrame) -> None:
    """A common model gets a model-level range; an unknown model falls back or returns None."""
    found = similar_listing_range(train, SWIFT, age_window=1)
    assert found is not None and found["scope"] == "model"
    assert found["q25"] <= found["median"] <= found["q75"]
    fallback = similar_listing_range(train, {**SWIFT, "model": "Nope"}, age_window=1)
    assert fallback is not None and fallback["scope"] == "brand"
