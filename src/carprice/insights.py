"""Explanations and lookups shared by the reports (Phase 6) and the GUI (Phase 7).

- Per-car "why this price": exact TreeSHAP contributions from XGBoost's native
  ``pred_contribs`` (identical to the shap library's TreeExplainer, verified in tests),
  which avoids importing shap in the app (D-032).
- What-if curves: re-predict one car while sweeping one input.
- Model spec lookup: most common engine, power, mileage, seats, fuel and transmission per
  brand/model, built from train.csv only, used to pre-fill the GUI form.
- Similar listings: price range of comparable training cars, for a one-line message.
"""

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from carprice.config import get_config
from carprice.features import CATEGORICAL_FEATURES
from carprice.predict import FIELD_LABELS, CarSpec, PricePredictor

FACTOR_LABELS: dict[str, str] = {
    **FIELD_LABELS,
    "km_per_year": "Kilometres per year",
    "power_per_litre": "Power per litre",
    "is_luxury_brand": "Luxury brand",
    "age_x_power": "Age and power together",
}
SPEC_COLUMNS: list[str] = [
    "engine",
    "max_power",
    "mileage",
    "seats",
    "fuel_type",
    "transmission_type",
]

# ---------------------------------------------------------------------------
# Contributions (TreeSHAP)
# ---------------------------------------------------------------------------


def explanation_frame(final_model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Rows as the estimator sees them: engineered, encoded, with output column names."""
    features = final_model.regressor_.named_steps["features"]
    names = features.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame(features.transform(X), columns=names, index=X.index)


def original_feature_of(column: str) -> str:
    """Map a one-hot output column such as 'brand_Bmw' back to its input feature."""
    for categorical in CATEGORICAL_FEATURES:
        if column.startswith(f"{categorical}_"):
            return categorical
    return column


def group_shap_by_feature(shap_values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum contributions of one-hot columns so each input feature has one value per row."""
    frame = pd.DataFrame(shap_values, columns=columns)
    groups = [original_feature_of(c) for c in columns]
    return frame.T.groupby(groups, sort=False).sum().T


def contributions(final_model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-row log-price contributions grouped by input feature, plus the base value.

    Returns:
        tuple: (rows x features DataFrame indexed like X, base value per row). For each
        row, base + sum(contributions) equals log1p(predicted price).
    """
    transformed = explanation_frame(final_model, X)
    booster = final_model.regressor_.named_steps["model"].get_booster()
    raw = booster.predict(xgb.DMatrix(transformed.to_numpy()), pred_contribs=True)
    grouped = group_shap_by_feature(raw[:, :-1], list(transformed.columns))
    grouped.index = X.index
    return grouped, raw[:, -1]


def top_factors(row_contributions: pd.Series, n_factors: int) -> pd.DataFrame:
    """The n largest effects for one car, as % change in predicted price.

    Returns:
        pd.DataFrame: feature, label, log_contribution, effect_pct; largest |effect| first.
    """
    order = row_contributions.abs().sort_values(ascending=False).index[:n_factors]
    chosen = row_contributions[order]
    return pd.DataFrame(
        {
            "feature": order,
            "label": [FACTOR_LABELS.get(name, name) for name in order],
            "log_contribution": chosen.to_numpy(),
            "effect_pct": np.expm1(chosen.to_numpy()) * 100,
        }
    ).reset_index(drop=True)


def explain_car(predictor: PricePredictor, spec: CarSpec, n_factors: int) -> pd.DataFrame:
    """Top factors for one validated car (see top_factors)."""
    grouped, _ = contributions(predictor.pipeline, spec.to_frame())
    return top_factors(grouped.iloc[0], n_factors)


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------


def what_if(
    predictor: PricePredictor, spec: CarSpec, field: str, values: list[float]
) -> pd.DataFrame:
    """Predicted price and range as one input is swept and the others are held fixed.

    Raises:
        InputError: If any swept value is outside the validated input range.
    """
    cars = [replace(spec, **{field: value}) for value in values]
    for car in cars:
        predictor.check(car)
    scored = predictor.predict_batch(pd.DataFrame([asdict(car) for car in cars]))
    return scored[[field, "predicted_inr", "lower_inr", "upper_inr"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lookups from training data
# ---------------------------------------------------------------------------


def most_common(values: pd.Series) -> Any:
    """Mode with a deterministic tie-break (smallest value among the most frequent)."""
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def build_model_specs(train_df: pd.DataFrame) -> pd.DataFrame:
    """Most common spec per brand and model in the training data (for GUI pre-fill)."""
    grouped = train_df.groupby(["brand", "model"], sort=True)
    specs = grouped[SPEC_COLUMNS].agg(most_common)
    specs["n_training_rows"] = grouped.size()
    specs["median_vehicle_age"] = grouped["vehicle_age"].median()
    specs["median_km_driven"] = grouped["km_driven"].median()
    return specs.reset_index()


def default_spec(model_specs: pd.DataFrame, brand: str, model: str) -> dict[str, Any]:
    """Pre-fill values for one brand/model.

    Raises:
        KeyError: If the pair is not in the lookup.
    """
    match = model_specs[(model_specs["brand"] == brand) & (model_specs["model"] == model)]
    if match.empty:
        raise KeyError(f"No default specs for {brand} {model}")
    return match.iloc[0].to_dict()


def similar_listing_range(
    market: pd.DataFrame, values: Mapping[str, Any], age_window: int
) -> dict[str, Any] | None:
    """Inter-quartile price of same brand/model cars within ±age_window years.

    Falls back to the same brand and age window when the model has too few matches.
    Returns None when even the fallback is too thin to be meaningful.
    """
    min_rows = get_config().prediction.similar_min_listings
    near_age = (market["vehicle_age"] - values["vehicle_age"]).abs() <= age_window
    same_brand = market["brand"] == values["brand"]
    for scope, mask in (
        ("model", near_age & same_brand & (market["model"] == values["model"])),
        ("brand", near_age & same_brand),
    ):
        prices = market.loc[mask, "selling_price"]
        if len(prices) >= min_rows:
            return {
                "scope": scope,
                "n_listings": len(prices),
                "q25": float(prices.quantile(0.25)),
                "median": float(prices.median()),
                "q75": float(prices.quantile(0.75)),
            }
    return None
