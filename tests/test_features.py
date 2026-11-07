"""Unit and leakage tests for dataset splitting, feature engineering, and ColumnTransformers."""

import json

import numpy as np
import pandas as pd
import pytest

from carprice.config import get_config
from carprice.features import (
    DomainFeatureEngineer,
    RareCategoryEncoder,
    build_feature_pipeline,
)


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Provide a synthetic training partition for pipeline unit testing."""
    return pd.DataFrame(
        {
            "brand": ["Maruti", "Maruti", "Hyundai", "Honda", "Toyota", "Bmw", "Audi", "Ferrari"],
            "model": ["Swift", "Alto", "i20", "City", "Innova", "3 Series", "A4", "488"],
            "vehicle_age": [4, 8, 3, 5, 6, 2, 4, 1],
            "km_driven": [45000, 80000, 25000, 50000, 75000, 15000, 35000, 5000],
            "seller_type": [
                "Individual",
                "Individual",
                "Dealer",
                "Dealer",
                "Individual",
                "Dealer",
                "Dealer",
                "Individual",
            ],
            "fuel_type": [
                "Petrol",
                "Petrol",
                "Petrol",
                "Petrol",
                "Diesel",
                "Petrol",
                "Diesel",
                "Petrol",
            ],
            "transmission_type": [
                "Manual",
                "Manual",
                "Manual",
                "Manual",
                "Manual",
                "Automatic",
                "Automatic",
                "Automatic",
            ],
            "mileage": [20.5, 22.0, 19.8, 17.8, 12.8, 15.2, 16.5, 8.5],
            "engine": [1197, 796, 1197, 1498, 2393, 1998, 1968, 3902],
            "max_power": [82.0, 47.3, 81.8, 117.3, 147.5, 187.7, 187.7, 660.0],
            "seats": [5, 5, 5, 5, 7, 5, 5, 2],
            "selling_price": [
                450000,
                220000,
                550000,
                700000,
                1400000,
                2800000,
                2600000,
                35000000,
            ],
            "km_was_capped": [False, False, False, False, False, False, False, False],
        }
    )


def test_split_shapes_and_disjoint() -> None:
    """Verify that train and test partitions do not overlap and preserve quintile distribution."""
    config = get_config()
    train_path = config.paths.processed_data_dir / "train.csv"
    test_path = config.paths.processed_data_dir / "test.csv"
    indices_path = config.paths.processed_data_dir / "split_indices.json"

    assert train_path.exists(), "train.csv was not created"
    assert test_path.exists(), "test.csv was not created"
    assert indices_path.exists(), "split_indices.json was not created"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 80/20 partition verification
    total_len = len(train_df) + len(test_df)
    assert total_len == 15244
    assert len(train_df) == 12195
    assert len(test_df) == 3049

    with indices_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    train_set = set(meta["train_indices"])
    test_set = set(meta["test_indices"])
    assert len(train_set.intersection(test_set)) == 0, (
        "Data leakage: Train and test indices overlap!"
    )

    # Verify quintile balance within 2%
    clean_df = pd.read_csv(config.paths.interim_data_dir / "cars_clean.csv")
    global_quintiles = pd.qcut(clean_df["selling_price"], q=5, labels=False)
    train_q_shares = global_quintiles.iloc[meta["train_indices"]].value_counts(normalize=True)
    test_q_shares = global_quintiles.iloc[meta["test_indices"]].value_counts(normalize=True)

    for q in range(5):
        diff = abs(train_q_shares.get(q, 0.0) - test_q_shares.get(q, 0.0))
        assert diff <= 0.02, f"Quintile {q} distribution imbalance exceeds 2%: diff={diff:.4f}"


def test_rare_category_encoder() -> None:
    """Verify RareCategoryEncoder maps infrequent levels to 'Other'."""
    df = pd.DataFrame({"brand": ["Maruti", "Maruti", "Maruti", "RareBrand"]})
    encoder = RareCategoryEncoder(columns=["brand"], min_frequency=2)
    encoder.fit(df)
    transformed = encoder.transform(df)

    assert "Other" in transformed["brand"].values
    assert transformed.loc[3, "brand"] == "Other"
    assert transformed.loc[0, "brand"] == "Maruti"


def test_domain_feature_engineer(sample_train_df: pd.DataFrame) -> None:
    """Verify calculation and column output of DomainFeatureEngineer."""
    engineer = DomainFeatureEngineer(luxury_brands=["Bmw", "Audi", "Ferrari"])
    transformed = engineer.fit_transform(sample_train_df)

    assert "km_per_year" in transformed.columns
    assert "power_per_litre" in transformed.columns
    assert "is_luxury_brand" in transformed.columns
    assert "age_x_power" in transformed.columns

    # Check math on first row (Maruti Swift, age 4, km 45000, engine 1197, power 82.0)
    assert transformed.loc[0, "km_per_year"] == 45000 / 4
    assert round(float(transformed.loc[0, "power_per_litre"]), 2) == round(82.0 / 1.197, 2)
    assert transformed.loc[0, "is_luxury_brand"] == 0.0
    assert transformed.loc[5, "is_luxury_brand"] == 1.0  # BMW
    assert transformed.loc[0, "age_x_power"] == 4 * 82.0


def test_feature_pipeline_both_kinds(sample_train_df: pd.DataFrame) -> None:
    """Verify that both linear and tree pipelines fit, transform, and expose feature names."""
    y = np.log1p(sample_train_df["selling_price"])

    for kind in ["linear", "tree"]:
        pipe = build_feature_pipeline(kind=kind)
        X_trans = pipe.fit_transform(sample_train_df, y)

        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape[0] == len(sample_train_df)
        assert X_trans.shape[1] > 10

        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
        assert len(feature_names) == X_trans.shape[1]
        assert all(isinstance(name, str) for name in feature_names)
        assert "vehicle_age" in feature_names
        assert "power_per_litre" in feature_names


def test_unseen_category_handling(sample_train_df: pd.DataFrame) -> None:
    """Verify pipeline handles completely unseen brand and model categories gracefully."""
    y = np.log1p(sample_train_df["selling_price"])
    pipe = build_feature_pipeline(kind="tree")
    pipe.fit(sample_train_df, y)

    # Completely novel car listing
    unseen_row = pd.DataFrame(
        {
            "brand": ["Tesla"],
            "model": ["CyberTruck"],
            "vehicle_age": [1],
            "km_driven": [5000],
            "seller_type": ["Dealer"],
            "fuel_type": ["Petrol"],  # Valid fuel
            "transmission_type": ["Automatic"],
            "mileage": [18.0],
            "engine": [2000],
            "max_power": [300.0],
            "seats": [5],
            "km_was_capped": [False],
        }
    )

    out = pipe.transform(unseen_row)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 1
    assert not np.isnan(out).any(), "Transformation of unseen category produced NaN!"


def test_no_data_leakage() -> None:
    """Strict test verifying fitting feature pipelines never accesses test.csv."""
    config = get_config()
    train_df = pd.read_csv(config.paths.processed_data_dir / "train.csv")
    y_train = np.log1p(train_df["selling_price"])

    # Fit pipeline exclusively on train
    pipe = build_feature_pipeline(kind="linear")
    pipe.fit(train_df, y_train)

    # Verify that test dataset is never loaded into transformers
    preprocess = pipe.named_steps["preprocess"]
    num_step = preprocess.named_transformers_["num"]
    scaler = num_step.named_steps["scaler"]

    # Scaler mean must match train mean, not full dataset mean
    assert len(scaler.mean_) == 10
    expected_mean = float(train_df["vehicle_age"].mean())
    assert np.isclose(scaler.mean_[0], expected_mean, atol=1e-3)
