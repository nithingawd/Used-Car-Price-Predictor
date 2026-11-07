"""Unit and integration tests for data loading, validation, and cleansing rules."""

import pandas as pd
import pytest

from carprice.config import get_config
from carprice.data import (
    drop_exact_duplicates,
    drop_identifier_columns,
    fix_fuel_labels,
    fix_invalid_seats,
    normalise_text,
    run_cleaning,
    treat_km_outliers,
    validate_ranges,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provide a small synthetic DataFrame simulating raw data."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4],
            "car_name": [
                "Maruti Swift",
                "Maruti Swift",
                "Honda City",
                "Toyota Camry",
                "Hyundai i20",
            ],
            "brand": ["Maruti", "Maruti ", "Honda", "Toyota", "hyundai"],
            "model": ["Swift", "Swift", "City", "Camry", "i20"],
            "vehicle_age": [5, 5, 3, 4, 2],
            "km_driven": [45000, 45000, 30000, 600000, 15000],
            "seller_type": ["Individual", "Individual", "Dealer", "Dealer", "Individual"],
            "fuel_type": ["Petrol", "Petrol", "Petrol", "Electric", "Diesel"],
            "transmission_type": ["Manual", "Manual", "Automatic", "Automatic", "Manual"],
            "mileage": [20.5, 20.5, 17.8, 19.1, 22.0],
            "engine": [1197, 1197, 1498, 2494, 1197],
            "max_power": [82.0, 82.0, 118.0, 157.7, 83.0],
            "seats": [5, 5, 0, 5, 5],
            "selling_price": [450000, 450000, 750000, 1800000, 600000],
        }
    )


def test_drop_identifier_columns(sample_raw_df: pd.DataFrame) -> None:
    """Verify that Unnamed: 0 and car_name columns are dropped."""
    cleaned, log = drop_identifier_columns(sample_raw_df)
    assert "Unnamed: 0" not in cleaned.columns
    assert "car_name" not in cleaned.columns
    assert log.rule_name == "drop_identifier_columns"
    assert log.rows_after == len(sample_raw_df)


def test_normalise_text(sample_raw_df: pd.DataFrame) -> None:
    """Verify string trimming and title casing across text features."""
    cleaned, log = normalise_text(sample_raw_df)
    assert cleaned.loc[1, "brand"] == "Maruti"
    assert cleaned.loc[4, "brand"] == "Hyundai"
    assert cleaned.loc[4, "model"] == "I20"
    assert log.rule_name == "normalise_text"


def test_drop_exact_duplicates(sample_raw_df: pd.DataFrame) -> None:
    """Verify that exact duplicate rows are identified and dropped."""
    df_no_id, _ = drop_identifier_columns(sample_raw_df)
    df_norm, _ = normalise_text(df_no_id)
    cleaned, log = drop_exact_duplicates(df_norm)
    assert len(cleaned) == 4
    assert log.rows_affected == 1
    assert log.rows_before == 5
    assert log.rows_after == 4


def test_fix_invalid_seats(sample_raw_df: pd.DataFrame) -> None:
    """Verify that seats=0 is imputed with brand/model modal count."""
    cleaned, log = fix_invalid_seats(sample_raw_df)
    assert cleaned.loc[2, "seats"] == 5
    assert (cleaned["seats"] >= 2).all()
    assert (cleaned["seats"] <= 9).all()
    assert log.rows_affected == 1


def test_fix_fuel_labels(sample_raw_df: pd.DataFrame) -> None:
    """Verify that Electric rows are recoded to Petrol."""
    # Add dummy rows so fuel assertion passes (>= 40 rows per category)
    df = sample_raw_df.copy()
    dummy_petrol = pd.DataFrame([df.iloc[0].to_dict() for _ in range(50)])
    dummy_diesel = pd.DataFrame([df.iloc[4].to_dict() for _ in range(50)])
    expanded = pd.concat([df, dummy_petrol, dummy_diesel], ignore_index=True)

    cleaned, log = fix_fuel_labels(expanded)
    assert (cleaned["fuel_type"] != "Electric").all()
    assert log.rows_affected == 1


def test_treat_km_outliers(sample_raw_df: pd.DataFrame) -> None:
    """Verify that km_driven is capped and km_was_capped flag is created."""
    cleaned, log = treat_km_outliers(sample_raw_df, km_cap=500000)
    assert "km_was_capped" in cleaned.columns
    assert cleaned.loc[3, "km_driven"] == 500000
    assert bool(cleaned.loc[3, "km_was_capped"]) is True
    assert bool(cleaned.loc[0, "km_was_capped"]) is False
    assert log.rows_affected == 1


def test_validate_ranges() -> None:
    """Verify that validate_ranges raises on non-physical negative numbers."""
    valid_df = pd.DataFrame(
        {
            "engine": [1197],
            "max_power": [82.0],
            "mileage": [20.5],
            "vehicle_age": [3],
            "selling_price": [450000],
        }
    )
    cleaned, log = validate_ranges(valid_df)
    assert log.rule_name == "validate_ranges"

    invalid_df = valid_df.copy()
    invalid_df.loc[0, "engine"] = -100
    with pytest.raises(ValueError, match="Non-positive engine"):
        validate_ranges(invalid_df)


def test_run_cleaning_end_to_end() -> None:
    """Test full cleaning pipeline on the actual raw cardekho dataset."""
    config = get_config()
    cleaned_df, log_df = run_cleaning()

    # Reconcile row counts: raw 15,411 minus 167 duplicates = 15,244 rows
    assert len(cleaned_df) == 15244
    assert len(cleaned_df.columns) == 13
    assert not cleaned_df.isnull().any().any()
    assert not cleaned_df.duplicated().any()

    # Verify no seats outside 2 to 9
    assert ((cleaned_df["seats"] >= 2) & (cleaned_df["seats"] <= 9)).all()

    # Verify fuel types (no Electric remains)
    assert "Electric" not in cleaned_df["fuel_type"].values

    # Verify output files exist
    assert (config.paths.interim_data_dir / "cars_clean.csv").exists()
    assert (config.paths.tables_dir / "cleaning_log.csv").exists()

    # Verify cleaning log structure
    assert len(log_df) == 7
    total_dropped = log_df["rows_before"].iloc[0] - log_df["rows_after"].iloc[-1]
    assert total_dropped == 167
