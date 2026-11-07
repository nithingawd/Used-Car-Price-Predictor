"""Data loading, schema validation, and cleansing pipelines."""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from carprice.config import AppConfig, get_config

logger = logging.getLogger(__name__)

RAW_SCHEMA: dict[str, Any] = {
    "Unnamed: 0": "int64",
    "car_name": "string",
    "brand": "string",
    "model": "string",
    "vehicle_age": "int64",
    "km_driven": "int64",
    "seller_type": "string",
    "fuel_type": "string",
    "transmission_type": "string",
    "mileage": "float64",
    "engine": "int64",
    "max_power": "float64",
    "seats": "int64",
    "selling_price": "int64",
}

CLEANED_REQUIRED_COLUMNS: list[str] = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "selling_price",
    "km_was_capped",
]


@dataclass(frozen=True)
class CleaningLogEntry:
    """Record of a single data cleansing rule execution."""

    rule_name: str
    rows_before: int
    rows_after: int
    rows_affected: int
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary for reporting."""
        return asdict(self)


def load_raw_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load raw dataset with explicit schema and check integrity.

    Args:
        data_path: Optional path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded immutable DataFrame.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If file schema deviates from expectation.
    """
    config: AppConfig = get_config()
    path = data_path or config.paths.raw_data_path

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    # Explicit dtypes cast on load
    dtype_map = {col: dtype for col, dtype in RAW_SCHEMA.items() if dtype != "string"}
    df = pd.read_csv(path, dtype=dtype_map)

    # Cast string columns explicitly
    for col, dtype in RAW_SCHEMA.items():
        if dtype == "string":
            df[col] = df[col].astype("string")

    logger.info("Loaded raw data from %s with shape %s", path, df.shape)
    validate_schema(df, is_cleaned=False)
    return df


def validate_schema(df: pd.DataFrame, is_cleaned: bool = False) -> None:
    """Validate DataFrame schema, column existence, and essential types.

    Args:
        df: DataFrame to validate.
        is_cleaned: Flag indicating whether cleaned or raw schema is expected.

    Raises:
        ValueError: If required columns or data constraints fail.
    """
    expected_cols = CLEANED_REQUIRED_COLUMNS if is_cleaned else list(RAW_SCHEMA.keys())
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    if df.empty:
        raise ValueError("Schema violation: DataFrame is empty")

    if is_cleaned:
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"Cleaned data contains unexpected nulls in columns: {null_cols}")
        if df.duplicated().any():
            raise ValueError("Cleaned data contains unexpected duplicate rows")


def drop_identifier_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 1: Drop redundant identifier columns Unnamed: 0 and car_name.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Transformed DataFrame and log entry.
    """
    rows_before = len(df)
    drop_cols = [col for col in ["Unnamed: 0", "car_name"] if col in df.columns]
    cleaned = df.drop(columns=drop_cols).copy()
    entry = CleaningLogEntry(
        rule_name="drop_identifier_columns",
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_affected=rows_before,
        details=f"Dropped identifier columns: {drop_cols}",
    )
    return cleaned, entry


def normalise_text(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 2: Strip whitespace and standardize casing on categorical string features.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Transformed DataFrame and log entry.
    """
    rows_before = len(df)
    cleaned = df.copy()
    text_cols = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]

    affected = 0
    for col in text_cols:
        if col in cleaned.columns:
            original = cleaned[col].astype(str)
            stripped = original.str.strip().str.title()
            if not original.equals(stripped):
                diff_mask = original != stripped
                affected += int(diff_mask.sum())
            cleaned[col] = stripped.astype("string")

    entry = CleaningLogEntry(
        rule_name="normalise_text",
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_affected=affected,
        details="Trimmed whitespace and applied title casing to categorical columns",
    )
    return cleaned, entry


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 3: Remove duplicate rows after identifier stripping.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Deduplicated DataFrame and log entry.
    """
    rows_before = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    cleaned = df.drop_duplicates(keep="first").copy()
    rows_after = len(cleaned)

    # Project rule: warn if a step drops more than the configured fraction of rows
    max_drop = get_config().cleaning.max_drop_row_fraction
    drop_pct = dup_count / rows_before
    if drop_pct > max_drop:
        logger.warning(
            "Duplicate removal dropped %.2f%% of rows (limit %.2f%%)",
            drop_pct * 100,
            max_drop * 100,
        )

    entry = CleaningLogEntry(
        rule_name="drop_exact_duplicates",
        rows_before=rows_before,
        rows_after=rows_after,
        rows_affected=dup_count,
        details=f"Removed {dup_count} exact duplicate feature rows (kept first occurrence)",
    )
    return cleaned, entry


def fix_invalid_seats(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 4: Impute invalid seats (seats = 0) with mode for brand and model.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Transformed DataFrame and log entry.

    Raises:
        AssertionError: If any seats remain outside the configured physical range.
    """
    cleaning = get_config().cleaning
    rows_before = len(df)
    cleaned = df.copy()
    invalid_mask = cleaned["seats"] == 0
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        # Group by brand and model to find modal seat count
        valid_seats = cleaned[cleaned["seats"] > 0]
        model_modes = (
            valid_seats.groupby(["brand", "model"])["seats"]
            .agg(lambda s: s.mode().iloc[0] if not s.empty else cleaning.default_seats)
            .to_dict()
        )

        for idx in cleaned[invalid_mask].index:
            key = (cleaned.at[idx, "brand"], cleaned.at[idx, "model"])
            imputed = model_modes.get(key, cleaning.default_seats)
            cleaned.at[idx, "seats"] = imputed

    # Assert all seat values are within the configured physical range
    in_range = cleaned["seats"].between(cleaning.seats_min, cleaning.seats_max)
    assert in_range.all(), "Invalid seat count detected"

    entry = CleaningLogEntry(
        rule_name="fix_invalid_seats",
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_affected=invalid_count,
        details=f"Imputed {invalid_count} rows with seats=0 to brand/model modal count",
    )
    return cleaned, entry


def fix_fuel_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 5: Recode mislabeled Electric Camry hybrid vehicles to Petrol.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Transformed DataFrame and log entry.

    Raises:
        AssertionError: If any fuel category has fewer rows than the configured threshold.
    """
    rows_before = len(df)
    cleaned = df.copy()
    electric_mask = cleaned["fuel_type"] == "Electric"
    electric_count = int(electric_mask.sum())

    if electric_count > 0:
        cleaned.loc[electric_mask, "fuel_type"] = "Petrol"

    # Verify every remaining fuel category is large enough for stratified CV folds
    min_rows = get_config().cleaning.rare_fuel_threshold
    counts = cleaned["fuel_type"].value_counts()
    for fuel, count in counts.items():
        if count < min_rows:
            raise AssertionError(f"Unexpected rare fuel type: {fuel} has only {count} rows")

    entry = CleaningLogEntry(
        rule_name="fix_fuel_labels",
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_affected=electric_count,
        details=f"Recoded {electric_count} Electric hybrid Camry rows to Petrol",
    )
    return cleaned, entry


def treat_km_outliers(
    df: pd.DataFrame, km_cap: int | None = None
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 6: Cap extreme km_driven values at threshold and add km_was_capped flag.

    Args:
        df: Input DataFrame.
        km_cap: Optional kilometer threshold (defaults to config value).

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Transformed DataFrame and log entry.
    """
    rows_before = len(df)
    cap = km_cap if km_cap is not None else get_config().cleaning.km_cap
    cleaned = df.copy()

    capped_mask = cleaned["km_driven"] > cap
    affected = int(capped_mask.sum())

    cleaned["km_was_capped"] = capped_mask
    cleaned.loc[capped_mask, "km_driven"] = cap

    entry = CleaningLogEntry(
        rule_name="treat_km_outliers",
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_affected=affected,
        details=f"Capped {affected} rows with km_driven > {cap:,} and added km_was_capped flag",
    )
    return cleaned, entry


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Rule 7: Validate physical plausibility of numerical measurements.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, CleaningLogEntry]: Verified DataFrame and log entry.

    Raises:
        ValueError: If non-physical negative or zero values are detected.
    """
    rows_before = len(df)
    violations = []

    if (df["engine"] <= 0).any():
        violations.append(f"Non-positive engine: {df[df['engine'] <= 0].index.tolist()}")
    if (df["max_power"] <= 0).any():
        violations.append(f"Non-positive max_power: {df[df['max_power'] <= 0].index.tolist()}")
    if (df["mileage"] <= 0).any():
        violations.append(f"Non-positive mileage: {df[df['mileage'] <= 0].index.tolist()}")
    if (df["vehicle_age"] < 0).any():
        violations.append(f"Negative vehicle_age: {df[df['vehicle_age'] < 0].index.tolist()}")
    if (df["selling_price"] <= 0).any():
        violations.append(
            f"Non-positive selling_price: {df[df['selling_price'] <= 0].index.tolist()}"
        )

    if violations:
        raise ValueError(f"Range validation failed: {'; '.join(violations)}")

    entry = CleaningLogEntry(
        rule_name="validate_ranges",
        rows_before=rows_before,
        rows_after=rows_before,
        rows_affected=0,
        details="Verified physical plausibility of engine, power, mileage, age, and price",
    )
    return df.copy(), entry


def run_cleaning(
    raw_path: Path | None = None,
    interim_path: Path | None = None,
    log_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all cleaning rules sequentially and export interim dataset and log.

    Args:
        raw_path: Optional path to raw dataset.
        interim_path: Optional destination for cleaned data CSV.
        log_path: Optional destination for cleaning log CSV.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned DataFrame and Cleaning Log DataFrame.
    """
    config: AppConfig = get_config()
    raw = raw_path or config.paths.raw_data_path
    interim = interim_path or (config.paths.interim_data_dir / "cars_clean.csv")
    log_file = log_path or (config.paths.tables_dir / "cleaning_log.csv")

    df = load_raw_data(raw)
    logs: list[CleaningLogEntry] = []

    steps = [
        drop_identifier_columns,
        normalise_text,
        drop_exact_duplicates,
        fix_invalid_seats,
        fix_fuel_labels,
        treat_km_outliers,
        validate_ranges,
    ]

    current = df
    for step in steps:
        current, entry = step(current)
        logs.append(entry)
        logger.info(
            "Rule %s executed: %d -> %d rows (affected: %d)",
            entry.rule_name,
            entry.rows_before,
            entry.rows_after,
            entry.rows_affected,
        )

    validate_schema(current, is_cleaned=True)

    # Save cleaned interim data
    interim.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(interim, index=False)
    logger.info("Saved cleaned interim data to %s (shape: %s)", interim, current.shape)

    # Save cleaning log table
    log_df = pd.DataFrame([entry.to_dict() for entry in logs])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_file, index=False)
    logger.info("Saved cleaning audit log to %s", log_file)

    return current, log_df


def audit_schema_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Generate schema audit table detailing dtypes, nulls, and cardinality.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table of columns, types, null counts, and unique values.
    """
    records = []
    total_rows = len(df)
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        records.append(
            {
                "column_name": col,
                "data_type": str(df[col].dtype),
                "non_null_count": total_rows - null_count,
                "null_count": null_count,
                "null_percentage": round((null_count / total_rows) * 100, 3),
                "unique_values": int(df[col].nunique()),
            }
        )
    return pd.DataFrame(records)


def audit_numeric_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended descriptive statistics including custom percentiles.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics across numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    desc = df[numeric_cols].describe(percentiles=percentiles).T
    desc = desc.reset_index().rename(columns={"index": "feature"})
    return desc


def audit_duplicates_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Audit exact duplicates and identical feature rows with price variations.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table comparing duplicate definitions.
    """
    total = len(df)
    with_id = int(df.duplicated().sum())

    drop_cols = [c for c in ["Unnamed: 0", "car_name"] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)
    exact_no_id = int(feature_df.duplicated().sum())

    feature_cols = [c for c in feature_df.columns if c != "selling_price"]
    same_feat_diff_price = int(
        (
            feature_df.duplicated(subset=feature_cols, keep=False)
            & ~feature_df.duplicated(keep=False)
        ).sum()
    )

    records = [
        {
            "scope": "Raw with Unnamed: 0",
            "duplicate_rows": with_id,
            "percentage": round((with_id / total) * 100, 3),
            "relevance": "Index column prevents exact row matching",
        },
        {
            "scope": "Excluding Identifiers (Exact Duplicates)",
            "duplicate_rows": exact_no_id,
            "percentage": round((exact_no_id / total) * 100, 3),
            "relevance": "Must be removed to prevent train/test leakage",
        },
        {
            "scope": "Identical Specs, Different Price",
            "duplicate_rows": same_feat_diff_price,
            "percentage": round((same_feat_diff_price / total) * 100, 3),
            "relevance": "Genuine market pricing variance; keep in dataset",
        },
    ]
    return pd.DataFrame(records)


def audit_categorical_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Audit categorical features and flag rare categories (< 1% frequency).

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Table of categorical levels, counts, shares, and rarity flags.
    """
    cat_cols = ["brand", "seller_type", "fuel_type", "transmission_type"]
    records = []
    total = len(df)

    for col in cat_cols:
        if col in df.columns:
            counts = df[col].value_counts()
            for val, count in counts.items():
                pct = (count / total) * 100
                records.append(
                    {
                        "feature": col,
                        "category": str(val),
                        "row_count": int(count),
                        "percentage": round(pct, 3),
                        "rare_flag_under_1pct": bool(pct < 1.0),
                    }
                )
    return pd.DataFrame(records)


def audit_logical_consistency(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Audit domain-specific engineering checks and isolate offending rows.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: Summary table and dict of anomaly subsets.
    """
    config = get_config()
    cleaning, audit = config.cleaning, config.audit
    total = len(df)
    subsets: dict[str, pd.DataFrame] = {}

    # Check 1: Seats outside the physical range
    seats_mask = ~df["seats"].between(cleaning.seats_min, cleaning.seats_max)
    subsets["seats_anomalies"] = df[seats_mask].copy()

    # Check 2: Vehicle age of 0
    age_zero_mask = df["vehicle_age"] == 0
    subsets["age_zero"] = df[age_zero_mask].copy()

    # Check 3: Very high annual usage
    safe_age = df["vehicle_age"].replace(0, 1)
    km_rate_mask = (df["km_driven"] / safe_age) > audit.max_km_per_year
    subsets["km_per_year_high"] = df[km_rate_mask].copy()

    # Check 4: Fuel efficiency outside the plausible band
    mileage_mask = (df["mileage"] < audit.mileage_min_kmpl) | (
        df["mileage"] > audit.mileage_max_kmpl
    )
    subsets["mileage_outliers"] = df[mileage_mask].copy()

    # Check 5: Power density outside the plausible band
    power_density = df["max_power"] / (df["engine"] / 1000.0)
    power_density_mask = (power_density < audit.power_per_litre_min) | (
        power_density > audit.power_per_litre_max
    )
    subsets["power_density_outliers"] = df[power_density_mask].copy()

    # Check 6: Very low selling price
    price_low_mask = df["selling_price"] < audit.low_price_inr
    subsets["price_below_50k"] = df[price_low_mask].copy()

    # Check 7: Electric label rows
    electric_mask = df["fuel_type"] == "Electric"
    subsets["electric_rows"] = df[electric_mask].copy()

    # Check 8: Odometer above the cleaning cap
    km_outlier_mask = df["km_driven"] > cleaning.km_cap
    subsets["km_outliers"] = df[km_outlier_mask].copy()

    checks = [
        (
            f"Seats outside {cleaning.seats_min} to {cleaning.seats_max}",
            int(seats_mask.sum()),
            "Data entry error (seats=0) requiring imputation",
        ),
        (
            "Vehicle age == 0 years",
            int(age_zero_mask.sum()),
            "Brand new showroom cars (< 1 year old)",
        ),
        (
            f"Kilometres per year > {audit.max_km_per_year:,}",
            int(km_rate_mask.sum()),
            "Extreme commercial usage or odometer anomalies",
        ),
        (
            f"Mileage outside {audit.mileage_min_kmpl:g} to {audit.mileage_max_kmpl:g} kmpl",
            int(mileage_mask.sum()),
            "Supercars (low mileage) or hyper-milers",
        ),
        (
            (
                f"Power per litre outside {audit.power_per_litre_min:g}-"
                f"{audit.power_per_litre_max:g} bhp/L"
            ),
            int(power_density_mask.sum()),
            "Naturally aspirated entry vs high-boost turbo",
        ),
        (
            f"Selling price < {audit.low_price_inr:,} INR",
            int(price_low_mask.sum()),
            "Heavily depreciated, scrap, or project cars",
        ),
        (
            "Fuel type == 'Electric'",
            int(electric_mask.sum()),
            "Toyota Camry hybrids mislabelled as Electric",
        ),
        (
            f"Odometer km_driven > {cleaning.km_cap:,}",
            int(km_outlier_mask.sum()),
            "Extreme high-leverage outliers requiring capping",
        ),
    ]

    summary_df = pd.DataFrame(
        [
            {
                "check_description": name,
                "offending_rows": count,
                "percentage": round((count / total) * 100, 3),
                "engineering_verdict": verdict,
            }
            for name, count, verdict in checks
        ]
    )
    return summary_df, subsets


def run_data_audit(
    raw_path: Path | None = None,
    tables_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute complete data audit suite and write all reports/tables/audit_*.csv files.

    Args:
        raw_path: Optional path to raw dataset.
        tables_dir: Optional directory for exported audit tables.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of generated audit DataFrames.
    """
    config: AppConfig = get_config()
    raw = raw_path or config.paths.raw_data_path
    out_dir = tables_dir or config.paths.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw)
    audit_results: dict[str, pd.DataFrame] = {}

    # 1. Schema & Missing
    schema_df = audit_schema_and_missing(df)
    schema_df.to_csv(out_dir / "audit_schema.csv", index=False)
    audit_results["schema"] = schema_df

    # 2. Numeric distribution
    numeric_df = audit_numeric_distribution(df)
    numeric_df.to_csv(out_dir / "audit_numeric_summary.csv", index=False)
    audit_results["numeric"] = numeric_df

    # 3. Duplicates
    dup_df = audit_duplicates_summary(df)
    dup_df.to_csv(out_dir / "audit_duplicates.csv", index=False)
    audit_results["duplicates"] = dup_df

    # 4. Categorical frequencies
    cat_df = audit_categorical_distributions(df)
    cat_df.to_csv(out_dir / "audit_categorical_frequencies.csv", index=False)
    audit_results["categorical"] = cat_df

    # 5. Name consistency check
    name_check = pd.DataFrame(
        [
            {
                "total_rows": len(df),
                "name_equals_brand_plus_model": int(
                    (df["car_name"] == (df["brand"] + " " + df["model"])).sum()
                ),
                "mismatch_count": int((df["car_name"] != (df["brand"] + " " + df["model"])).sum()),
            }
        ]
    )
    name_check.to_csv(out_dir / "audit_name_consistency.csv", index=False)
    audit_results["name_consistency"] = name_check

    # 6. Logical checks and anomaly tables
    logical_df, anomaly_subsets = audit_logical_consistency(df)
    logical_df.to_csv(out_dir / "audit_logical_checks.csv", index=False)
    audit_results["logical_checks"] = logical_df

    anomaly_subsets["electric_rows"].to_csv(out_dir / "audit_electric_rows.csv", index=False)
    audit_results["electric_rows"] = anomaly_subsets["electric_rows"]

    anomaly_subsets["seats_anomalies"].to_csv(out_dir / "audit_seats_anomalies.csv", index=False)
    audit_results["seats_anomalies"] = anomaly_subsets["seats_anomalies"]

    anomaly_subsets["km_outliers"].to_csv(out_dir / "audit_km_outliers.csv", index=False)
    audit_results["km_outliers"] = anomaly_subsets["km_outliers"]

    logger.info("Successfully exported 9 audit tables to %s", out_dir)
    return audit_results
