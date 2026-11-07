"""Feature engineering transformers and leak-free preprocessing pipelines."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from carprice.config import AppConfig, get_config
from carprice.data import validate_schema

logger = logging.getLogger(__name__)

BASE_NUMERIC_FEATURES: list[str] = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]
# Engineered by DomainFeatureEngineer; the kept set from D-013.
ENGINEERED_FEATURES: list[str] = [
    "km_per_year",
    "power_per_litre",
    "is_luxury_brand",
    "age_x_power",
]
NUMERIC_FEATURES: list[str] = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES: list[str] = ["brand", "fuel_type", "transmission_type", "seller_type"]
HIGH_CARDINALITY_FEATURE = "model"

# (column, description, mode) screened by evaluate_feature_candidates (D-010, D-013).
CANDIDATE_FEATURES: list[tuple[str, str, str]] = [
    ("km_per_year", "Odometer / vehicle_age ratio (wear intensity)", "add"),
    ("power_per_litre", "max_power / (engine/1000) (engine tuning intensity)", "add"),
    ("log_km_driven", "log1p(km_driven) instead of raw capped km", "replace"),
    ("is_luxury_brand", "Binary indicator for luxury/ultra-luxury marques", "add"),
    ("age_x_power", "Interaction between age and rated horsepower", "add"),
    ("power_x_automatic", "max_power x automatic-transmission indicator", "add"),
]


def create_train_test_split(
    interim_path: Path | None = None,
    processed_dir: Path | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """Partition cleaned dataset into a train/test split stratified on target quantile bins.

    Stratification column and bin count come from ``config.splitting`` (D-012); the bins
    are used only to balance the split and are not saved as a feature.

    Args:
        interim_path: Optional path to cars_clean.csv.
        processed_dir: Optional destination directory.
        test_size: Test fraction (defaults to config: 0.20).
        seed: Random seed for reproducibility (defaults to config: 42).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]: Train df, test df, split indices.
    """
    config: AppConfig = get_config()
    source = interim_path or (config.paths.interim_data_dir / "cars_clean.csv")
    dest_dir = processed_dir or config.paths.processed_data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    split_seed = seed if seed is not None else config.project.random_seed
    fraction = test_size if test_size is not None else config.splitting.test_size

    df = pd.read_csv(source)
    validate_schema(df, is_cleaned=True)

    # Compute target and stratification bins
    df["log_selling_price"] = np.log1p(df["selling_price"])
    strata = pd.qcut(
        df[config.splitting.stratify_by], q=config.splitting.stratify_bins, labels=False
    )

    train_indices, test_indices = train_test_split(
        df.index.to_numpy(),
        test_size=fraction,
        random_state=split_seed,
        stratify=strata,
    )

    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)

    # Save indices metadata
    indices_dict = {
        "random_seed": split_seed,
        "test_size": fraction,
        "train_indices": [int(i) for i in train_indices],
        "test_indices": [int(i) for i in test_indices],
    }
    with (dest_dir / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump(indices_dict, f, indent=2)

    # Save partitioned datasets
    train_df.to_csv(dest_dir / "train.csv", index=False)
    test_df.to_csv(dest_dir / "test.csv", index=False)

    logger.info(
        "Created split: train=%d, test=%d (seed=%d)",
        len(train_df),
        len(test_df),
        split_seed,
    )
    return train_df, test_df, indices_dict


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """Group infrequent categorical levels into 'Other' based on training frequencies."""

    def __init__(self, columns: list[str] | None = None, min_frequency: int | None = None) -> None:
        """Initialize transformer.

        Args:
            columns: List of columns to encode.
            min_frequency: Minimum occurrences required to keep individual level
                (defaults to ``features.rare_category_min_frequency``).
        """
        self.columns = columns
        self.min_frequency = min_frequency
        self.frequent_categories_: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RareCategoryEncoder":  # noqa: ARG002
        """Learn frequent levels from training fold only.

        Args:
            X: Input DataFrame.
            y: Ignored target.

        Returns:
            self: Fitted instance.
        """
        cols = self.columns or list(X.columns)
        threshold = self.min_frequency or get_config().features.rare_category_min_frequency
        self.frequent_categories_ = {}
        for col in cols:
            if col in X.columns:
                counts = X[col].astype(str).value_counts()
                frequent = set(counts[counts >= threshold].index)
                self.frequent_categories_[col] = frequent
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace infrequent categories with 'Other'.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        out = X.copy()
        for col, frequent in self.frequent_categories_.items():
            if col in out.columns:
                out[col] = (
                    out[col]
                    .astype(str)
                    .apply(lambda val, freq=frequent: val if val in freq else "Other")
                )
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return feature names.

        Args:
            input_features: Optional input feature names.

        Returns:
            np.ndarray: Output column names.
        """
        return (
            np.array(input_features)
            if input_features is not None
            else np.array(list(self.frequent_categories_.keys()))
        )


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer domain-specific physical ratios and interaction terms."""

    def __init__(self, luxury_brands: list[str] | None = None) -> None:
        """Initialize feature engineer.

        Args:
            luxury_brands: List of recognized luxury marque names.
        """
        self.luxury_brands = luxury_brands or get_config().features.luxury_brands

    def fit(
        self,
        X: pd.DataFrame,  # noqa: ARG002
        y: Any = None,  # noqa: ARG002
    ) -> "DomainFeatureEngineer":
        """Fit transformer (stateless).

        Args:
            X: Input DataFrame.
            y: Ignored target.

        Returns:
            self: Fitted instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features on input DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame augmented with engineered features.
        """
        out = X.copy()
        safe_age = np.maximum(out["vehicle_age"], 1)

        # 1. km_per_year (annual odometer intensity)
        out["km_per_year"] = out["km_driven"] / safe_age

        # 2. power_per_litre (engine tuning density)
        engine_litres = np.maximum(out["engine"] / 1000.0, 0.5)
        out["power_per_litre"] = out["max_power"] / engine_litres

        # 3. is_luxury_brand (brand equity tier indicator)
        out["is_luxury_brand"] = out["brand"].isin(self.luxury_brands).astype(float)

        # 4. age_x_power (age-power interaction term)
        out["age_x_power"] = out["vehicle_age"] * out["max_power"]

        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return expanded feature names.

        Args:
            input_features: Base feature names.

        Returns:
            np.ndarray: Augmented column list.
        """
        base = list(input_features) if input_features is not None else []
        return np.array(base + ENGINEERED_FEATURES)


def make_cv_splitter() -> KFold:
    """Return the project's shuffled K-fold splitter (folds and seed from config).

    Returns:
        KFold: Splitter with ``splitting.cv_folds`` folds and the global seed.
    """
    config = get_config()
    return KFold(
        n_splits=config.splitting.cv_folds, shuffle=True, random_state=config.project.random_seed
    )


def build_model_encoder(model_encoding: str = "target") -> Pipeline:
    """Build the encoder for the high-cardinality ``model`` column (D-014).

    Args:
        model_encoding: 'target' (cross-fitted TargetEncoder, the production choice) or
            'onehot' (the alternative used only for the encoder comparison).

    Returns:
        Pipeline: Single-step encoder pipeline.

    Raises:
        ValueError: If model_encoding is not 'target' or 'onehot'.
    """
    if model_encoding == "target":
        return Pipeline([("target_enc", TargetEncoder(cv=make_cv_splitter(), smooth="auto"))])
    if model_encoding == "onehot":
        min_freq = get_config().feature_checks.model_onehot_min_frequency
        encoder = OneHotEncoder(
            handle_unknown="ignore", min_frequency=min_freq, sparse_output=False
        )
        return Pipeline([("onehot", encoder)])
    raise ValueError(f"Invalid model_encoding: '{model_encoding}'. Expected 'target' or 'onehot'.")


def build_preprocessor(kind: str = "tree", model_encoding: str = "target") -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer configured for linear or tree model families.

    Args:
        kind: Model architecture type ('linear' or 'tree').
        model_encoding: Encoder for the ``model`` column; see build_model_encoder().

    Returns:
        ColumnTransformer: Leak-free multi-feature preprocessor.

    Raises:
        ValueError: If kind is not 'linear' or 'tree'.
    """
    if kind not in ["linear", "tree"]:
        raise ValueError(f"Invalid preprocessor kind: '{kind}'. Expected 'linear' or 'tree'.")

    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "linear":
        num_steps.append(("scaler", StandardScaler()))

    min_freq = get_config().features.rare_category_min_frequency
    cat_pipeline = Pipeline(
        [
            (
                "rare_encoder",
                RareCategoryEncoder(columns=CATEGORICAL_FEATURES, min_frequency=min_freq),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("model", build_model_encoder(model_encoding), [HIGH_CARDINALITY_FEATURE]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(kind: str = "tree", model_encoding: str = "target") -> Pipeline:
    """Construct complete end-to-end feature pipeline including domain engineering.

    Args:
        kind: 'linear' or 'tree'.
        model_encoding: Encoder for the ``model`` column ('target' in production).

    Returns:
        Pipeline: End-to-end feature preparation pipeline.
    """
    return Pipeline(
        [
            ("engineer", DomainFeatureEngineer()),
            ("preprocess", build_preprocessor(kind=kind, model_encoding=model_encoding)),
        ]
    )


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add every candidate feature (kept or not) so the Ridge probe can select among them.

    Args:
        df: Training DataFrame with the cleaned schema.

    Returns:
        pd.DataFrame: Copy with engineered features plus the rejected candidates.
    """
    out = DomainFeatureEngineer().transform(df)
    out["log_km_driven"] = np.log1p(out["km_driven"])
    is_automatic = (out["transmission_type"] == "Automatic").astype(float)
    out["power_x_automatic"] = out["max_power"] * is_automatic
    return out


def build_ridge_probe(numeric_cols: list[str], model_encoding: str = "target") -> Pipeline:
    """Build the scaled Ridge pipeline used to screen features and encoders (D-013, D-014).

    Args:
        numeric_cols: Numeric columns to include.
        model_encoding: Encoder for the ``model`` column ('target' or 'onehot').

    Returns:
        Pipeline: Unfitted preprocessing + Ridge pipeline.
    """
    config = get_config()
    numeric = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=config.features.rare_category_min_frequency,
        sparse_output=False,
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, numeric_cols),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("model", build_model_encoder(model_encoding), [HIGH_CARDINALITY_FEATURE]),
        ]
    )
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=config.feature_checks.probe_ridge_alpha))])


def cv_log_rmse(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Return mean K-fold CV RMSE of a pipeline on the log1p price scale.

    Args:
        pipe: Unfitted pipeline.
        X: Training features.
        y: log1p(selling_price).

    Returns:
        float: Mean RMSE across the project's CV folds (unrounded).
    """
    scores = cross_val_score(
        pipe, X, y, cv=make_cv_splitter(), scoring="neg_root_mean_squared_error"
    )
    return -float(scores.mean())


def candidate_numeric_columns(name: str, mode: str) -> list[str]:
    """Return the probe's numeric columns for one candidate.

    Args:
        name: Candidate column name.
        mode: 'add' appends it to the base set; 'replace' swaps it in for km_driven.

    Returns:
        list[str]: Numeric column list for the probe.
    """
    if mode == "replace":
        return [name if col == "km_driven" else col for col in BASE_NUMERIC_FEATURES]
    return [*BASE_NUMERIC_FEATURES, name]


def feature_check_record(
    name: str, description: str, rmse: float, baseline_rmse: float
) -> dict[str, Any]:
    """Apply the keep rule (gain >= feature_checks.min_rmse_gain) to one measured candidate.

    Args:
        name: Candidate identifier.
        description: Plain-language description.
        rmse: Measured CV log RMSE with the candidate.
        baseline_rmse: Measured CV log RMSE without it.

    Returns:
        dict[str, Any]: Row for feature_checks.csv.
    """
    min_gain = get_config().feature_checks.min_rmse_gain
    delta = rmse - baseline_rmse
    decision = "KEEP" if delta <= -min_gain else "REJECT"
    verb = "lowers" if decision == "KEEP" else "does not lower"
    return {
        "candidate_feature": name,
        "description": description,
        "baseline_cv_rmse": round(baseline_rmse, 4),
        "cv_rmse": round(rmse, 4),
        "delta_vs_baseline": round(delta, 4),
        "decision": decision,
        "rationale": f"Measured: {verb} Ridge CV log RMSE by at least {min_gain}",
    }


def evaluate_feature_candidates(
    train_df: pd.DataFrame,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """Measure each candidate feature with the Ridge probe on train-only CV and save the table.

    Every number in feature_checks.csv is computed here; nothing is typed in by hand.

    Args:
        train_df: Training partition DataFrame (never the test set).
        tables_dir: Optional destination path for summary table.

    Returns:
        pd.DataFrame: One row per candidate plus a 'combined_kept' row.
    """
    out_dir = tables_dir or get_config().paths.tables_dir
    X = add_candidate_columns(train_df)
    y = np.log1p(train_df["selling_price"])

    base_rmse = cv_log_rmse(build_ridge_probe(BASE_NUMERIC_FEATURES), X, y)
    logger.info("Baseline Ridge CV log RMSE: %.4f", base_rmse)
    records = []
    for name, description, mode in CANDIDATE_FEATURES:
        rmse = cv_log_rmse(build_ridge_probe(candidate_numeric_columns(name, mode)), X, y)
        records.append(feature_check_record(name, description, rmse, base_rmse))

    kept = [r["candidate_feature"] for r in records if r["decision"] == "KEEP"]
    if kept != ENGINEERED_FEATURES:
        logger.warning(
            "Kept features %s differ from pipeline features %s", kept, ENGINEERED_FEATURES
        )
    combined = cv_log_rmse(build_ridge_probe(NUMERIC_FEATURES), X, y)
    records.append(
        feature_check_record("combined_kept", "All pipeline features together", combined, base_rmse)
    )

    checks_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks_df.to_csv(out_dir / "feature_checks.csv", index=False)
    logger.info("Saved feature checks table to %s", out_dir / "feature_checks.csv")
    return checks_df


def main() -> None:
    """Reproduce Phase 3: write the train/test split, then the feature-check table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_df, _, _ = create_train_test_split()
    evaluate_feature_candidates(train_df)


if __name__ == "__main__":
    main()
