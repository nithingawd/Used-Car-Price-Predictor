"""Derived figures quoted in the written report, computed from saved outputs (Phase 8).

Shares, ratios and relative improvements are not stored in any Phase 1 to 6 table, but the
final report quotes them. Rather than computing them by hand, this module derives each one
from the saved tables and data files and writes ``reports/tables/report_key_figures.csv``
with the formula and source files, so every quoted number has a traceable origin.

Run with ``python -m carprice.report_facts`` (or ``python tasks.py facts``).
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carprice.config import AppConfig, get_config

logger = logging.getLogger(__name__)

OUTPUT_NAME = "report_key_figures.csv"
DECIMALS = 2


@dataclass(frozen=True)
class KeyFigure:
    """One derived figure with how it was computed.

    Attributes:
        figure_id: Stable snake_case identifier.
        value: The derived value, rounded to ``DECIMALS``.
        unit: Unit of the value (pct, ratio, count, pct_points).
        formula: Plain description of the calculation.
        sources: Files the inputs were read from (``;``-separated).
    """

    figure_id: str
    value: float
    unit: str
    formula: str
    sources: str


def share_pct(counts: pd.Series, level: str) -> float:
    """Percentage share of one level in a count series (``count / total * 100``)."""
    return round(float(counts[level] / counts.sum() * 100), DECIMALS)


def ratio(numerator: float, denominator: float) -> float:
    """Rounded ratio of two positive numbers."""
    return round(float(numerator / denominator), DECIMALS)


def reduction_pct(baseline: float, improved: float) -> float:
    """Percentage by which ``improved`` is lower than ``baseline``."""
    return round(float((baseline - improved) / baseline * 100), DECIMALS)


def _read_table(config: AppConfig, name: str) -> pd.DataFrame:
    """Read ``reports/tables/<name>.csv``."""
    return pd.read_csv(config.paths.tables_dir / f"{name}.csv")


def _counts(table: pd.DataFrame, key: str) -> pd.Series:
    """Row counts indexed by a category column of an EDA medians table."""
    return table.set_index(key)["count"]


def _medians(table: pd.DataFrame, key: str) -> pd.Series:
    """Median prices indexed by a category column of an EDA medians table."""
    return table.set_index(key)["median"]


def share_figures(config: AppConfig) -> list[KeyFigure]:
    """Market shares of brands, transmissions, fuels and seat counts in the cleaned data."""
    specs = [
        ("eda_brand_medians", "brand", ["Maruti", "Hyundai"]),
        ("eda_transmission_medians", "transmission_type", ["Manual", "Automatic"]),
        ("eda_fuel_medians", "fuel_type", ["Petrol", "Diesel", "Cng", "Lpg"]),
        ("eda_seats_medians", "seats", [5, 7]),
    ]
    figures = []
    for table_name, key, levels in specs:
        counts = _counts(_read_table(config, table_name), key)
        for level in levels:
            figure_id = f"share_{key}_{str(level).lower()}_pct"
            formula = f"count({key}={level}) / all cleaned rows * 100"
            value = share_pct(counts, level)
            figures.append(KeyFigure(figure_id, value, "pct", formula, f"{table_name}.csv"))
    brand_counts = _counts(_read_table(config, "eda_brand_medians"), "brand")
    top_two = round(float(brand_counts.nlargest(2).sum() / brand_counts.sum() * 100), DECIMALS)
    figures.append(
        KeyFigure(
            "share_top_two_brands_pct",
            top_two,
            "pct",
            "(count of the two largest brands) / all cleaned rows * 100",
            "eda_brand_medians.csv",
        )
    )
    return figures


def median_ratio_figures(config: AppConfig) -> list[KeyFigure]:
    """Ratios of median prices between categories (raw gaps before controlling for specs)."""
    specs = [
        ("eda_transmission_medians", "transmission_type", "Automatic", "Manual"),
        ("eda_fuel_medians", "fuel_type", "Diesel", "Petrol"),
        ("eda_seller_medians", "seller_type", "Dealer", "Individual"),
        ("eda_seats_medians", "seats", 7, 5),
    ]
    figures = []
    for table_name, key, top, bottom in specs:
        medians = _medians(_read_table(config, table_name), key)
        figure_id = f"median_ratio_{str(top).lower()}_vs_{str(bottom).lower()}"
        formula = f"median price ({top}) / median price ({bottom})"
        value = ratio(medians[top], medians[bottom])
        figures.append(KeyFigure(figure_id, value, "ratio", formula, f"{table_name}.csv"))
    return figures


def _metric(metrics: pd.DataFrame, model_id: str, column: str) -> float:
    """One metric value for one model from test_metrics.csv."""
    return float(metrics.set_index("model_id").loc[model_id, column])


def test_improvement_figures(config: AppConfig) -> list[KeyFigure]:
    """How much the final model improves on the Ridge and dummy baselines on the test set."""
    metrics = _read_table(config, "test_metrics")
    figures = []
    for baseline in ("ridge_baseline", "dummy_baseline"):
        short = baseline.split("_")[0]
        for column in ("mae_inr", "rmse_log"):
            value = reduction_pct(
                _metric(metrics, baseline, column), _metric(metrics, "final", column)
            )
            formula = f"({column} {short} - {column} final) / {column} {short} * 100"
            figures.append(
                KeyFigure(
                    f"test_{column}_reduction_vs_{short}_pct",
                    value,
                    "pct",
                    formula,
                    "test_metrics.csv",
                )
            )
        gain = _metric(metrics, "final", "within_10pct") - _metric(
            metrics, baseline, "within_10pct"
        )
        figures.append(
            KeyFigure(
                f"test_within_10pct_gain_vs_{short}_points",
                round(gain, DECIMALS),
                "pct_points",
                f"within_10pct final - within_10pct {short}",
                "test_metrics.csv",
            )
        )
    return figures


def data_scope_figures(config: AppConfig) -> list[KeyFigure]:
    """Counts of brands and models in the cleaned data and the training split."""
    clean = pd.read_csv(config.paths.interim_data_dir / "cars_clean.csv")
    train = pd.read_csv(config.paths.processed_data_dir / "train.csv")
    figures = []
    for frame, label, source in (
        (clean, "clean", "data/interim/cars_clean.csv"),
        (train, "train", "data/processed/train.csv"),
    ):
        for column in ("brand", "model"):
            figures.append(
                KeyFigure(
                    f"{label}_{column}_count",
                    float(frame[column].nunique()),
                    "count",
                    f"distinct {column} values",
                    source,
                )
            )
    return figures


def distribution_figures(config: AppConfig) -> list[KeyFigure]:
    """Skewness of the target before and after log1p, and the number of rare brands."""
    target = config.features.target_column
    figures = []
    for label, path in (
        ("raw", config.paths.raw_data_path),
        ("clean", config.paths.interim_data_dir / "cars_clean.csv"),
    ):
        price = pd.read_csv(path)[target]
        source = path.relative_to(config.paths.root_dir).as_posix()
        for transform, values in (("price", price), ("log1p_price", np.log1p(price))):
            figures.append(
                KeyFigure(
                    f"skewness_{label}_{transform}",
                    round(float(values.skew()), DECIMALS),
                    "skewness",
                    f"sample skewness of {transform} (pandas Series.skew)",
                    source,
                )
            )
    rare = _read_table(config, "eda_rare_brands")
    figures.append(
        KeyFigure(
            "rare_brand_count",
            float(len(rare)),
            "count",
            f"brands with fewer than {config.eda.rare_brand_report_threshold} cleaned rows",
            "eda_rare_brands.csv",
        )
    )
    return figures


def build_key_figures(config: AppConfig) -> pd.DataFrame:
    """Every derived figure as one table (columns follow the KeyFigure fields)."""
    figures = (
        share_figures(config)
        + median_ratio_figures(config)
        + test_improvement_figures(config)
        + data_scope_figures(config)
        + distribution_figures(config)
    )
    return pd.DataFrame([figure.__dict__ for figure in figures])


def write_key_figures(config: AppConfig | None = None) -> Path:
    """Build the key-figure table and save it to ``reports/tables/report_key_figures.csv``."""
    config = config or get_config()
    table = build_key_figures(config)
    path = config.paths.tables_dir / OUTPUT_NAME
    table.to_csv(path, index=False)
    logger.info("wrote %d key figures to %s", len(table), path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    write_key_figures()
