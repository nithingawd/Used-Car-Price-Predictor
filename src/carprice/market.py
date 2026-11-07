"""Market data summaries for the GUI's Market Explorer and comparison panels (Phase 7).

Pure functions on a listings DataFrame (the cleaned data, ``cars_clean.csv``). Every number
shown in Market Explorer is computed here from the filtered data, so text and charts agree.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MarketFilter:
    """User filter. Empty tuples mean 'all'; ranges are inclusive (price in rupees)."""

    brands: tuple[str, ...] = ()
    fuels: tuple[str, ...] = ()
    transmissions: tuple[str, ...] = ()
    age_range: tuple[int, int] | None = None
    price_range: tuple[float, float] | None = None


def apply_filter(listings: pd.DataFrame, market_filter: MarketFilter) -> pd.DataFrame:
    """Rows matching every active part of the filter."""
    mask = pd.Series(True, index=listings.index)
    for column, chosen in (
        ("brand", market_filter.brands),
        ("fuel_type", market_filter.fuels),
        ("transmission_type", market_filter.transmissions),
    ):
        if chosen:
            mask &= listings[column].isin(chosen)
    if market_filter.age_range is not None:
        mask &= listings["vehicle_age"].between(*market_filter.age_range)
    if market_filter.price_range is not None:
        mask &= listings["selling_price"].between(*market_filter.price_range)
    return listings[mask]


def summary(listings: pd.DataFrame) -> dict[str, float]:
    """Listing count, median price and median age."""
    return {
        "n_listings": len(listings),
        "median_price": float(listings["selling_price"].median()),
        "median_age": float(listings["vehicle_age"].median()),
    }


def median_by_age(listings: pd.DataFrame) -> pd.DataFrame:
    """Median price and listing count for each vehicle age."""
    grouped = listings.groupby("vehicle_age")["selling_price"]
    return pd.DataFrame(
        {"median_price": grouped.median(), "n_listings": grouped.size()}
    ).reset_index()


def annual_depreciation_pct(by_age: pd.DataFrame) -> float | None:
    """Count-weighted log-linear trend: average % change in median price per year of age.

    Returns:
        float | None: Negative when prices fall with age; None with fewer than two ages.
    """
    if len(by_age) < 2:
        return None
    slope, _ = np.polyfit(
        by_age["vehicle_age"], np.log(by_age["median_price"]), 1, w=np.sqrt(by_age["n_listings"])
    )
    return float(np.expm1(slope) * 100)


def trend_line(by_age: pd.DataFrame) -> pd.Series:
    """Fitted median price at each age from the count-weighted log-linear trend."""
    slope, intercept = np.polyfit(
        by_age["vehicle_age"], np.log(by_age["median_price"]), 1, w=np.sqrt(by_age["n_listings"])
    )
    return pd.Series(np.exp(intercept + slope * by_age["vehicle_age"]), index=by_age.index)


def brand_ranking(listings: pd.DataFrame, min_listings: int, top_n: int) -> pd.DataFrame:
    """Brands with at least min_listings rows, by median price (highest first)."""
    grouped = listings.groupby("brand")["selling_price"]
    table = pd.DataFrame({"median_price": grouped.median(), "n_listings": grouped.size()})
    table = table[table["n_listings"] >= min_listings]
    return table.sort_values("median_price", ascending=False).head(top_n).reset_index()


def power_price_correlation(listings: pd.DataFrame) -> float | None:
    """Spearman rank correlation between max power and price (None if too few rows)."""
    if len(listings) < 3 or listings["max_power"].nunique() < 2:
        return None
    return float(stats.spearmanr(listings["max_power"], listings["selling_price"]).statistic)


def peer_median(listings: pd.DataFrame, brand: str, age: int) -> dict[str, float] | None:
    """Median price of the same brand at the same age (None when no such listings)."""
    peers = listings[(listings["brand"] == brand) & (listings["vehicle_age"] == age)]
    if peers.empty:
        return None
    return {"median_price": float(peers["selling_price"].median()), "n_listings": len(peers)}
