"""Page 2: Market explorer. Filter the cleaned listings and read the market's story."""

import pandas as pd
import streamlit as st

from carprice.config import get_config
from carprice.market import (
    MarketFilter,
    annual_depreciation_pct,
    apply_filter,
    brand_ranking,
    median_by_age,
    power_price_correlation,
    summary,
    trend_line,
)
import components  # noqa: F401  (puts src/ on the import path)
from components.charts import age_trend_chart, brand_chart, power_scatter, price_histogram
from components.data import market_listings
from components.ui import (
    chart_card,
    empty_state,
    format_lakh,
    metric_card,
    page_header,
    panel,
    safely,
)

LAKH = 100000.0


def filter_bar(listings: pd.DataFrame) -> MarketFilter:
    """Brand, fuel, transmission, age and price filters in one row (7.3.1)."""
    with panel("filters"):
        top = st.columns([3, 2, 2])
        brands = top[0].multiselect(
            "Brands", sorted(listings["brand"].unique()), placeholder="All brands"
        )
        fuels = top[1].multiselect(
            "Fuel", sorted(listings["fuel_type"].unique()), placeholder="All fuels"
        )
        gears = top[2].multiselect(
            "Transmission", sorted(listings["transmission_type"].unique()), placeholder="Both"
        )
        bottom = st.columns(2)
        max_age = int(listings["vehicle_age"].max())
        ages = bottom[0].slider("Vehicle age (years)", 0, max_age, (0, max_age))
        max_lakh = float(listings["selling_price"].max() / LAKH)
        prices = bottom[1].slider(
            "Price (₹ lakh)", 0.0, round(max_lakh, 0), (0.0, round(max_lakh, 0)), step=0.5
        )
    return MarketFilter(
        tuple(brands), tuple(fuels), tuple(gears), ages, (prices[0] * LAKH, prices[1] * LAKH)
    )


def summary_strip(filtered: pd.DataFrame) -> None:
    """Listings, median price and median age for the current filter (7.3.3)."""
    stats = summary(filtered)
    cells = st.columns(3)
    with cells[0]:
        metric_card("Listings", f"{stats['n_listings']:,}", "Cars matching the filters.")
    with cells[1]:
        metric_card(
            "Median price", format_lakh(stats["median_price"]), "Half of these cars list for less."
        )
    with cells[2]:
        metric_card("Median age", f"{stats['median_age']:.0f} years", "Typical age of these cars.")


def distribution_chart(filtered: pd.DataFrame) -> None:
    """Price distribution with an optional log axis."""
    log_x = st.toggle(
        "Log price axis", value=True, help="A log axis spreads out the many cheaper cars."
    )
    median = filtered["selling_price"].median()
    q75 = filtered["selling_price"].quantile(0.75)
    chart_card(
        price_histogram(filtered, log_x),
        f"Half of these listings ask less than {format_lakh(median)}",
        f"Three in four list below {format_lakh(q75)}; a long tail of luxury cars stretches far "
        "to the right, which is why the model works on log prices.",
    )


def depreciation_chart(filtered: pd.DataFrame) -> None:
    """Median price by age with the count-weighted trend."""
    min_rows = get_config().gui.age_min_listings
    by_age = median_by_age(filtered)
    by_age = by_age[by_age["n_listings"] >= min_rows].reset_index(drop=True)
    rate = annual_depreciation_pct(by_age)
    if rate is None:
        empty_state("Not enough ages", "Widen the age filter to see how prices change with age.")
        return
    chart_card(
        age_trend_chart(by_age, trend_line(by_age)),
        f"Prices {'fall' if rate < 0 else 'rise'} about {abs(rate):.0f}% per year of age",
        "Bigger dots mean more listings; the dashed line weights each age by how many cars it "
        f"has. Ages with fewer than {min_rows} listings are left out so single cars do not "
        "distort the picture.",
    )


def power_chart(filtered: pd.DataFrame) -> None:
    """Price against power, coloured by fuel (sampled for speed)."""
    limit = get_config().gui.scatter_max_points
    sample = filtered.sample(
        min(limit, len(filtered)), random_state=get_config().project.random_seed
    )
    rho = power_price_correlation(filtered)
    title = "More power, higher price" + (
        f" (rank correlation {rho:.2f})" if rho is not None else ""
    )
    note = (
        f"Showing {len(sample):,} of {len(filtered):,} listings. "
        if len(sample) < len(filtered)
        else ""
    )
    chart_card(
        power_scatter(sample),
        title,
        note + "Each dot is one listing; hover to see the car. Across the whole dataset, power "
        "has the strongest rank correlation with price of any specification.",
    )


def brands_chart(filtered: pd.DataFrame) -> None:
    """Top brands by median price with listing counts."""
    gui = get_config().gui
    ranking = brand_ranking(filtered, gui.brand_chart_min_listings, gui.brand_chart_top_n)
    if ranking.empty:
        empty_state(
            "Too few listings per brand",
            f"Brands need at least {gui.brand_chart_min_listings} listings to be ranked.",
        )
        return
    leader = ranking.iloc[0]
    chart_card(
        brand_chart(ranking),
        f"{leader['brand']} has the highest median price ({format_lakh(leader['median_price'])})",
        f"Only brands with at least {gui.brand_chart_min_listings} listings are ranked; the "
        "label shows how many listings each median rests on.",
    )


def explorer() -> None:
    """Filter bar, summary strip and the four charts."""
    listings = market_listings()
    filtered = apply_filter(listings, filter_bar(listings))
    if filtered.empty:
        empty_state("No listings match", "Loosen one of the filters above to see the market again.")
        return
    summary_strip(filtered)
    left, right = st.columns(2, gap="medium")
    with left:
        distribution_chart(filtered)
        power_chart(filtered)
    with right:
        depreciation_chart(filtered)
        brands_chart(filtered)


page_header(
    "Market explorer",
    "How used car prices vary by age, power, fuel and brand across "
    "every cleaned listing in the dataset.",
)
safely(explorer, "Market data")
