"""Plotly chart builders for every page (A7 palette, finding titles live in chart_card)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from carprice.formatting import format_lakh
from components.palette import (
    ACCENT_AMBER,
    CATEGORICAL_PALETTE,
    INK_COLOR,
    PRIMARY_TEAL,
    SLATE,
    SURFACE_COLOR,
    TEAL_TINT,
    style_figure,
)

LAKH = 100000.0
FUEL_COLOURS = {
    "Petrol": PRIMARY_TEAL,
    "Diesel": INK_COLOR,
    "Cng": CATEGORICAL_PALETTE[3],
    "Lpg": CATEGORICAL_PALETTE[4],
}
ERROR_DISPLAY_LIMIT = 60
HIST_BINS = 40
# Brand bars carry '₹ x lakh · n listings' outside the bar; leave room for it.
BRAND_LABEL_ROOM = 1.75
LAKH_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]


def lakh_ticks(low: float, high: float) -> list[float]:
    """1-2-5 tick values (₹ lakh) that fall within [low, high]."""
    return [t for t in LAKH_TICKS if low * 0.8 <= t <= high * 1.25]


def factor_chart(factors: pd.DataFrame) -> go.Figure:
    """Top drivers for one car: teal raises the price, slate lowers it; values sit in the labels."""
    ordered = factors.iloc[::-1]
    colours = [PRIMARY_TEAL if v >= 0 else SLATE for v in ordered["effect_pct"]]
    labels = [
        f"{name} · {v:+.0f}%"
        for name, v in zip(ordered["label"], ordered["effect_pct"], strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["effect_pct"],
            y=labels,
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y} compared with an average car<extra></extra>",
        )
    )
    style_figure(fig, 48 * len(factors) + 48)
    fig.update_xaxes(
        title="Effect vs an average car",
        showticklabels=False,
        zeroline=True,
        zerolinecolor=INK_COLOR,
        zerolinewidth=1.5,
        showgrid=False,
    )
    fig.update_yaxes(automargin=True, showgrid=False)
    return fig


def whatif_chart(curve: pd.DataFrame, field: str, current: float, x_label: str) -> go.Figure:
    """Predicted price and 80% range as one input changes; the current choice marked in ink."""
    x = curve[field]
    fig = go.Figure(
        [
            go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(curve["upper_inr"] / LAKH) + list(curve["lower_inr"][::-1] / LAKH),
                fill="toself",
                fillcolor=TEAL_TINT,
                line={"width": 0},
                hoverinfo="skip",
                name="80% range",
            ),
            go.Scatter(
                x=x,
                y=curve["predicted_inr"] / LAKH,
                mode="lines",
                line={"color": PRIMARY_TEAL, "width": 2.5, "shape": "hv"},
                name="Estimate",
                hovertemplate="%{x:,}: ₹ %{y:.2f} lakh<extra></extra>",
            ),
        ]
    )
    here = curve.loc[(x - current).abs().idxmin(), "predicted_inr"] / LAKH
    fig.add_trace(
        go.Scatter(
            x=[current],
            y=[here],
            mode="markers",
            name="This car",
            marker={"color": INK_COLOR, "size": 11, "line": {"color": SURFACE_COLOR, "width": 2}},
            hovertemplate="This car: ₹ %{y:.2f} lakh<extra></extra>",
        )
    )
    style_figure(fig, 280)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Estimate (₹ lakh)")
    return fig


def compare_chart(estimate: float, peer_price: float, peer_label: str) -> go.Figure:
    """This car's estimate beside the median listing price of its peers (values in the labels)."""
    labels = [f"{peer_label}<br>{format_lakh(peer_price)}", f"This car<br>{format_lakh(estimate)}"]
    fig = go.Figure(
        go.Bar(
            x=[peer_price / LAKH, estimate / LAKH],
            y=labels,
            orientation="h",
            marker_color=[SLATE, PRIMARY_TEAL],
            hovertemplate="%{y}<extra></extra>",
        )
    )
    style_figure(fig, 170)
    fig.update_xaxes(title="Price (₹ lakh)", rangemode="tozero", nticks=4)
    fig.update_yaxes(automargin=True, showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Market Explorer
# ---------------------------------------------------------------------------


def price_histogram(listings: pd.DataFrame, log_x: bool) -> go.Figure:
    """Distribution of listing prices; on the log view bins are uniform in log10(price)."""
    prices = listings["selling_price"] / LAKH
    values = np.log10(prices) if log_x else prices
    counts, edges = np.histogram(values, bins=HIST_BINS)
    centres = (edges[:-1] + edges[1:]) / 2
    shown = 10**centres if log_x else centres
    fig = go.Figure(
        go.Bar(
            x=centres,
            y=counts,
            width=np.diff(edges) * 0.92,
            marker_color=PRIMARY_TEAL,
            customdata=shown,
            hovertemplate="About ₹ %{customdata:.1f} lakh: %{y:,} listings<extra></extra>",
        )
    )
    median = float(prices.median())
    position = np.log10(median) if log_x else median
    fig.add_vline(x=position, line={"color": INK_COLOR, "dash": "dash", "width": 1.5})
    fig.add_annotation(
        x=position,
        y=1,
        yref="paper",
        text=f"Median {format_lakh(median * LAKH)}",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font={"color": INK_COLOR},
    )
    style_figure(fig, 320)
    fig.update_yaxes(title="Listings")
    if log_x:
        ticks = lakh_ticks(float(prices.min()), float(prices.max()))
        fig.update_xaxes(
            title="Listing price (₹ lakh, log scale)",
            tickvals=np.log10(ticks),
            ticktext=[f"{t:g}" for t in ticks],
        )
    else:
        fig.update_xaxes(title="Listing price (₹ lakh)")
    return fig


def age_trend_chart(by_age: pd.DataFrame, trend: pd.Series) -> go.Figure:
    """Median price per age (dot size = listings) with the count-weighted trend."""
    sizes = 6 + 22 * np.sqrt(by_age["n_listings"] / by_age["n_listings"].max())
    fig = go.Figure(
        [
            go.Scatter(
                x=by_age["vehicle_age"],
                y=by_age["median_price"] / LAKH,
                mode="lines+markers",
                line={"color": PRIMARY_TEAL, "width": 2},
                marker={
                    "size": sizes,
                    "color": PRIMARY_TEAL,
                    "line": {"color": SURFACE_COLOR, "width": 1},
                },
                name="Median price",
                customdata=by_age["n_listings"],
                hovertemplate="%{x} years: ₹ %{y:.2f} lakh (%{customdata:,} "
                "listings)<extra></extra>",
            ),
            go.Scatter(
                x=by_age["vehicle_age"],
                y=trend / LAKH,
                mode="lines",
                name="Weighted trend",
                line={"color": INK_COLOR, "dash": "dash", "width": 1.5},
                hoverinfo="skip",
            ),
        ]
    )
    style_figure(fig, 320, show_legend=True)
    fig.update_xaxes(title="Vehicle age (years)")
    ticks = lakh_ticks(
        float(by_age["median_price"].min() / LAKH), float(by_age["median_price"].max() / LAKH)
    )
    fig.update_yaxes(
        title="Median price (₹ lakh, log scale)",
        type="log",
        tickvals=ticks,
        ticktext=[f"{t:g}" for t in ticks],
    )
    return fig


def power_scatter(sample: pd.DataFrame) -> go.Figure:
    """Price against power, one colour per fuel; hover names the car."""
    fig = go.Figure()
    for fuel, group in sample.groupby("fuel_type"):
        fig.add_trace(
            go.Scattergl(
                x=group["max_power"],
                y=group["selling_price"] / LAKH,
                mode="markers",
                name=fuel,
                marker={"color": FUEL_COLOURS.get(fuel, SLATE), "size": 6, "opacity": 0.55},
                customdata=np.stack(
                    [group["brand"], group["model"], group["vehicle_age"]], axis=-1
                ),
                hovertemplate="%{customdata[0]} %{customdata[1]}, %{customdata[2]} yrs<br>"
                "%{x:.0f} bhp, ₹ %{y:.2f} lakh<extra></extra>",
            )
        )
    style_figure(fig, 320, show_legend=True)
    fig.update_xaxes(title="Max power (bhp)")
    ticks = lakh_ticks(
        float(sample["selling_price"].min() / LAKH), float(sample["selling_price"].max() / LAKH)
    )
    fig.update_yaxes(
        title="Listing price (₹ lakh, log scale)",
        type="log",
        tickvals=ticks,
        ticktext=[f"{t:g}" for t in ticks],
    )
    return fig


def brand_chart(ranking: pd.DataFrame) -> go.Figure:
    """Median price by brand with listing counts; the top brand is the view's amber highlight."""
    ordered = ranking.iloc[::-1]
    colours = [
        ACCENT_AMBER if b == ranking.iloc[0]["brand"] else PRIMARY_TEAL for b in ordered["brand"]
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["median_price"] / LAKH,
            y=ordered["brand"],
            orientation="h",
            marker_color=colours,
            text=[
                f"{format_lakh(p)} · {n:,} listings"
                for p, n in zip(ordered["median_price"], ordered["n_listings"], strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: median ₹ %{x:.2f} lakh<extra></extra>",
        )
    )
    style_figure(fig, 34 * len(ranking) + 60)
    fig.update_xaxes(
        title="Median listing price (₹ lakh)",
        range=[0, float(ranking["median_price"].max() / LAKH) * BRAND_LABEL_ROOM],
    )
    fig.update_yaxes(automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------


def predicted_vs_actual(predictions: pd.DataFrame) -> go.Figure:
    """Held-out test cars: actual vs predicted on log axes, hover shows the car."""
    actual, pred = predictions["actual_inr"] / LAKH, predictions["predicted_inr"] / LAKH
    lo, hi = float(min(actual.min(), pred.min())) * 0.9, float(max(actual.max(), pred.max())) * 1.1
    fig = go.Figure(
        [
            go.Scattergl(
                x=pred,
                y=actual,
                mode="markers",
                name="Test car",
                marker={"color": PRIMARY_TEAL, "size": 6, "opacity": 0.45},
                customdata=np.stack(
                    [
                        predictions["brand"],
                        predictions["model"],
                        predictions["vehicle_age"],
                        predictions["pct_error"],
                    ],
                    axis=-1,
                ),
                hovertemplate="%{customdata[0]} %{customdata[1]}, %{customdata[2]} yrs<br>"
                "Predicted ₹ %{x:.2f} lakh, actual ₹ %{y:.2f} lakh "
                "(%{customdata[3]:+.1f}%)<extra></extra>",
            ),
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect prediction",
                line={"color": INK_COLOR, "dash": "dash", "width": 1.5},
                hoverinfo="skip",
            ),
        ]
    )
    style_figure(fig, 420, show_legend=True)
    ticks = lakh_ticks(lo, hi)
    ticktext = [f"{t:g}" for t in ticks]
    fig.update_xaxes(
        title="Predicted price (₹ lakh, log scale)", type="log", tickvals=ticks, ticktext=ticktext
    )
    fig.update_yaxes(
        title="Actual price (₹ lakh, log scale)", type="log", tickvals=ticks, ticktext=ticktext
    )
    return fig


def error_histogram(predictions: pd.DataFrame) -> go.Figure:
    """Signed % error with the ±10% band shaded."""
    errors = predictions["pct_error"].clip(-ERROR_DISPLAY_LIMIT, ERROR_DISPLAY_LIMIT)
    fig = go.Figure(
        go.Histogram(
            x=errors,
            xbins={"start": -ERROR_DISPLAY_LIMIT, "end": ERROR_DISPLAY_LIMIT, "size": 2},
            marker_color=PRIMARY_TEAL,
            marker_line={"color": SURFACE_COLOR, "width": 1},
            hovertemplate="%{x}%: %{y:,} cars<extra></extra>",
        )
    )
    fig.add_vrect(x0=-10, x1=10, fillcolor=TEAL_TINT, line_width=0, layer="below")
    style_figure(fig, 320)
    fig.update_xaxes(title="Error: (predicted - actual) / actual (%)", ticksuffix="%")
    fig.update_yaxes(title="Test cars")
    return fig


def segment_chart(segments: pd.DataFrame, min_rows: int) -> go.Figure:
    """Within-10% share per segment; the weakest eligible segment is the amber highlight."""
    ordered = segments.iloc[::-1]
    eligible = segments[segments["n_test"] >= min_rows]
    weakest = eligible["within_10pct"].idxmin() if not eligible.empty else None
    colours = [
        ACCENT_AMBER if i == weakest else (PRIMARY_TEAL if n >= min_rows else SLATE)
        for i, n in zip(ordered.index, ordered["n_test"], strict=True)
    ]
    labels = [f"{s}  (n={n:,})" for s, n in zip(ordered["segment"], ordered["n_test"], strict=True)]
    fig = go.Figure(
        go.Bar(
            x=ordered["within_10pct"],
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.1f}%" for v in ordered["within_10pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}% within 10%<extra></extra>",
        )
    )
    style_figure(fig, 30 * len(segments) + 60)
    fig.update_xaxes(title="Cars priced within 10% (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(automargin=True)
    return fig
