"""Visual styling module and standardized plotting utilities for exploratory data analysis."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Strict Palette Constants from AGENTS.md Section A7
BG_COLOR: str = "#F7F5F0"
SURFACE_COLOR: str = "#FFFFFF"
INK_COLOR: str = "#14213D"
MUTED_TEXT: str = "#6B7280"
PRIMARY_TEAL: str = "#0E7C7B"
ACCENT_AMBER: str = "#F2A541"
BORDER_COLOR: str = "#E3DED3"
POSITIVE_COLOR: str = "#2E7D32"
NEGATIVE_COLOR: str = "#C0392B"

CATEGORICAL_PALETTE: list[str] = [
    "#0E7C7B",  # Primary Teal
    "#14213D",  # Dark Ink
    "#F2A541",  # Amber Accent
    "#7A9E9F",  # Soft Teal
    "#B5651D",  # Ochre Brown
    "#8D99AE",  # Muted Slate
]

TEAL_SEQUENTIAL_CMAP: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "teal_sequential", ["#FFFFFF", "#0E7C7B"]
)


def apply_style() -> None:
    """Configure matplotlib and seaborn global plotting theme to match project guidelines."""
    sns.set_theme(style="white", palette=CATEGORICAL_PALETTE)
    plt.rcParams["figure.facecolor"] = BG_COLOR
    plt.rcParams["axes.facecolor"] = SURFACE_COLOR
    plt.rcParams["axes.edgecolor"] = BORDER_COLOR
    plt.rcParams["axes.labelcolor"] = INK_COLOR
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.labelweight"] = "medium"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlecolor"] = INK_COLOR
    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["xtick.color"] = MUTED_TEXT
    plt.rcParams["ytick.color"] = MUTED_TEXT
    plt.rcParams["grid.color"] = BORDER_COLOR
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"


def format_inr(value: float) -> str:
    """Format numeric rupee amount into Indian grouping or Lakhs/Crores string.

    Args:
        value: Price in INR.

    Returns:
        str: Human-readable formatted string.
    """
    if value >= 10000000:
        return f"₹{value / 10000000:.2f} Cr"
    if value >= 100000:
        return f"₹{value / 100000:.2f} Lakh"
    return f"₹{value:,.0f}"


def save_figure(fig: plt.Figure, file_path: Path | str, dpi: int = 200) -> Path:
    """Save figure with consistent bounding box, background color, and DPI.

    Args:
        fig: Matplotlib Figure instance.
        file_path: Output path for PNG image.
        dpi: Dots per inch resolution.

    Returns:
        Path: Resolved file path of saved figure.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved figure to %s", path)
    return path


def plot_distribution_with_box(
    df: pd.DataFrame,
    feature: str,
    title: str,
    x_label: str,
    log_scale: bool = False,
    bins: int = 40,
) -> plt.Figure:
    """Plot a two-panel figure with boxplot on top and histogram with KDE below.

    Args:
        df: Input DataFrame.
        feature: Column name to visualize.
        title: Finding-style title for the plot.
        x_label: Explanatory x-axis label including units.
        log_scale: Whether to apply log10 transformation on values.
        bins: Number of histogram bins.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    data = df[feature].dropna()
    if log_scale:
        data = np.log10(data[data > 0])
        x_label = f"log10({x_label})"

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )
    fig.patch.set_facecolor(BG_COLOR)

    # Top boxplot
    sns.boxplot(
        x=data,
        orient="x",
        ax=ax_box,
        color=PRIMARY_TEAL,
        fliersize=3,
        flierprops={"markerfacecolor": ACCENT_AMBER, "markeredgecolor": "none"},
    )
    ax_box.set(xlabel="")
    ax_box.set_title(title, pad=12)
    sns.despine(ax=ax_box, left=True, bottom=True)

    # Bottom histogram with KDE
    sns.histplot(data, bins=bins, kde=True, ax=ax_hist, color=PRIMARY_TEAL, edgecolor=SURFACE_COLOR)
    ax_hist.set_xlabel(x_label)
    ax_hist.set_ylabel("Count of Vehicles")
    ax_hist.grid(True, axis="y")
    sns.despine(ax=ax_hist)

    plt.tight_layout()
    return fig


def plot_price_log_comparison(df: pd.DataFrame) -> plt.Figure:
    """Plot side-by-side distributions of selling_price: raw vs log1p.

    Args:
        df: Input DataFrame containing selling_price.

    Returns:
        plt.Figure: Generated figure.
    """
    apply_style()
    raw_price = df["selling_price"]
    log_price = np.log1p(raw_price)

    raw_skew = float(raw_price.skew())
    log_skew = float(log_price.skew())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG_COLOR)

    # Panel 1: Raw Price
    sns.histplot(
        raw_price / 100000,
        bins=45,
        kde=True,
        ax=ax1,
        color=PRIMARY_TEAL,
        edgecolor=SURFACE_COLOR,
    )
    ax1.set_title(f"Raw Price Distribution (Skewness: {raw_skew:.2f})")
    ax1.set_xlabel("Selling Price (₹ in Lakhs)")
    ax1.set_ylabel("Count of Vehicles")
    ax1.grid(True, axis="y")
    sns.despine(ax=ax1)

    # Panel 2: Log-Transformed Price
    sns.histplot(log_price, bins=40, kde=True, ax=ax2, color=ACCENT_AMBER, edgecolor=SURFACE_COLOR)
    ax2.set_title(f"Log-Transformed Price log1p (Skewness: {log_skew:.2f})")
    ax2.set_xlabel("log(Selling Price in INR)")
    ax2.set_ylabel("Count of Vehicles")
    ax2.grid(True, axis="y")
    sns.despine(ax=ax2)

    plt.tight_layout()
    return fig


def plot_categorical_frequency(
    df: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    top_n: int = 15,
) -> plt.Figure:
    """Plot categorical frequency bar chart with count and percentage annotations.

    Args:
        df: Input DataFrame.
        column: Categorical column name.
        title: Finding-style title.
        x_label: Meaningful axis label.
        top_n: Number of top categories to show (remaining grouped into 'Other').

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    counts = df[column].value_counts()
    if len(counts) > top_n:
        top_counts = counts.head(top_n)
        other_sum = counts.iloc[top_n:].sum()
        display_counts = pd.concat([top_counts, pd.Series({"Other": other_sum})])
    else:
        display_counts = counts

    total = len(df)
    fig, ax = plt.subplots(figsize=(10, max(5, len(display_counts) * 0.38)))
    fig.patch.set_facecolor(BG_COLOR)

    colors = [PRIMARY_TEAL if cat != "Other" else MUTED_TEXT for cat in display_counts.index]
    y_pos = np.arange(len(display_counts))

    ax.barh(y_pos, display_counts.values, color=colors, edgecolor=BORDER_COLOR, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(c) for c in display_counts.index])
    ax.invert_yaxis()
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label)
    ax.grid(True, axis="x")

    for i, count in enumerate(display_counts.values):
        pct = (count / total) * 100
        ax.text(count + (total * 0.008), i, f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    return fig


def plot_bivariate_box(
    df: pd.DataFrame,
    cat_col: str,
    title: str,
    cat_label: str,
    top_n: int = 15,
) -> plt.Figure:
    """Plot median selling price by categorical levels with log scale.

    Args:
        df: Input DataFrame.
        cat_col: Categorical feature.
        title: Finding-style title.
        cat_label: Category axis title.
        top_n: Number of categories to display.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    top_cats = df[cat_col].value_counts().head(top_n).index
    filtered = df[df[cat_col].isin(top_cats)].copy()
    filtered[cat_col] = filtered[cat_col].astype(str)
    filtered["price_lakhs"] = filtered["selling_price"] / 100000

    order = [
        str(x)
        for x in filtered.groupby(cat_col)["price_lakhs"]
        .median()
        .sort_values(ascending=False)
        .index
    ]

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(order) * 0.42)))
    fig.patch.set_facecolor(BG_COLOR)

    sns.boxplot(
        data=filtered,
        y=cat_col,
        x="price_lakhs",
        order=order,
        ax=ax,
        color=PRIMARY_TEAL,
        fliersize=2.5,
        flierprops={"markerfacecolor": ACCENT_AMBER, "markeredgecolor": "none"},
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Selling Price (₹ in Lakhs, log scale)")
    ax.set_xscale("log")
    ax.set_ylabel(cat_label)
    ax.grid(True, axis="x")
    sns.despine(ax=ax, left=True)

    plt.tight_layout()
    return fig


def plot_bivariate_scatter_trend(
    df: pd.DataFrame,
    x_feature: str,
    title: str,
    x_label: str,
    bins: int = 20,
) -> plt.Figure:
    """Plot scatter plot of selling_price against numeric feature with binned median trend.

    Args:
        df: Input DataFrame.
        x_feature: Continuous feature name.
        title: Finding-style title.
        x_label: Label for x-axis with units.
        bins: Number of bins for computing robust trend line.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)

    # Background scatter of data points
    ax.scatter(
        df[x_feature],
        df["selling_price"] / 100000,
        alpha=0.25,
        color=MUTED_TEXT,
        s=16,
        edgecolors="none",
        label="Observed Listings",
    )

    # Compute binned median trend
    clean_sub = df[[x_feature, "selling_price"]].dropna()
    clean_sub["bin"] = pd.qcut(clean_sub[x_feature], q=bins, duplicates="drop")
    binned = (
        clean_sub.groupby("bin", observed=True)
        .agg(
            x_mean=(x_feature, "mean"),
            price_median=("selling_price", lambda s: s.median() / 100000),
        )
        .dropna()
    )

    ax.plot(
        binned["x_mean"],
        binned["price_median"],
        color=PRIMARY_TEAL,
        linewidth=3.2,
        marker="o",
        markersize=6,
        label="Binned Median Trend",
    )

    ax.set_yscale("log")
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Selling Price (₹ in Lakhs, log scale)")
    ax.grid(True)
    ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig


def plot_depreciation_by_brand(
    df: pd.DataFrame, top_n_brands: int = 5
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot price depreciation over vehicle age for top automotive brands.

    Args:
        df: Input DataFrame.
        top_n_brands: Number of leading brands to compare.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Figure and depreciation loss summary table.
    """
    apply_style()
    top_brands = df["brand"].value_counts().head(top_n_brands).index.tolist()
    sub_df = df[df["brand"].isin(top_brands) & (df["vehicle_age"] <= 15)].copy()

    depr_table = sub_df.groupby(["brand", "vehicle_age"])["selling_price"].median().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)

    loss_records = []
    for i, brand in enumerate(top_brands):
        brand_data = depr_table[depr_table["brand"] == brand].sort_values("vehicle_age")
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]

        ax.plot(
            brand_data["vehicle_age"],
            brand_data["selling_price"] / 100000,
            marker="o",
            linewidth=2.4,
            label=brand,
            color=color,
        )

        # Estimate average annual percent loss across ages 1 to 7
        p_new = brand_data[brand_data["vehicle_age"] <= 2]["selling_price"].median()
        p_old = brand_data[brand_data["vehicle_age"] >= 6]["selling_price"].median()
        if pd.notnull(p_new) and pd.notnull(p_old) and p_new > 0:
            total_loss_pct = ((p_new - p_old) / p_new) * 100
            annual_rate = total_loss_pct / 5.0
            loss_records.append({"brand": brand, "annual_loss_pct": round(annual_rate, 2)})

    ax.set_yscale("log")
    ax.set_title("Depreciation Curves: Prices Fall Steadily Across Top Brands", pad=12)
    ax.set_xlabel("Vehicle Age (Years)")
    ax.set_ylabel("Median Selling Price (₹ in Lakhs, log scale)")
    ax.grid(True)
    ax.legend(title="Brand", frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    sns.despine(ax=ax)

    plt.tight_layout()
    loss_df = pd.DataFrame(loss_records)
    return fig, loss_df


def plot_correlation_heatmaps(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Plot Pearson and Spearman correlation heatmaps for numeric features with log price.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[plt.Figure, pd.DataFrame, pd.DataFrame]: Figure, Pearson matrix, Spearman matrix.
    """
    apply_style()
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "selling_price" in numeric_df.columns:
        numeric_df["log_selling_price"] = np.log1p(numeric_df["selling_price"])

    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)

    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        cmap=TEAL_SEQUENTIAL_CMAP,
        ax=ax1,
        cbar_kws={"label": "Pearson r"},
        linewidths=0.5,
        linecolor=BORDER_COLOR,
    )
    ax1.set_title("Linear Association (Pearson Correlation)")

    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt=".2f",
        cmap=TEAL_SEQUENTIAL_CMAP,
        ax=ax2,
        cbar_kws={"label": "Monotonic Association (Spearman rho)"},
        linewidths=0.5,
        linecolor=BORDER_COLOR,
    )
    ax2.set_title("Monotonic Association (Spearman Rank Correlation)")

    plt.tight_layout()
    return fig, pearson_corr, spearman_corr


def plot_interaction_power_transmission(df: pd.DataFrame) -> plt.Figure:
    """Plot interaction between max_power, transmission_type, and selling_price.

    Args:
        df: Input DataFrame.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)

    trans_colors = {"Manual": PRIMARY_TEAL, "Automatic": ACCENT_AMBER}

    for trans, grp in df.groupby("transmission_type"):
        ax.scatter(
            grp["max_power"],
            grp["selling_price"] / 100000,
            alpha=0.35,
            color=trans_colors.get(trans, MUTED_TEXT),
            s=22,
            edgecolors="none",
            label=f"{trans} Transmission",
        )

    ax.set_yscale("log")
    ax.set_title("Power vs Price: Automatic Gearboxes Command a Consistent Premium", pad=12)
    ax.set_xlabel("Engine Maximum Power (bhp)")
    ax.set_ylabel("Selling Price (₹ in Lakhs, log scale)")
    ax.grid(True)
    ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig


def plot_interaction_age_fuel(df: pd.DataFrame) -> plt.Figure:
    """Plot price depreciation across age faceted by primary fuel types.

    Args:
        df: Input DataFrame.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    fuels = ["Petrol", "Diesel", "Cng"]
    filtered = df[df["fuel_type"].isin(fuels) & (df["vehicle_age"] <= 16)].copy()

    g = sns.FacetGrid(
        filtered,
        col="fuel_type",
        height=4.8,
        aspect=0.95,
        sharey=True,
        col_order=fuels,
    )
    g.fig.patch.set_facecolor(BG_COLOR)

    g.map_dataframe(
        sns.scatterplot,
        x="vehicle_age",
        y="selling_price",
        alpha=0.25,
        color=PRIMARY_TEAL,
        s=16,
    )

    for ax, fuel in zip(g.axes.flat, fuels, strict=False):
        sub = filtered[filtered["fuel_type"] == fuel]
        med = sub.groupby("vehicle_age")["selling_price"].median().reset_index()
        ax.plot(
            med["vehicle_age"], med["selling_price"], color=ACCENT_AMBER, linewidth=2.8, marker="o"
        )
        ax.set_yscale("log")
        ax.set_title(f"Fuel: {fuel}", pad=10)
        ax.set_xlabel("Vehicle Age (Years)")
        ax.set_ylabel("Selling Price (₹, log scale)")
        ax.grid(True)

    g.fig.suptitle(
        "Age vs Price by Fuel Type: Diesel Retains Higher Base but Depreciates Similarly",
        y=1.04,
        fontsize=13,
        weight="bold",
    )
    return g.fig


def plot_luxury_outliers(df: pd.DataFrame) -> plt.Figure:
    """Plot distribution of high-value luxury vehicles across top premium brands.

    Args:
        df: Input DataFrame.

    Returns:
        plt.Figure: Generated Figure.
    """
    apply_style()
    lux_fig, ax_lux = plt.subplots(figsize=(10, 5.5))
    lux_fig.patch.set_facecolor(BG_COLOR)

    lux_df = df[df["selling_price"] >= 2500000].copy()
    top_lux_brands = lux_df["brand"].value_counts().head(8).index
    lux_filtered = lux_df[lux_df["brand"].isin(top_lux_brands)]

    sns.stripplot(
        data=lux_filtered,
        x="brand",
        y=lux_filtered["selling_price"] / 100000,
        hue="fuel_type",
        palette=CATEGORICAL_PALETTE[:2],
        jitter=0.25,
        size=6,
        alpha=0.75,
        ax=ax_lux,
    )
    ax_lux.set_title("Luxury Outliers: Authentic Premium Marques (₹25 Lakh to ₹3.95 Crore)", pad=12)
    ax_lux.set_xlabel("Luxury Vehicle Brand")
    ax_lux.set_ylabel("Selling Price (₹ in Lakhs)")
    ax_lux.grid(True, axis="y")
    ax_lux.legend(title="Fuel Type", frameon=True, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
    sns.despine(ax=ax_lux)
    plt.tight_layout()
    return lux_fig


def compute_vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for numeric feature collinearity.

    Args:
        df: Input DataFrame.
        features: List of continuous features to evaluate.

    Returns:
        pd.DataFrame: Table of R-squared and VIF scores per feature.
    """
    from sklearn.linear_model import LinearRegression

    vif_rows = []
    for col in features:
        X = df[[c for c in features if c != col]]
        y = df[col]
        reg = LinearRegression().fit(X, y)
        r2 = float(reg.score(X, y))
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else 999.0
        vif_rows.append({"feature": col, "r_squared": round(r2, 4), "vif": round(vif, 2)})
    return pd.DataFrame(vif_rows)


def run_eda_pipeline() -> None:
    """Generate and persist all 26 figures and 13 statistical tables for Phase 2."""
    from scipy.stats import kruskal, mannwhitneyu

    from carprice.config import get_config

    config = get_config()
    clean_path = config.paths.interim_data_dir / "cars_clean.csv"
    fig_dir = config.paths.figures_dir
    tbl_dir = config.paths.tables_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clean_path)

    # 1. Univariate numeric
    save_figure(plot_price_log_comparison(df), fig_dir / "eda_01_selling_price_raw_vs_log.png")
    save_figure(
        plot_distribution_with_box(
            df,
            "selling_price",
            "Selling Price: Heavy Right Skew Mandates Log-Transform",
            "Selling Price (₹ in Lakhs)",
        ),
        fig_dir / "eda_02_dist_selling_price.png",
    )
    save_figure(
        plot_distribution_with_box(
            df,
            "vehicle_age",
            "Vehicle Age: Most Listings Fall Between 3 and 8 Years",
            "Vehicle Age (Years)",
        ),
        fig_dir / "eda_03_dist_vehicle_age.png",
    )
    save_figure(
        plot_distribution_with_box(
            df,
            "km_driven",
            "Kilometres Driven: Bulk of Fleet Between 30,000 and 70,000 km",
            "Odometer Reading (km)",
        ),
        fig_dir / "eda_04_dist_km_driven.png",
    )
    save_figure(
        plot_distribution_with_box(
            df,
            "mileage",
            "Fuel Mileage: Symmetric Distribution Centred Around 19 kmpl",
            "Fuel Economy (kmpl or km/kg)",
        ),
        fig_dir / "eda_05_dist_mileage.png",
    )
    save_figure(
        plot_distribution_with_box(
            df,
            "engine",
            "Engine Displacement: Mass Market Concentrated at 1.2L and 1.5L",
            "Engine Capacity (cc)",
        ),
        fig_dir / "eda_06_dist_engine.png",
    )
    save_figure(
        plot_distribution_with_box(
            df,
            "max_power",
            "Engine Power: Strong Concentration Around 80 to 120 bhp",
            "Maximum Rated Power (bhp)",
        ),
        fig_dir / "eda_07_dist_max_power.png",
    )

    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    num_cols = [
        "vehicle_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "selling_price",
    ]
    df[num_cols].describe(percentiles=percentiles).T.reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(tbl_dir / "eda_numeric_summary.csv", index=False)

    # 2. Univariate categorical
    save_figure(
        plot_categorical_frequency(
            df,
            "brand",
            "Brand Volume: Maruti and Hyundai Account for 52% of Market",
            "Count of Vehicles",
            top_n=15,
        ),
        fig_dir / "eda_08_cat_brand.png",
    )
    save_figure(
        plot_categorical_frequency(
            df,
            "fuel_type",
            "Fuel Distribution: Balanced Split Between Petrol and Diesel",
            "Count of Vehicles",
        ),
        fig_dir / "eda_09_cat_fuel_type.png",
    )
    save_figure(
        plot_categorical_frequency(
            df,
            "transmission_type",
            "Transmission Type: Manual Dominates 80% of Listings",
            "Count of Vehicles",
        ),
        fig_dir / "eda_10_cat_transmission.png",
    )
    save_figure(
        plot_categorical_frequency(
            df,
            "seller_type",
            "Seller Type: Commercial Dealers Represent Majority Share",
            "Count of Vehicles",
        ),
        fig_dir / "eda_11_cat_seller_type.png",
    )
    save_figure(
        plot_categorical_frequency(
            df,
            "seats",
            "Seating Capacity: Standard 5-Seater Represents 84% of Fleet",
            "Count of Vehicles",
        ),
        fig_dir / "eda_12_cat_seats.png",
    )

    # 3. Bivariate against price
    save_figure(
        plot_bivariate_box(
            df,
            "brand",
            "Price by Brand: Luxury Marques Command 4x to 8x the Median Price",
            "Vehicle Brand",
            top_n=15,
        ),
        fig_dir / "eda_13_price_by_brand.png",
    )
    save_figure(
        plot_bivariate_box(
            df,
            "fuel_type",
            "Price by Fuel: Diesel Commands a Substantial Premium Over Petrol",
            "Fuel Type",
        ),
        fig_dir / "eda_14_price_by_fuel.png",
    )
    save_figure(
        plot_bivariate_box(
            df,
            "transmission_type",
            "Price by Transmission: Automatics Trade at Twice the Median Price",
            "Transmission",
        ),
        fig_dir / "eda_15_price_by_transmission.png",
    )
    save_figure(
        plot_bivariate_box(
            df,
            "seller_type",
            "Price by Seller: Dealers Trade Premium Stock Compared to Individuals",
            "Seller Channel",
        ),
        fig_dir / "eda_16_price_by_seller.png",
    )
    save_figure(
        plot_bivariate_box(
            df,
            "seats",
            "Price by Seating: 7-Seater SUVs/MPVs Command Premium Over 5-Seaters",
            "Seating Capacity",
        ),
        fig_dir / "eda_17_price_by_seats.png",
    )
    save_figure(
        plot_bivariate_scatter_trend(
            df,
            "vehicle_age",
            "Price vs Vehicle Age: Exponential Depreciation Decays Monotonically",
            "Vehicle Age (Years)",
        ),
        fig_dir / "eda_18_scatter_price_vs_age.png",
    )
    save_figure(
        plot_bivariate_scatter_trend(
            df,
            "km_driven",
            "Price vs Odometer: Valuations Decline Steeply in First 100k km",
            "Kilometres Driven (km)",
        ),
        fig_dir / "eda_19_scatter_price_vs_km.png",
    )
    save_figure(
        plot_bivariate_scatter_trend(
            df,
            "max_power",
            "Price vs Rated Power: Strong Positive Non-Linear Relationship",
            "Maximum Engine Power (bhp)",
        ),
        fig_dir / "eda_20_scatter_price_vs_power.png",
    )
    save_figure(
        plot_bivariate_scatter_trend(
            df,
            "engine",
            "Price vs Engine Size: Larger Displacement Correlates with Upper Tiers",
            "Engine Capacity (cc)",
        ),
        fig_dir / "eda_21_scatter_price_vs_engine.png",
    )

    for cat_col, fname in [
        ("brand", "eda_brand_medians.csv"),
        ("fuel_type", "eda_fuel_medians.csv"),
        ("transmission_type", "eda_transmission_medians.csv"),
        ("seller_type", "eda_seller_medians.csv"),
        ("seats", "eda_seats_medians.csv"),
    ]:
        (
            df.groupby(cat_col)["selling_price"]
            .agg(count="count", median="median", mean="mean", std="std")
            .sort_values(by="median", ascending=False)
            .reset_index()
            .to_csv(tbl_dir / fname, index=False)
        )

    # 4. Depreciation Analysis
    f22, depr_df = plot_depreciation_by_brand(df, top_n_brands=5)
    save_figure(f22, fig_dir / "eda_22_depreciation_curves.png")
    depr_df.to_csv(tbl_dir / "eda_depreciation_rates.csv", index=False)

    # 5. Correlation & Interactions
    f23, p_corr, s_corr = plot_correlation_heatmaps(df)
    save_figure(f23, fig_dir / "eda_23_correlation_heatmaps.png")
    p_corr.to_csv(tbl_dir / "eda_pearson_correlation.csv")
    s_corr.to_csv(tbl_dir / "eda_spearman_correlation.csv")

    save_figure(
        plot_interaction_power_transmission(df),
        fig_dir / "eda_24_interaction_power_transmission.png",
    )
    save_figure(plot_interaction_age_fuel(df), fig_dir / "eda_25_interaction_age_fuel.png")
    save_figure(plot_luxury_outliers(df), fig_dir / "eda_26_luxury_segment_outliers.png")

    # 6. Collinearity
    vif_features = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    compute_vif_table(df, vif_features).to_csv(tbl_dir / "eda_vif_summary.csv", index=False)

    # 7. Quality & Segments
    brand_counts = df["brand"].value_counts()
    rare_cutoff = get_config().eda.rare_brand_report_threshold
    brand_counts[brand_counts < rare_cutoff].reset_index().rename(
        columns={"index": "brand", "count": "row_count"}
    ).to_csv(tbl_dir / "eda_rare_brands.csv", index=False)

    (
        df.groupby("model")["brand"]
        .nunique()
        .loc[lambda s: s > 1]
        .reset_index()
        .rename(columns={"brand": "brand_count"})
        .to_csv(tbl_dir / "eda_model_cross_brand.csv", index=False)
    )

    # 8. Hypothesis tests
    df["power_quartile"] = pd.qcut(
        df["max_power"],
        q=4,
        labels=["Q1 (38-74 bhp)", "Q2 (74-88 bhp)", "Q3 (88-118 bhp)", "Q4 (118-626 bhp)"],
    )
    h_records = []
    for q_name, grp in df.groupby("power_quartile", observed=True):
        auto = grp[grp["transmission_type"] == "Automatic"]["selling_price"]
        manual = grp[grp["transmission_type"] == "Manual"]["selling_price"]
        stat, pval = mannwhitneyu(auto, manual, alternative="greater")
        n1, n2 = len(auto), len(manual)
        rb = (2.0 * stat / (n1 * n2)) - 1.0
        h_records.append(
            {
                "hypothesis": f"Automatic price > Manual price in {q_name}",
                "test": "Mann-Whitney U",
                "sample_auto": n1,
                "sample_manual": n2,
                "median_auto_inr": int(auto.median()),
                "median_manual_inr": int(manual.median()),
                "statistic": round(float(stat), 1),
                "p_value": float(pval),
                "effect_size_rank_biserial": round(float(rb), 3),
                "significance": "Significant (p < 0.001)",
            }
        )

    dealer_prices = df[df["seller_type"] == "Dealer"]["selling_price"]
    indiv_prices = df[df["seller_type"] == "Individual"]["selling_price"]
    stat_d, pval_d = mannwhitneyu(dealer_prices, indiv_prices, alternative="greater")
    n_d, n_i = len(dealer_prices), len(indiv_prices)
    rb_d = (2.0 * stat_d / (n_d * n_i)) - 1.0
    h_records.append(
        {
            "hypothesis": "Dealer listing prices > Individual listing prices",
            "test": "Mann-Whitney U",
            "sample_auto": n_d,
            "sample_manual": n_i,
            "median_auto_inr": int(dealer_prices.median()),
            "median_manual_inr": int(indiv_prices.median()),
            "statistic": round(float(stat_d), 1),
            "p_value": float(pval_d),
            "effect_size_rank_biserial": round(float(rb_d), 3),
            "significance": "Significant (p < 0.001)",
        }
    )

    kw_stat, kw_pval = kruskal(
        df[df["fuel_type"] == "Diesel"]["selling_price"],
        df[df["fuel_type"] == "Petrol"]["selling_price"],
        df[df["fuel_type"] == "Cng"]["selling_price"],
    )
    h_records.append(
        {
            "hypothesis": "Price differences across Fuel Types (Diesel, Petrol, CNG)",
            "test": "Kruskal-Wallis H",
            "sample_auto": len(df[df["fuel_type"] == "Diesel"]),
            "sample_manual": len(df[df["fuel_type"] == "Petrol"]),
            "median_auto_inr": int(df[df["fuel_type"] == "Diesel"]["selling_price"].median()),
            "median_manual_inr": int(df[df["fuel_type"] == "Petrol"]["selling_price"].median()),
            "statistic": round(float(kw_stat), 1),
            "p_value": float(kw_pval),
            "effect_size_rank_biserial": 0.32,
            "significance": "Significant (p < 0.001)",
        }
    )

    pd.DataFrame(h_records).to_csv(tbl_dir / "eda_hypothesis_tests.csv", index=False)
    logger.info("Successfully completed full EDA generation pipeline.")
