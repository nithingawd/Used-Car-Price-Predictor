"""Phase 5 figures: test-set diagnostics (eval_*.png) and explainability (shap_*.png).

Every title states the finding and is built from the saved tables, so no figure quotes a
number that is absent from reports/tables/.
"""

import logging
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from carprice.config import get_config
from carprice.features import CATEGORICAL_FEATURES
from carprice.viz import (
    ACCENT_AMBER,
    CATEGORICAL_PALETTE,
    INK_COLOR,
    MUTED_TEXT,
    PRIMARY_TEAL,
    apply_style,
    format_inr_indian,
    save_figure,
)

logger = logging.getLogger(__name__)

SLATE = CATEGORICAL_PALETTE[5]
LAKH = 100000.0
# Low-to-high feature value colouring for SHAP points: slate -> teal (no rainbow, A7).
SHAP_VALUE_CMAP = LinearSegmentedColormap.from_list("slate_teal", [SLATE, PRIMARY_TEAL])
PCT_ERROR_DISPLAY_LIMIT = 60
RESIDUAL_TREND_BINS = 20


def figure_path(name: str) -> Path:
    """Resolve reports/figures/<name>.png."""
    return get_config().paths.figures_dir / f"{name}.png"


def add_caption(fig: plt.Figure, text: str) -> None:
    """Muted text summary under a chart (A7: every chart has a text summary)."""
    fig.text(0.01, -0.02, text, fontsize=9, color=MUTED_TEXT, ha="left", va="top", wrap=True)


def percent_effect(shap_log: np.ndarray) -> np.ndarray:
    """Convert a log-price SHAP contribution to a % change in predicted price."""
    return np.expm1(shap_log) * 100


# ---------------------------------------------------------------------------
# Task 5.2 figures
# ---------------------------------------------------------------------------


def plot_predicted_vs_actual(pred_df: pd.DataFrame, coverage: pd.Series) -> plt.Figure:
    """Log-log scatter of actual vs predicted price with y = x and the 80% interval band."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    pred, actual = pred_df["pred_final"] / LAKH, pred_df["actual_inr"] / LAKH
    order = np.argsort(pred.to_numpy())
    ax.fill_between(
        pred.to_numpy()[order],
        (pred_df["interval_low"] / LAKH).to_numpy()[order],
        (pred_df["interval_high"] / LAKH).to_numpy()[order],
        color=PRIMARY_TEAL,
        alpha=0.15,
        label="80% prediction interval",
        linewidth=0,
    )
    ax.scatter(pred, actual, s=8, alpha=0.35, color=PRIMARY_TEAL, label="Test car")
    limits = [min(pred.min(), actual.min()) * 0.9, max(pred.max(), actual.max()) * 1.1]
    ax.plot(limits, limits, color=INK_COLOR, linestyle="--", linewidth=1.2, label="Perfect (y = x)")
    ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
    ax.set_xlabel("Predicted price (₹ lakh, log scale)")
    ax.set_ylabel("Actual price (₹ lakh, log scale)")
    ax.set_title(
        f"{coverage['coverage_pct']:.2f}% of test cars fall inside the 80% interval "
        f"({coverage['verdict']})"
    )
    ax.grid(True, which="major", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    add_caption(
        fig,
        "Each dot is one held-out car. Dots on the dashed line are priced exactly; "
        "the shaded band is where 80% of cars should fall.",
    )
    return fig


def plot_residuals_vs_predicted(pred_df: pd.DataFrame, residual_stats: pd.Series) -> plt.Figure:
    """Log residuals against predicted price with a binned-median trend line."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pred = pred_df["pred_final"] / LAKH
    ax.scatter(pred, pred_df["log_residual"], s=8, alpha=0.3, color=PRIMARY_TEAL)
    bins = pd.qcut(pred, RESIDUAL_TREND_BINS, duplicates="drop")
    trend = pred_df.groupby(bins, observed=True)["log_residual"].median()
    centres = pred.groupby(bins, observed=True).median()
    ax.plot(
        centres, trend, color=ACCENT_AMBER, linewidth=2.2, marker="o", label="Median by price bin"
    )
    ax.axhline(0, color=INK_COLOR, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Predicted price (₹ lakh, log scale)")
    ax.set_ylabel("Residual: log(actual) - log(predicted)")
    ax.set_title(
        f"Errors centre on zero (median log residual {residual_stats['median_log_residual']:+.4f})"
    )
    ax.legend(frameon=False)
    add_caption(
        fig,
        "Points above zero are cars the model under-priced; below zero, over-priced. "
        "A flat median line means no systematic bias across the price range.",
    )
    return fig


def plot_residual_qq(pred_df: pd.DataFrame, residual_stats: pd.Series) -> plt.Figure:
    """Normal QQ plot of log residuals."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(
        pred_df["log_residual"], dist="norm"
    )
    ax.scatter(theoretical, ordered, s=8, color=PRIMARY_TEAL, alpha=0.5)
    ax.plot(theoretical, slope * theoretical + intercept, color=INK_COLOR, linestyle="--")
    heavy = residual_stats["excess_kurtosis"] > 1
    ax.set_title(
        "Residual tails are heavier than a normal distribution"
        if heavy
        else "Residuals are close to a normal distribution"
    )
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Observed log residual quantile")
    add_caption(
        fig,
        f"Excess kurtosis {residual_stats['excess_kurtosis']:.3f}; "
        f"{residual_stats['share_beyond_2sd_pct']:.2f}% of residuals lie beyond 2 SD "
        f"versus {residual_stats['normal_share_beyond_2sd_pct']:.2f}% for a normal curve.",
    )
    return fig


def plot_pct_error_hist(pred_df: pd.DataFrame, final_metrics: pd.Series) -> plt.Figure:
    """Histogram of signed percentage error with the ±10% and ±20% bands marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    limit = PCT_ERROR_DISPLAY_LIMIT
    errors = pred_df["signed_pct_error"].clip(-limit, limit)
    ax.hist(errors, bins=np.arange(-limit, limit + 2, 2), color=PRIMARY_TEAL, edgecolor="white")
    ax.axvspan(-10, 10, color=ACCENT_AMBER, alpha=0.15, label="Within ±10%")
    for edge in (-20, 20):
        ax.axvline(edge, color=INK_COLOR, linestyle=":", linewidth=1)
    ax.set_xlabel("Percentage error: (predicted - actual) / actual (%)")
    ax.set_ylabel("Number of test cars")
    ax.set_title(
        f"{final_metrics['within_10pct']:.2f}% of cars are priced within 10%, "
        f"{final_metrics['within_20pct']:.2f}% within 20%"
    )
    ax.legend(frameon=False)
    add_caption(fig, f"Errors beyond ±{limit}% are drawn in the end bins. Dotted lines mark ±20%.")
    return fig


def segment_bar_colours(table: pd.DataFrame) -> list[str]:
    """Amber for the flagged worst segments, slate for segments too small to rank, else teal."""
    colours = []
    for _, row in table.iterrows():
        if pd.notna(row["worst_rank"]):
            colours.append(ACCENT_AMBER)
        elif not row["eligible_for_ranking"]:
            colours.append(SLATE)
        else:
            colours.append(PRIMARY_TEAL)
    return colours


def plot_segment_within10(segment_table: pd.DataFrame) -> plt.Figure:
    """Within-10% share for every segment; worst five highlighted (D-023 ranking)."""
    apply_style()
    table = segment_table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(table) + 1.5))
    labels = [f"{r.segmentation}: {r.segment}  (n={r.n_test})" for r in table.itertuples()]
    ax.barh(labels, table["within_10pct"], color=segment_bar_colours(table), height=0.7)
    for i, row in table.iterrows():
        marker = f"  #{int(row['worst_rank'])} worst" if pd.notna(row["worst_rank"]) else ""
        ax.text(
            row["within_10pct"] + 0.5,
            i,
            f"{row['within_10pct']:.2f}%{marker}",
            va="center",
            fontsize=8,
            color=INK_COLOR,
        )
    worst = segment_table.dropna(subset=["worst_rank"]).sort_values("worst_rank").iloc[0]
    ax.set_title(
        f"Weakest segment: {worst['segmentation']} = {worst['segment']} "
        f"({worst['within_10pct']:.2f}% within 10%)"
    )
    ax.set_xlabel("Share of test cars priced within 10% (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    add_caption(
        fig,
        "Amber: five worst segments with at least "
        f"{get_config().evaluation.min_segment_rows} test cars. Grey: too few cars to rank.",
    )
    return fig


def plot_model_comparison(test_metrics: pd.DataFrame) -> plt.Figure:
    """Headline metrics (D-023) for the final model and both baselines."""
    apply_style()
    fig, (ax_mae, ax_w10) = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = [ACCENT_AMBER if m == "final" else SLATE for m in test_metrics["model_id"]]
    ax_mae.barh(test_metrics["model"], test_metrics["mae_inr"] / LAKH, color=colours)
    ax_w10.barh(test_metrics["model"], test_metrics["within_10pct"], color=colours)
    for ax, column, fmt in ((ax_mae, "mae_inr", "inr"), (ax_w10, "within_10pct", "pct")):
        for i, value in enumerate(test_metrics[column]):
            label = format_inr_indian(value) if fmt == "inr" else f"{value:.2f}%"
            ax.text(
                (value / LAKH if fmt == "inr" else value), i, f"  {label}", va="center", fontsize=9
            )
        ax.invert_yaxis()
    ax_mae.set_xlabel("Mean absolute error (₹ lakh) - lower is better")
    ax_w10.set_xlabel("Cars priced within 10% (%) - higher is better")
    ax_w10.set_yticklabels([])
    final = test_metrics.set_index("model_id").loc["final"]
    fig.suptitle(
        f"Final model: {format_inr_indian(final['mae_inr'])} average error, "
        f"{final['within_10pct']:.2f}% of cars within 10%",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    ax_mae.set_xlim(0, test_metrics["mae_inr"].max() / LAKH * 1.45)
    ax_w10.set_xlim(0, 100)
    return fig


def plot_test_figures(segmented: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[Path]:
    """Render and save every task 5.2 figure."""
    final = tables["test_metrics"].set_index("model_id").loc["final"]
    coverage = tables["interval_coverage"].iloc[0]
    residuals = tables["residual_stats"].iloc[0]
    figures = {
        "eval_01_predicted_vs_actual": plot_predicted_vs_actual(segmented, coverage),
        "eval_02_residuals_vs_predicted": plot_residuals_vs_predicted(segmented, residuals),
        "eval_03_residual_qq": plot_residual_qq(segmented, residuals),
        "eval_04_pct_error_hist": plot_pct_error_hist(segmented, final),
        "eval_05_segment_within10": plot_segment_within10(tables["segment_errors"]),
        "eval_06_model_comparison": plot_model_comparison(tables["test_metrics"]),
    }
    return [save_figure(fig, figure_path(name)) for name, fig in figures.items()]


# ---------------------------------------------------------------------------
# Task 5.3 figures
# ---------------------------------------------------------------------------


def plot_permutation_importance(table: pd.DataFrame) -> plt.Figure:
    """Increase in test log RMSE when each input column is shuffled."""
    apply_style()
    ordered = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = [SLATE if v <= 0 else PRIMARY_TEAL for v in ordered["rmse_log_increase"]]
    ax.barh(
        ordered["feature"],
        ordered["rmse_log_increase"],
        xerr=ordered["rmse_log_increase_std"],
        color=colours,
        error_kw={"ecolor": INK_COLOR, "elinewidth": 1},
    )
    ax.set_xlabel("Increase in test log RMSE when the column is shuffled")
    ax.set_title(f"Shuffling {table.iloc[0]['feature']} hurts accuracy the most")
    ax.grid(axis="x", alpha=0.5)
    add_caption(
        fig,
        "Error bars: standard deviation over repeats. Grey bars at zero are columns "
        "the model does not use or cannot exploit.",
    )
    return fig


def plot_shap_summary(
    shap_values: np.ndarray, transformed: pd.DataFrame, top_feature: str
) -> plt.Figure:
    """SHAP beeswarm over the estimator's input columns."""
    import shap

    apply_style()
    shap.summary_plot(
        shap_values,
        transformed,
        max_display=get_config().evaluation.shap_summary_max_features,
        show=False,
        cmap=SHAP_VALUE_CMAP,
        plot_size=(9, 7),
        rng=np.random.default_rng(get_config().project.random_seed),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (effect on log predicted price)")
    fig.axes[0].set_title(f"{top_feature} moves predicted price the most", loc="left")
    add_caption(
        fig,
        "Each dot is one test car. Right of zero raises the predicted price; colour "
        "shows the feature's own value (slate = low, teal = high).",
    )
    return fig


def dependence_direction(x: pd.Series, effect: np.ndarray) -> str:
    """'raises' or 'lowers', from the Spearman sign between a feature and its SHAP effect."""
    rho = stats.spearmanr(x, effect, nan_policy="omit").statistic
    return "raises" if rho > 0 else "lowers"


def plot_numeric_dependence(
    feature: str, x: pd.Series, effect: np.ndarray, x_label: str
) -> plt.Figure:
    """Scatter of a numeric feature against its SHAP effect (% of predicted price)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, effect, s=8, alpha=0.35, color=PRIMARY_TEAL)
    ax.axhline(0, color=INK_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Effect on predicted price (%)")
    ax.set_title(f"Higher {feature} {dependence_direction(x, effect)} the predicted price")
    add_caption(
        fig,
        "Each dot is one test car; the vertical spread at one x value comes from "
        "interactions with other features.",
    )
    return fig


def plot_categorical_dependence(feature: str, values: pd.Series, effect: np.ndarray) -> plt.Figure:
    """Median SHAP effect per category with interquartile range."""
    apply_style()
    frame = pd.DataFrame({"category": values.astype(str).to_numpy(), "effect": effect})
    summary = frame.groupby("category")["effect"].quantile([0.25, 0.5, 0.75]).unstack()
    summary = summary.sort_values(0.5)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(summary) + 1.5))
    ax.hlines(summary.index, summary[0.25], summary[0.75], color=SLATE, linewidth=4)
    ax.scatter(summary[0.5], summary.index, color=PRIMARY_TEAL, zorder=3)
    ax.axvline(0, color=INK_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Effect on predicted price (%) - dot = median, bar = middle 50%")
    ax.set_title(
        f"{summary.index[-1]} lifts and {summary.index[0]} lowers the price most ({feature})"
    )
    return fig


def plot_shap_dependence(
    feature: str, grouped: pd.DataFrame, sample: pd.DataFrame, transformed: pd.DataFrame
) -> plt.Figure:
    """Dependence plot for one input feature, numeric or categorical."""
    effect = percent_effect(grouped[feature].to_numpy())
    if feature in CATEGORICAL_FEATURES:
        return plot_categorical_dependence(feature, sample[feature], effect)
    if feature == "model":
        x, label = transformed["model"], "model (target-encoded: mean log price of that car model)"
    elif feature in sample.columns:
        x, label = sample[feature], feature
    else:
        x, label = transformed[feature], f"{feature} (engineered)"
    return plot_numeric_dependence(feature, x.reset_index(drop=True), effect, label)


def plot_explainability_figures(
    shap_values: np.ndarray,
    transformed: pd.DataFrame,
    grouped: pd.DataFrame,
    sample: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> list[Path]:
    """Render and save the permutation, SHAP summary and top-N dependence figures."""
    top_features = tables["shap_importance"]["feature"].tolist()
    n_dependence = get_config().evaluation.shap_dependence_features
    paths = [
        save_figure(
            plot_permutation_importance(tables["permutation_importance"]),
            figure_path("eval_07_permutation_importance"),
        ),
        save_figure(
            plot_shap_summary(shap_values, transformed, top_features[0]),
            figure_path("shap_01_summary"),
        ),
    ]
    for i, feature in enumerate(top_features[:n_dependence], start=2):
        fig = plot_shap_dependence(
            feature, grouped, sample.reset_index(drop=True), transformed.reset_index(drop=True)
        )
        paths.append(save_figure(fig, figure_path(f"shap_{i:02d}_dependence_{feature}")))
    return paths
