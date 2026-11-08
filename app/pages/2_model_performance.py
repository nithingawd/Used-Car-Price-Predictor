"""Page 3: Model performance. Every number is read from reports/tables/ (7.4.5)."""

import pandas as pd
import streamlit as st

from carprice.config import get_config
from carprice.insights import FACTOR_LABELS
import components  # noqa: F401  (puts src/ on the import path)
from components.charts import error_histogram, predicted_vs_actual, segment_chart
from components.data import model_card, report_figure, report_table
from components.palette import TEAL_TINT
from components.ui import (
    chart_card,
    format_inr,
    info_note,
    metric_card,
    page_header,
    panel,
    safely,
    section_header,
)

SEGMENTATIONS = {
    "Price band": "Price band (train quintiles)",
    "Brand": "Brand (top 10)",
    "Fuel": "Fuel type",
    "Vehicle age": "Vehicle age band",
}


def factor_label(feature: str) -> str:
    """Readable name for a model input, sentence case."""
    return FACTOR_LABELS.get(feature, feature.replace("_", " ").capitalize())


def headline_cards(final: pd.Series, coverage: pd.Series) -> None:
    """Four headline metrics with plain-English captions (7.4.1)."""
    cells = st.columns(4)
    cards = [
        (
            "Average error",
            format_inr(final["mae_inr"]),
            "Typical miss on a car the model never saw.",
        ),
        (
            "Within 10%",
            f"{final['within_10pct']:.1f}%",
            f"Cars priced within 10% ({final['within_20pct']:.1f}% within 20%).",
        ),
        (
            "R² (log price)",
            f"{final['r2_log']:.3f}",
            "Share of price variation explained; 1 is perfect.",
        ),
        (
            "RMSE",
            format_inr(final["rmse_inr"]),
            "Weighs big misses more, so costly cars raise it.",
        ),
    ]
    for cell, (label, value, caption) in zip(cells, cards, strict=True):
        with cell:
            metric_card(label, value, caption)
    info_note(
        f"The 80% price range contained the true price for {coverage['coverage_pct']:.1f}% of "
        f"{int(coverage['n_test']):,} held-out cars: {coverage['verdict']}."
    )


def comparison_table() -> None:
    """Untuned 5-fold CV benchmark with the chosen algorithm highlighted (7.4.2)."""
    cv = report_table("cv_results")
    card = report_table("train_vs_test")
    chosen = model_card()["model_name"].split(" ")[0]
    shown = cv[["model_id", "log_rmse_mean", "log_rmse_std", "r2_mean", "inr_mae"]].rename(
        columns={
            "model_id": "Model",
            "log_rmse_mean": "CV log RMSE",
            "log_rmse_std": "± std",
            "r2_mean": "CV R²",
            "inr_mae": "CV MAE (₹)",
        }
    )
    styled = shown.style.apply(
        lambda row: [
            f"background-color: {TEAL_TINT}; font-weight: 600" if row["Model"] == chosen else ""
            for _ in row
        ],
        axis=1,
    ).format(
        {"CV log RMSE": "{:.4f}", "± std": "{:.4f}", "CV R²": "{:.4f}", "CV MAE (₹)": format_inr}
    )
    with panel("benchmark"):
        section_header("Ten algorithms compared before tuning (5-fold CV on training data)")
        st.dataframe(styled, hide_index=True, width="stretch")
        test = card[card["split"].str.startswith("test")].iloc[0]
        cv_row = card[card["split"].str.contains("out-of-fold")].iloc[0]
        st.html(
            '<p class="cp-caption">The highlighted algorithm was tuned and chosen. Tuned, it '
            "scored "
            f"{cv_row['rmse_log']:.4f} log RMSE in cross-validation and {test['rmse_log']:.4f} on "
            "the "
            "held-out test set, so the CV estimate held up.</p>"
        )


def accuracy_charts(predictions: pd.DataFrame, final: pd.Series) -> None:
    """Interactive predicted vs actual and the error distribution."""
    left, right = st.columns([3, 2], gap="medium")
    with left:
        chart_card(
            predicted_vs_actual(predictions),
            "Predictions track actual prices across the range",
            "Each dot is a held-out test car; hover to see which one. Dots on the dashed line are "
            "priced exactly; both axes are logarithmic, so equal distances mean equal % errors.",
        )
    with right:
        chart_card(
            error_histogram(predictions),
            f"{final['within_10pct']:.1f}% of cars land inside the shaded ±10% band",
            f"Median absolute error is {final['mdape_pct']:.2f}%; errors beyond ±60% are drawn "
            "in the end bins.",
        )


def segment_section() -> None:
    """Accuracy by segment with a note on where the model is weaker (7.4.3)."""
    segments = report_table("segment_errors")
    min_rows = get_config().evaluation.min_segment_rows
    choice = st.segmented_control(
        "Segment by", list(SEGMENTATIONS), default="Price band", key="segment_by"
    )
    table = segments[segments["segmentation"] == SEGMENTATIONS[choice or "Price band"]].reset_index(
        drop=True
    )
    eligible = table[table["n_test"] >= min_rows]
    weakest = eligible.loc[eligible["within_10pct"].idxmin()]
    best = eligible.loc[eligible["within_10pct"].idxmax()]
    chart_card(
        segment_chart(table, min_rows),
        f"Weakest: {weakest['segment']} ({weakest['within_10pct']:.1f}% within 10%)",
        f"Strongest is {best['segment']} at {best['within_10pct']:.1f}%. Overall the model is "
        "weaker "
        f"for old cars, luxury cars and the cheapest cars; grey bars have fewer than {min_rows} "
        "test "
        "cars and are not ranked.",
    )


def drivers_section() -> None:
    """SHAP summary image and a plain-language paragraph from shap_importance.csv (7.4.4)."""
    shap_table = report_table("shap_importance")
    top = shap_table.head(4)
    left, right = st.columns([3, 2], gap="medium")
    with left, panel("drivers-text"):
        section_header(f"{factor_label(top.iloc[0]['feature'])} moves predicted price the most")
        st.image(str(report_figure("shap_01_summary")), width="stretch")
    with right, panel("drivers-figure"):
        section_header("What drives the price")
        shares = ", ".join(
            f"{factor_label(r.feature).lower()} ({r.share_pct:.1f}%)" for r in top.itertuples()
        )
        st.html(
            '<p class="cp-caption">The model prices a car mainly by which model it is, how old '
            "it is and how powerful it is. Their shares of the total SHAP effect are "
            f"{shares}. Older cars are priced lower and more powerful ones higher, matching the "
            "market analysis. Fuel, transmission and seller type adjust the price far less once "
            "the car model and power are known.</p>"
        )


def performance() -> bool:
    """Assemble the page from saved tables; True when it rendered."""
    metrics = report_table("test_metrics").set_index("model_id")
    final = metrics.loc["final"]
    page_header(
        "Model performance",
        f"How accurate the model was on {int(final['n_test']):,} cars it "
        "never saw in training, and where it is weaker.",
    )
    headline_cards(final, report_table("interval_coverage").iloc[0])
    accuracy_charts(
        report_table("test_predictions").rename(
            columns={"pred_final": "predicted_inr", "signed_pct_error": "pct_error"}
        ),
        final,
    )
    segment_section()
    comparison_table()
    drivers_section()
    return True


if safely(performance, "Model evaluation results") is None:
    page_header("Model performance", "Accuracy of the model on held-out cars.")
