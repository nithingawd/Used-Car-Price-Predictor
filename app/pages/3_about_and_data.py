"""Page 4: About and data. Dataset, cleaning, how the model works, limits, model card (7.5)."""

import pandas as pd
import streamlit as st

from carprice.predict import FIELD_LABELS
import components  # noqa: F401  (puts src/ on the import path)
from components.data import library_versions, model_card, report_table
from components.ui import info_note, metric_card, page_header, panel, safely, section_header

RULE_NAMES = {
    "drop_identifier_columns": "Drop ID columns",
    "normalise_text": "Tidy text labels",
    "drop_exact_duplicates": "Remove exact duplicates",
    "fix_invalid_seats": "Fix impossible seat counts",
    "fix_fuel_labels": "Correct mislabelled fuel",
    "treat_km_outliers": "Cap extreme odometer readings",
    "validate_ranges": "Check physical ranges",
}


def dataset_section(log: pd.DataFrame) -> None:
    """What the data is, with row counts from the cleaning log."""
    raw_rows, clean_rows = int(log.iloc[0]["rows_before"]), int(log.iloc[-1]["rows_after"])
    cells = st.columns(3)
    with cells[0]:
        metric_card(
            "Raw listings", f"{raw_rows:,}", "Used cars listed on CarDekho, one row per listing."
        )
    with cells[1]:
        metric_card(
            "After cleaning", f"{clean_rows:,}", "Exact duplicates removed; every other row kept."
        )
    with cells[2]:
        metric_card(
            "Details per car",
            f"{len(FIELD_LABELS):,}",
            "Brand, model, age, km, seller, fuel, gearbox, engine, power, mileage and seats.",
        )


def cleaning_section(log: pd.DataFrame) -> None:
    """Compact cleaning summary from cleaning_log.csv."""
    table = pd.DataFrame(
        {
            "Step": log["rule_name"].map(RULE_NAMES).fillna(log["rule_name"]),
            "Rows affected": log["rows_affected"],
            "Rows after": log["rows_after"],
            "What it did": log["details"],
        }
    )
    with panel("cleaning"):
        section_header("How the data was cleaned")
        st.table(
            table.style.format({"Rows affected": "{:,}", "Rows after": "{:,}"}).hide(axis="index")
        )
        st.html(
            '<p class="cp-caption">The raw file is never modified; each step produces a new table, '
            "and every choice is justified in the project's decision log.</p>"
        )


def how_it_works(card: dict) -> None:
    """Four sentences on the model (7.5)."""
    cv = card["cross_validation"]
    with panel("how-it-works"):
        section_header("How the model works")
        st.html(
            '<p class="cp-caption">The model learns from past listings how price depends on a '
            "car's "
            "brand, model, age, kilometres, fuel, gearbox, seller and engine figures. It is a "
            "tuned "
            f"gradient-boosted tree ensemble ({card['model_name']}) that predicts the logarithm of "
            "price, so its errors are judged in proportion to the price rather than in rupees. "
            f"It was trained on {card['training_sample_count']:,} listings and checked "
            f"with {cv['n_splits']}-fold cross-validation before a single test on unseen cars. "
            "Its 80% "
            "price range comes from how far its own cross-validated predictions missed.</p>"
        )


def limitations() -> None:
    """Limitations box with figures from segment_errors.csv."""
    segments = report_table("segment_errors").set_index(["segmentation", "segment"])
    old = segments.loc[("Vehicle age band", "12+ yrs")]
    lux = segments.loc[("Luxury band", "Above ₹20 lakh")]
    info_note(
        "Use estimates as a first opinion, not a valuation.",
        kind="info",
        title="Limitations.",
        items=[
            f"Old cars (12+ years) are priced within 10% only {old['within_10pct']:.1f}% of the "
            "time: "
            "condition, which the data does not record, matters most for them.",
            f"Cars above ₹20 lakh are priced within 10% {lux['within_10pct']:.1f}% of the time; "
            "trim and "
            "variant, which the data lacks, move their prices a lot.",
            "Rare models and fuels rest on few examples; the estimator flags them.",
            "Prices are listing prices from one platform and period, not final sale prices, and "
            "do not "
            "reflect city, accident history, number of owners or colour.",
        ],
    )


def model_card_section(card: dict) -> None:
    """Version, training date, commit and library versions."""
    rows = [
        ("Model", card["model_name"]),
        ("Trained (UTC)", card["timestamp_utc"][:19].replace("T", " ")),
        ("Code version (git)", card["git_commit"]),
        ("Training rows", f"{card['training_sample_count']:,}"),
        ("Target", card["target_transform"]),
    ]
    rows += [(f"Library: {name}", version) for name, version in library_versions().items()]
    with panel("model-card"):
        section_header("Model card")
        st.table(pd.DataFrame(rows, columns=["Item", "Value"]).style.hide(axis="index"))


def about() -> bool:
    """Assemble the page."""
    log, card = report_table("cleaning_log"), model_card()
    dataset_section(log)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        cleaning_section(log)
        how_it_works(card)
    with right:
        limitations()
        model_card_section(card)
    return True


page_header(
    "About and data",
    "Where the data comes from, how it was prepared, how the model works and what it cannot do.",
)
safely(about, "Project information")
