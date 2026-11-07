"""Unit tests for visualization helpers, palette configuration, and figure generators."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from carprice.viz import (
    apply_style,
    format_inr,
    plot_categorical_frequency,
    plot_correlation_heatmaps,
    plot_distribution_with_box,
    plot_price_log_comparison,
    save_figure,
)


@pytest.fixture
def sample_eda_df() -> pd.DataFrame:
    """Provide synthetic cleaned data for visualization testing."""
    return pd.DataFrame(
        {
            "brand": ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "BMW"],
            "model": ["Swift", "i20", "City", "Innova", "EcoSport", "3 Series"],
            "vehicle_age": [3, 5, 2, 7, 4, 6],
            "km_driven": [35000, 52000, 18000, 95000, 42000, 60000],
            "seller_type": [
                "Individual",
                "Dealer",
                "Individual",
                "Dealer",
                "Dealer",
                "Individual",
            ],
            "fuel_type": ["Petrol", "Diesel", "Petrol", "Diesel", "Diesel", "Petrol"],
            "transmission_type": [
                "Manual",
                "Manual",
                "Automatic",
                "Manual",
                "Manual",
                "Automatic",
            ],
            "mileage": [21.2, 19.8, 17.4, 12.8, 18.2, 14.5],
            "engine": [1197, 1396, 1498, 2393, 1498, 1998],
            "max_power": [82.0, 89.0, 118.0, 147.5, 99.0, 188.0],
            "seats": [5, 5, 5, 7, 5, 5],
            "selling_price": [520000, 480000, 780000, 1450000, 620000, 2400000],
            "km_was_capped": [False, False, False, False, False, False],
        }
    )


def test_format_inr() -> None:
    """Verify INR formatting utility with Lakhs, Crores, and standard rupee notation."""
    assert format_inr(45000) == "₹45,000"
    assert format_inr(550000) == "₹5.50 Lakh"
    assert format_inr(15000000) == "₹1.50 Cr"


def test_apply_style_sets_rcparams() -> None:
    """Verify that apply_style modifies matplotlib theme to match project palette."""
    apply_style()
    assert plt.rcParams["figure.facecolor"] == "#F7F5F0"
    assert plt.rcParams["axes.facecolor"] == "#FFFFFF"


def test_plot_distribution_with_box(sample_eda_df: pd.DataFrame) -> None:
    """Verify generation of two-panel box and histogram figure."""
    fig = plot_distribution_with_box(
        sample_eda_df,
        feature="km_driven",
        title="Odometer Distribution",
        x_label="Kilometres Driven",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_price_log_comparison(sample_eda_df: pd.DataFrame) -> None:
    """Verify raw vs log price side-by-side comparison figure."""
    fig = plot_price_log_comparison(sample_eda_df)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_categorical_frequency(sample_eda_df: pd.DataFrame) -> None:
    """Verify horizontal bar chart generation for categorical columns."""
    fig = plot_categorical_frequency(
        sample_eda_df,
        column="fuel_type",
        title="Fuel Types Count",
        x_label="Number of Cars",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmaps(sample_eda_df: pd.DataFrame) -> None:
    """Verify generation of correlation matrices and twin heatmaps."""
    fig, pearson, spearman = plot_correlation_heatmaps(sample_eda_df)
    assert isinstance(fig, plt.Figure)
    assert "selling_price" in pearson.columns
    assert "log_selling_price" in spearman.columns
    plt.close(fig)


def test_save_figure_creates_png(sample_eda_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify that save_figure outputs a valid image file."""
    fig = plot_price_log_comparison(sample_eda_df)
    out_file = tmp_path / "test_eda_price.png"
    saved = save_figure(fig, out_file)
    assert saved.exists()
    assert saved.stat().st_size > 0
