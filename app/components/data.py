"""Cached access to the model and data files (7.1.3).

st.cache_resource holds the PricePredictor (loaded once per server process, never on
interaction); st.cache_data holds DataFrames.
"""

from importlib import metadata
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from carprice.config import get_config
from carprice.predict import PricePredictor

LIBRARIES = ["scikit-learn", "xgboost", "pandas", "numpy", "streamlit", "plotly", "shap"]


@st.cache_resource(show_spinner="Loading the pricing model...")
def get_predictor() -> PricePredictor:
    """The validated predictor (pipeline, model card and input schema)."""
    return PricePredictor()


@st.cache_data(show_spinner=False)
def model_specs() -> pd.DataFrame:
    """models/model_specs.csv: default specs per brand and model (from train.csv)."""
    return read_csv(get_config().paths.models_dir / "model_specs.csv")


@st.cache_data(show_spinner=False)
def training_listings() -> pd.DataFrame:
    """The training partition, used for 'similar cars' and comparison panels."""
    return read_csv(get_config().paths.processed_data_dir / "train.csv")


@st.cache_data(show_spinner="Loading market data...")
def market_listings() -> pd.DataFrame:
    """Every cleaned listing (Market Explorer only)."""
    return read_csv(get_config().paths.interim_data_dir / "cars_clean.csv")


@st.cache_data(show_spinner=False)
def report_table(name: str) -> pd.DataFrame:
    """reports/tables/<name>.csv."""
    return read_csv(get_config().paths.tables_dir / f"{name}.csv")


@st.cache_data(show_spinner=False)
def model_card() -> dict[str, Any]:
    """models/model_card.json written in Phase 4."""
    path = get_config().paths.models_dir / "model_card.json"
    if not path.is_file():
        raise FileNotFoundError("model_card.json is missing. Run: python tasks.py train")
    return json.loads(path.read_text(encoding="utf-8"))


def report_figure(name: str) -> Path:
    """Path to reports/figures/<name>.png.

    Raises:
        FileNotFoundError: If the figure has not been generated.
    """
    path = get_config().paths.figures_dir / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing. Run: python tasks.py evaluate")
    return path


@st.cache_data(show_spinner=False)
def library_versions() -> dict[str, str]:
    """Installed versions of the libraries the model and app depend on."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV or fail with a message naming the command that creates it."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing. Run the pipeline commands in README.md")
    return pd.read_csv(path)
