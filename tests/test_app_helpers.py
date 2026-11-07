"""GUI helper tests: formatters, spec lookup, upload checks, market summaries, AppTest smoke."""

from pathlib import Path
import sys
import time

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from carprice.config import get_config  # noqa: E402
from carprice.insights import (  # noqa: E402
    build_model_specs,
    default_spec,
    explain_car,
    similar_listing_range,
)
from carprice.market import (  # noqa: E402
    MarketFilter,
    annual_depreciation_pct,
    apply_filter,
    brand_ranking,
    median_by_age,
    peer_median,
)
from carprice.predict import CarSpec, PricePredictor  # noqa: E402
from components.batch import check_upload  # noqa: E402
from components.ui import format_inr, format_lakh  # noqa: E402

HOME = str(APP_DIR / "Home.py")
APP_TIMEOUT = 180

# ---------------------------------------------------------------------------
# Formatters (A7 number formats)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(550000, "₹ 5,50,000"), (1234567, "₹ 12,34,567"), (950, "₹ 950"), (39500000, "₹ 3,95,00,000")],
)
def test_format_inr_indian_grouping(value: float, expected: str) -> None:
    """Indian digit grouping: 550000 -> ₹ 5,50,000."""
    assert format_inr(value) == expected


def test_format_lakh() -> None:
    """Lakh with two decimals."""
    assert format_lakh(550000) == "₹ 5.50 lakh"


# ---------------------------------------------------------------------------
# Model spec lookup (pre-fill)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def specs() -> pd.DataFrame:
    """The saved spec lookup."""
    return pd.read_csv(get_config().paths.models_dir / "model_specs.csv")


def test_every_brand_model_pair_has_specs(specs: pd.DataFrame) -> None:
    """The lookup covers exactly the brand/model pairs in the input schema."""
    schema = PricePredictor().schema
    pairs = {(b, m) for b, models in schema["brands"].items() for m in models}
    assert set(zip(specs["brand"], specs["model"], strict=True)) == pairs


def test_default_spec_values_are_valid(specs: pd.DataFrame) -> None:
    """Pre-filled values pass the predictor's validation for every model."""
    predictor = PricePredictor()
    for row in specs.head(40).itertuples():
        spec = default_spec(specs, row.brand, row.model)
        car = {**spec, "vehicle_age": 5, "km_driven": 50000, "seller_type": "Dealer"}
        assert predictor.check(car).model == row.model


def test_spec_lookup_matches_training_mode(specs: pd.DataFrame) -> None:
    """The saved file equals a fresh build from train.csv."""
    train = pd.read_csv(get_config().paths.processed_data_dir / "train.csv")
    pd.testing.assert_frame_equal(specs, build_model_specs(train), check_dtype=False)


# ---------------------------------------------------------------------------
# Batch upload checks
# ---------------------------------------------------------------------------


def template_bytes(rows: int = 2) -> bytes:
    """A valid upload with the given number of rows."""
    car = {
        "brand": "Maruti",
        "model": "Swift",
        "vehicle_age": 5,
        "km_driven": 60000,
        "seller_type": "Dealer",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "mileage": 21.5,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5,
    }
    return pd.DataFrame([car] * rows).to_csv(index=False).encode("utf-8")


def test_upload_valid_file() -> None:
    """A well-formed file passes with exactly the input columns."""
    frame, problem = check_upload(template_bytes())
    assert problem is None and len(frame) == 2


@pytest.mark.parametrize(
    ("content", "title"),
    [
        (b"", "Unreadable file."),
        (b"\xff\xfe\x00bad", "Unreadable file."),
        (b"brand,model\nMaruti,Swift\n", "Missing columns."),
    ],
)
def test_upload_bad_files_explained(content: bytes, title: str) -> None:
    """Bad files give a titled, plain-language problem instead of raising."""
    frame, problem = check_upload(content)
    assert frame is None and problem.title == title and problem.message


def test_upload_missing_columns_listed() -> None:
    """The missing column names are listed for the user."""
    _, problem = check_upload(b"brand,model\nMaruti,Swift\n")
    assert "vehicle_age" in problem.items


# ---------------------------------------------------------------------------
# Market summaries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    """Cleaned listings."""
    return pd.read_csv(get_config().paths.interim_data_dir / "cars_clean.csv")


def test_market_filter_and_summaries(market: pd.DataFrame) -> None:
    """Filters narrow the data; depreciation is negative; rankings respect the minimum."""
    diesel = apply_filter(market, MarketFilter(fuels=("Diesel",), age_range=(0, 10)))
    assert set(diesel["fuel_type"]) == {"Diesel"} and diesel["vehicle_age"].max() <= 10
    assert annual_depreciation_pct(median_by_age(market)) < 0
    ranking = brand_ranking(market, min_listings=30, top_n=5)
    assert len(ranking) == 5 and (ranking["n_listings"] >= 30).all()
    assert ranking["median_price"].is_monotonic_decreasing
    assert peer_median(market, "Maruti", 5)["n_listings"] > 0
    assert peer_median(market, "Maruti", 99) is None


# ---------------------------------------------------------------------------
# Response time and AppTest smoke tests
# ---------------------------------------------------------------------------


def test_estimate_with_explanation_under_two_seconds() -> None:
    """Prediction + TreeSHAP factors + similar listings stay within the GUI budget."""
    predictor = PricePredictor()
    train = pd.read_csv(get_config().paths.processed_data_dir / "train.csv")
    car = CarSpec.from_mapping(
        pd.read_csv(get_config().paths.models_dir / "model_specs.csv").iloc[0].to_dict()
        | {"vehicle_age": 5, "km_driven": 50000, "seller_type": "Dealer"}
    )
    started = time.perf_counter()
    predictor.predict_one(car)
    explain_car(predictor, car, 5)
    similar_listing_range(train, car.__dict__, 1)
    assert time.perf_counter() - started < get_config().gui.max_response_seconds


@pytest.fixture(scope="module")
def home() -> AppTest:
    """The app's default page after its first run."""
    return AppTest.from_file(HOME, default_timeout=APP_TIMEOUT).run()


def test_home_runs_and_prices_default_car(home: AppTest) -> None:
    """The estimator renders without exceptions and prices the default car."""
    assert not home.exception
    estimate = home.session_state["estimate"]
    assert estimate["result"].price_inr > 0
    assert len(estimate["factors"]) == get_config().prediction.explanation_factors


def test_brand_filters_models_and_prefills(home: AppTest) -> None:
    """Choosing a brand filters the model list and pre-fills the model's specs."""
    home.selectbox(key="brand").set_value("Hyundai").run()
    models = home.selectbox(key="model").options
    schema_models = set(PricePredictor().schema["brands"]["Hyundai"])
    assert set(models) == schema_models
    spec = default_spec(
        pd.read_csv(get_config().paths.models_dir / "model_specs.csv"),
        "Hyundai",
        home.session_state["model"],
    )
    assert home.session_state["engine"] == spec["engine"]


def test_estimate_button_updates_result(home: AppTest) -> None:
    """Pressing Estimate price re-prices the car shown in the form."""
    home.selectbox(key="brand").set_value("Honda").run()
    home.button[0].click().run()
    assert not home.exception
    assert home.session_state["estimate"]["car"].brand == "Honda"


@pytest.mark.parametrize(
    "page",
    ["pages/1_market_explorer.py", "pages/2_model_performance.py", "pages/3_about_and_data.py"],
)
def test_other_pages_render(page: str) -> None:
    """Every page renders with no exception and exactly one H1."""
    app = AppTest.from_file(HOME, default_timeout=APP_TIMEOUT).run()
    app.switch_page(page).run()
    assert not app.exception
    titles = [h for h in app.get("html") if 'class="cp-title"' in h.value]
    assert len(titles) == 1
