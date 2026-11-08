"""Used Car Price Predictor: Streamlit entry point and the Price Estimator page.

Run with:  streamlit run app/Home.py
"""

import time
from typing import Any

import pandas as pd
import streamlit as st

from carprice.config import get_config
from carprice.insights import default_spec, explain_car, similar_listing_range, what_if
from carprice.market import peer_median
from carprice.predict import FIELD_LABELS, CarSpec, InputError, PricePredictor
import components  # noqa: F401  (puts src/ on the import path)
from components.batch import check_upload
from components.charts import compare_chart, factor_chart, whatif_chart
from components.data import get_predictor, model_specs, training_listings
from components.palette import PLOTLY_CONFIG
from components.ui import (
    empty_state,
    format_inr,
    format_lakh,
    info_note,
    load_theme,
    page_header,
    panel,
    range_bar,
    safely,
    section_header,
)

SPEC_FIELDS = ["engine", "max_power", "mileage", "seats", "fuel_type", "transmission_type"]
INPUT_FIELDS = list(FIELD_LABELS)

# ---------------------------------------------------------------------------
# Form state
# ---------------------------------------------------------------------------


def models_of(specs: pd.DataFrame, brand: str) -> list[str]:
    """Model names of a brand, most common first."""
    rows = specs[specs["brand"] == brand].sort_values("n_training_rows", ascending=False)
    return rows["model"].tolist()


def prefill(specs: pd.DataFrame, brand: str, model: str) -> None:
    """Fill the spec fields with the model's most common values (7.2.2)."""
    spec = default_spec(specs, brand, model)
    for name in SPEC_FIELDS:
        st.session_state[name] = spec[name]
    st.session_state["seats"] = int(spec["seats"])
    st.session_state["engine"] = int(spec["engine"])


def on_brand_change() -> None:
    """New brand: pick its most common model and pre-fill."""
    specs = model_specs()
    st.session_state["model"] = models_of(specs, st.session_state["brand"])[0]
    prefill(specs, st.session_state["brand"], st.session_state["model"])


def on_model_change() -> None:
    """New model: pre-fill its specs."""
    prefill(model_specs(), st.session_state["brand"], st.session_state["model"])


def init_state(predictor: PricePredictor, specs: pd.DataFrame) -> None:
    """First visit: the most common car in the training data, with its typical age and km."""
    if "brand" in st.session_state:
        return
    top = specs.sort_values("n_training_rows", ascending=False).iloc[0]
    st.session_state["brand"], st.session_state["model"] = top["brand"], top["model"]
    prefill(specs, top["brand"], top["model"])
    step = get_config().gui.km_step
    st.session_state["vehicle_age"] = round(top["median_vehicle_age"])
    st.session_state["km_driven"] = int(round(top["median_km_driven"] / step) * step)
    sellers = predictor.schema["categories"]["seller_type"]
    st.session_state["seller_type"] = max(sellers, key=sellers.get)


def current_inputs() -> dict[str, Any]:
    """The form's values as a CarSpec-shaped dict."""
    return {name: st.session_state.get(name) for name in INPUT_FIELDS}


# ---------------------------------------------------------------------------
# Inputs (left column)
# ---------------------------------------------------------------------------


def vehicle_inputs(predictor: PricePredictor, specs: pd.DataFrame) -> None:
    """Brand, model (filtered by brand), age and seller."""
    section_header("Vehicle")
    st.selectbox("Brand", sorted(specs["brand"].unique()), key="brand", on_change=on_brand_change)
    st.selectbox(
        "Model",
        models_of(specs, st.session_state["brand"]),
        key="model",
        on_change=on_model_change,
        help="Listed most common first. Choosing a model fills in "
        "its typical engine, power, mileage, seats and fuel.",
    )
    st.slider("Vehicle age (years)", 0, get_config().gui.age_slider_max, key="vehicle_age")
    st.selectbox(
        "Seller type", sorted(predictor.schema["categories"]["seller_type"]), key="seller_type"
    )


def condition_inputs(predictor: PricePredictor) -> None:
    """Kilometres driven, bounded by the range seen in training."""
    section_header("Condition")
    km = predictor.schema["ranges"]["km_driven"]
    st.number_input(
        "Kilometres driven",
        min_value=int(km["min"]),
        max_value=int(km["max"]),
        step=get_config().gui.km_step,
        key="km_driven",
        format="%d",
    )


def engine_inputs(predictor: PricePredictor) -> None:
    """Fuel, transmission, engine, power, mileage and seats."""
    section_header("Engine and fuel")
    schema = predictor.schema
    st.radio(
        "Fuel type", sorted(schema["categories"]["fuel_type"]), key="fuel_type", horizontal=True
    )
    st.radio(
        "Transmission",
        sorted(schema["categories"]["transmission_type"]),
        key="transmission_type",
        horizontal=True,
    )
    left, right = st.columns(2)
    ranges = schema["ranges"]
    with left:
        st.number_input(
            "Engine size (cc)",
            int(ranges["engine"]["min"]),
            int(ranges["engine"]["max"]),
            step=50,
            key="engine",
            format="%d",
        )
        st.number_input(
            "Mileage (kmpl)",
            float(ranges["mileage"]["min"]),
            float(ranges["mileage"]["max"]),
            step=0.5,
            key="mileage",
            format="%.2f",
        )
    with right:
        st.number_input(
            "Max power (bhp)",
            float(ranges["max_power"]["min"]),
            float(ranges["max_power"]["max"]),
            step=5.0,
            key="max_power",
            format="%.1f",
        )
        st.selectbox("Seats", schema["seats_allowed"], key="seats")


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------


def run_estimate(predictor: PricePredictor, inputs: dict[str, Any]) -> None:
    """Price, explain and contextualise one car; store the outcome in session state."""
    started = time.perf_counter()
    try:
        car = predictor.check(inputs)
    except InputError as err:
        st.session_state["estimate"] = {"inputs": inputs, "problems": err.problems}
        return
    result = predictor.predict_one(car)
    settings = get_config().prediction
    listings = training_listings()
    st.session_state["estimate"] = {
        "inputs": inputs,
        "car": car,
        "result": result,
        "factors": explain_car(predictor, car, settings.explanation_factors),
        "similar": similar_listing_range(listings, inputs, settings.similar_age_window),
        "peer": peer_median(listings, car.brand, car.vehicle_age),
        "seconds": time.perf_counter() - started,
    }


def similar_message(car: CarSpec, similar: dict[str, Any] | None) -> str:
    """'Similar 5-year-old Hyundai I20s listed for about ...' (7.2.4)."""
    if similar is None:
        return "There are too few comparable listings in the data to quote a market range."
    window = get_config().prediction.similar_age_window
    who = f"{car.brand} {car.model}s" if similar["scope"] == "model" else f"{car.brand} cars"
    age = f"{car.vehicle_age}-year-old" if car.vehicle_age else "nearly new"
    return (
        f"Similar {age} {who} (within {window} year) listed for about "
        f"{format_lakh(similar['q25'])} to {format_lakh(similar['q75'])}, "
        f"the middle half of {similar['n_listings']:,} listings."
    )


def result_card(estimate: dict[str, Any], stale: bool) -> None:
    """The estimate (largest element), its 80% range, context and cautions."""
    result, car = estimate["result"], estimate["car"]
    with panel("result"):
        status = (
            '<span class="cp-pill">Inputs changed: press Estimate price to update</span>'
            if stale
            else ""
        )
        st.html(
            f'<p class="cp-estimate-label">Estimated price for this {car.brand} {car.model}</p>'
            f'<p class="cp-estimate">{format_inr(result.price_inr)}</p>'
            f'<p class="cp-caption"><span class="cp-mono">{format_lakh(result.price_inr)}</span>'
            f' · likely between <span class="cp-mono">{format_inr(result.lower_inr)}</span> and '
            f'<span class="cp-mono">{format_inr(result.upper_inr)}</span> (80% range)</p>{status}'
        )
        st.plotly_chart(
            range_bar(result.lower_inr, result.price_inr, result.upper_inr),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        info_note(similar_message(car, estimate["similar"]))
        if result.cautions:
            info_note(
                "Treat this estimate with care.",
                kind="caution",
                title="Lower confidence.",
                items=result.cautions,
            )
        st.html(
            f'<p class="cp-caption">{result.model_version} · answered in '
            f"{estimate['seconds']:.2f} s</p>"
        )


def result_column(estimate: dict[str, Any] | None) -> None:
    """Right column: empty state, friendly problems, or the result card."""
    if estimate is None:
        empty_state("No estimate yet", "Fill in the car on the left and press Estimate price.")
    elif "problems" in estimate:
        info_note(
            "We could not price this car yet.",
            kind="error",
            title="Please check the details.",
            items=estimate["problems"],
        )
    else:
        result_card(estimate, stale=estimate["inputs"] != current_inputs())


# ---------------------------------------------------------------------------
# Panels below the result
# ---------------------------------------------------------------------------


def why_panel(estimate: dict[str, Any]) -> None:
    """Top factors for this one car (TreeSHAP)."""
    with panel("why"):
        section_header("Why this price")
        st.plotly_chart(factor_chart(estimate["factors"]), config=PLOTLY_CONFIG, width="stretch")
        top = estimate["factors"].iloc[0]
        st.html(
            f'<p class="cp-caption">{top["label"]} moves this estimate most '
            f"({top['effect_pct']:+.0f}% compared with an average car).</p>"
        )


def compare_panel(estimate: dict[str, Any]) -> None:
    """This car against the median listing of the same brand and age."""
    car, peer = estimate["car"], estimate["peer"]
    with panel("compare"):
        section_header("Compare")
        if peer is None:
            st.html(
                f'<p class="cp-caption">No {car.brand} listings aged {car.vehicle_age} years to '
                "compare.</p>"
            )
            return
        label = f"Typical {car.brand}, {car.vehicle_age} yrs"
        st.plotly_chart(
            compare_chart(estimate["result"].price_inr, peer["median_price"], label),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        gap = (estimate["result"].price_inr / peer["median_price"] - 1) * 100
        st.html(
            f'<p class="cp-caption">This estimate is {abs(gap):.0f}% '
            f"{'above' if gap >= 0 else 'below'} "
            f"the median of {peer['n_listings']:,} {car.brand} listings of the same age.</p>"
        )


def whatif_values(field: str, predictor: PricePredictor) -> tuple[list[float], str]:
    """Sweep values and axis label for the what-if panel."""
    gui = get_config().gui
    if field == "vehicle_age":
        return list(range(0, gui.age_slider_max + 1)), "Vehicle age (years)"
    km_max = min(gui.whatif_km_max, int(predictor.schema["ranges"]["km_driven"]["max"]))
    return list(range(gui.whatif_km_step, km_max + 1, gui.whatif_km_step)), "Kilometres driven"


@st.fragment
def whatif_panel(estimate: dict[str, Any]) -> None:
    """Estimate as age or km changes, other inputs fixed (re-renders on its own)."""
    predictor, car = get_predictor(), estimate["car"]
    with panel("whatif"):
        section_header("What if")
        choice = st.segmented_control(
            "Change",
            ["Age", "Kilometres"],
            default="Age",
            key="whatif_field",
            label_visibility="collapsed",
        )
        field = "km_driven" if choice == "Kilometres" else "vehicle_age"
        values, label = whatif_values(field, predictor)
        curve = what_if(predictor, car, field, values)
        st.plotly_chart(
            whatif_chart(curve, field, getattr(car, field), label),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        first, last = curve["predicted_inr"].iloc[0], curve["predicted_inr"].iloc[-1]
        st.html(
            f'<p class="cp-caption">Across this range the estimate moves from {format_lakh(first)} '
            f"to {format_lakh(last)}; the ringed dot is this car.</p>"
        )


def panels(estimate: dict[str, Any] | None) -> None:
    """Why and compare side by side under the result (what-if goes full width below)."""
    if estimate is None or "result" not in estimate:
        return
    left, right = st.columns(2, gap="medium")
    with left:
        why_panel(estimate)
    with right:
        compare_panel(estimate)


# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------


def template_csv(specs: pd.DataFrame) -> bytes:
    """A two-row example file with exactly the columns the batch upload expects."""
    rows = []
    for _, top in specs.sort_values("n_training_rows", ascending=False).head(2).iterrows():
        rows.append(
            {
                **{k: top[k] for k in ["brand", "model", *SPEC_FIELDS]},
                "vehicle_age": round(top["median_vehicle_age"]),
                "km_driven": int(top["median_km_driven"]),
                "seller_type": "Dealer",
            }
        )
    return pd.DataFrame(rows)[INPUT_FIELDS].to_csv(index=False).encode("utf-8")


def read_upload(upload: Any) -> pd.DataFrame | None:
    """Parse the uploaded CSV, explaining any problem in plain language."""
    frame, problem = check_upload(upload.getvalue())
    if problem is not None:
        info_note(problem.message, kind="error", title=problem.title, items=problem.items)
    return frame


def batch_results(predictor: PricePredictor, frame: pd.DataFrame) -> None:
    """Score valid rows, show a preview and offer the full file for download."""
    with st.spinner("Pricing your cars..."):
        scored = predictor.predict_batch(frame, validate=False)
    refused = ~scored["passes_validation"]
    scored.loc[refused, ["predicted_inr", "lower_inr", "upper_inr"]] = None
    if refused.any():
        info_note(
            f"{int(refused.sum()):,} of {len(scored):,} rows could not be priced and are left "
            "blank "
            "in the download; the reasons are in the validation_problems column.",
            kind="caution",
            title="Some rows need attention.",
            items=scored.loc[refused, "validation_problems"].head(3).tolist(),
        )
    else:
        info_note(f"All {len(scored):,} cars were priced.", title="Done.")
    preview = scored.head(get_config().gui.batch_preview_rows).copy()
    for column in ["predicted_inr", "lower_inr", "upper_inr"]:
        preview[column] = preview[column].map(lambda v: format_inr(v) if pd.notna(v) else "")
    st.dataframe(preview, hide_index=True, width="stretch")
    st.download_button(
        "Download all predictions (CSV)",
        scored.round(0).to_csv(index=False).encode("utf-8"),
        file_name="car_price_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def batch_tab(predictor: PricePredictor, specs: pd.DataFrame) -> None:
    """7.2.6: upload, validate, preview, download."""
    left, right = st.columns([2, 3], gap="large")
    with left, panel("batch-upload"):
        section_header("Price many cars at once")
        st.html(
            '<p class="cp-caption">Upload a CSV with one car per row and the same columns as '
            "the template. Rows are checked with the same rules as the form.</p>"
        )
        st.download_button(
            "Download template", template_csv(specs), file_name="car_template.csv", mime="text/csv"
        )
        upload = st.file_uploader("CSV file", type=["csv"], help="Up to 5,000 rows.")
    with right:
        if upload is None:
            empty_state(
                "No file yet", "Download the template, fill in your cars and upload it here."
            )
            return
        frame = read_upload(upload)
        if frame is not None:
            batch_results(predictor, frame)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def single_tab(predictor: PricePredictor, specs: pd.DataFrame) -> None:
    """Form on the left (about 40%); result and its panels on the right (about 60%)."""
    init_state(predictor, specs)
    if "estimate" not in st.session_state:
        run_estimate(predictor, current_inputs())
    left, right = st.columns([2, 3], gap="large")
    with left, panel("inputs"):
        vehicle_inputs(predictor, specs)
        condition_inputs(predictor)
        engine_inputs(predictor)
        if st.button("Estimate price", type="primary", width="stretch"):
            with st.spinner("Estimating..."):
                run_estimate(predictor, current_inputs())
    with right:
        result_column(st.session_state.get("estimate"))
        panels(st.session_state.get("estimate"))
    estimate = st.session_state.get("estimate")
    if estimate is not None and "result" in estimate:
        whatif_panel(estimate)


def estimator_page() -> None:
    """Page 1: Price estimator."""
    page_header(
        "Price estimator",
        "Describe a used car to get an estimated selling price, a likely "
        "range and the reasons behind it.",
    )
    loaded = safely(lambda: (get_predictor(), model_specs()), "The pricing model")
    if loaded is None:
        info_note(
            "Run `python tasks.py train` and `python tasks.py predict`, then reload this page."
        )
        return
    predictor, specs = loaded
    single, batch = st.tabs(["Single car", "Batch estimate"])
    with single:
        single_tab(predictor, specs)
    with batch:
        batch_tab(predictor, specs)


def main() -> None:
    """Configure the app, inject the theme and route to the selected page."""
    st.set_page_config(
        page_title="Used Car Price Estimator",
        page_icon=":material/directions_car:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_theme()
    pages = [
        st.Page(estimator_page, title="Price estimator", icon=":material/sell:", default=True),
        st.Page("pages/1_market_explorer.py", title="Market explorer", icon=":material/insights:"),
        st.Page(
            "pages/2_model_performance.py", title="Model performance", icon=":material/fact_check:"
        ),
        st.Page("pages/3_about_and_data.py", title="About and data", icon=":material/info:"),
    ]
    st.navigation(pages, position="top").run()


main()
