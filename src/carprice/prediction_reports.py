"""Phase 6 deliverables: test-set prediction file, worked examples, what-if sweep, timing.

Run with ``python tasks.py predict`` (or ``python -m carprice.prediction_reports``).
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from carprice.config import get_config
from carprice.formatting import format_inr_indian, format_lakh
from carprice.insights import (
    build_model_specs,
    contributions,
    default_spec,
    top_factors,
    what_if,
)
from carprice.predict import CarSpec, PricePredictor, write_input_schema
from carprice.viz import (
    CATEGORICAL_PALETTE,
    INK_COLOR,
    MUTED_TEXT,
    apply_style,
    save_figure,
)

logger = logging.getLogger(__name__)

KEY_INPUTS: list[str] = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "fuel_type",
    "transmission_type",
    "seller_type",
    "engine",
    "max_power",
    "mileage",
    "seats",
]
PREDICTION_COLUMNS: list[str] = [
    "actual_inr",
    "predicted_inr",
    "lower_inr",
    "upper_inr",
    "abs_error_inr",
    "pct_error",
    "abs_pct_error",
    "inside_interval",
    "passes_validation",
    "validation_problems",
]
EXAMPLE_FACTORS = 2


def predictions_dir() -> Path:
    """reports/predictions/ (created on demand)."""
    path = get_config().paths.reports_dir / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Task 6.5: every test row
# ---------------------------------------------------------------------------


def score_test_set(predictor: PricePredictor) -> pd.DataFrame:
    """Predict every test row (validate=False so none is dropped) with errors and flags (D-031)."""
    test = pd.read_csv(get_config().paths.processed_data_dir / "test.csv")
    scored = predictor.predict_batch(test[KEY_INPUTS], validate=False)
    scored["actual_inr"] = test["selling_price"].to_numpy()
    scored["abs_error_inr"] = (scored["predicted_inr"] - scored["actual_inr"]).abs()
    scored["pct_error"] = (scored["predicted_inr"] / scored["actual_inr"] - 1) * 100
    scored["abs_pct_error"] = scored["pct_error"].abs()
    scored["inside_interval"] = scored["actual_inr"].between(
        scored["lower_inr"], scored["upper_inr"]
    )
    rounding = {
        "predicted_inr": 0,
        "lower_inr": 0,
        "upper_inr": 0,
        "abs_error_inr": 0,
        "pct_error": 2,
        "abs_pct_error": 2,
    }
    return scored[KEY_INPUTS + PREDICTION_COLUMNS].round(rounding)


# ---------------------------------------------------------------------------
# Task 6.6: worked examples
# ---------------------------------------------------------------------------


def example_filters(scored: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the eight 'typical car' categories (D-031)."""
    config = get_config()
    luxury = scored["brand"].isin(config.features.luxury_brands)
    cheap_cut = scored["actual_inr"].quantile(1 / config.evaluation.price_band_bins)
    return {
        "Cheap petrol hatchback": (scored["actual_inr"] <= cheap_cut)
        & (scored["fuel_type"] == "Petrol")
        & (scored["transmission_type"] == "Manual")
        & (scored["seats"] == 5),
        "Family diesel SUV or MPV": (scored["fuel_type"] == "Diesel")
        & (scored["seats"] >= 7)
        & ~luxury,
        "Old car (12+ years)": scored["vehicle_age"] >= config.evaluation.age_band_edges[-1],
        "Luxury car": luxury & (scored["actual_inr"] > config.evaluation.luxury_threshold_inr),
        "CNG car": scored["fuel_type"] == "Cng",
        "Automatic petrol city car": (scored["transmission_type"] == "Automatic")
        & (scored["fuel_type"] == "Petrol")
        & ~luxury,
        "High-kilometre car": scored["km_driven"] >= max(get_config().prediction.whatif_km),
        "Nearly new car (0-1 year)": scored["vehicle_age"] <= 1,
    }


def typical_row(scored: pd.DataFrame, mask: pd.Series, taken: set[int]) -> int:
    """Index of the row whose |% error| is closest to its category's median (unbiased pick)."""
    pool = scored[mask & ~scored.index.isin(taken)]
    if pool.empty:
        raise ValueError("No test car left for a worked-example category")
    gap = (pool["abs_pct_error"] - pool["abs_pct_error"].median()).abs()
    return int(gap.sort_values(kind="stable").index[0])


def select_worked_examples(scored: pd.DataFrame) -> pd.DataFrame:
    """Eight typical cars by category, plus the single best and worst prediction."""
    chosen: dict[int, str] = {}
    for category, mask in example_filters(scored).items():
        chosen[typical_row(scored, mask, set(chosen))] = category
    remaining = scored[~scored.index.isin(list(chosen))]
    chosen[int(remaining["abs_pct_error"].idxmin())] = "Best prediction in the test set"
    chosen[int(remaining["abs_pct_error"].idxmax())] = "Worst prediction in the test set"
    examples = scored.loc[list(chosen)].copy()
    examples.insert(0, "category", list(chosen.values()))
    return examples


def verdict(abs_pct_error: float) -> str:
    """'Accurate', 'Close' or 'Miss' using the prediction.worked_example_* thresholds."""
    settings = get_config().prediction
    if abs_pct_error <= settings.worked_example_good_ape_pct:
        return "Accurate"
    if abs_pct_error > settings.worked_example_poor_ape_pct:
        return "Miss"
    return "Close"


def example_context(row: pd.Series, training_rows: int) -> str:
    """Why this car was easy (accurate) or hard (close or miss), per the D-026 segment rules."""
    config = get_config()
    rare = training_rows < config.prediction.rare_training_rows
    if verdict(row["abs_pct_error"]) == "Accurate":
        if rare:
            return f"it is accurate despite only {training_rows} training cars of this model"
        return f"this is a well-represented model with {training_rows} training cars"
    return weakness_context(row, training_rows)


def weakness_context(row: pd.Series, training_rows: int) -> str:
    """The first D-026 weak-segment reason that applies to an imperfect prediction."""
    config = get_config()
    if not row["passes_validation"]:
        return "the brand never appears in training, so the model fell back on generic values"
    if training_rows < config.prediction.rare_training_rows:
        return f"only {training_rows} cars of this model were in the training data"
    if row["vehicle_age"] >= config.prediction.old_car_age_years:
        return "old cars differ in condition, which the data does not record"
    if row["brand"] in config.features.luxury_brands:
        return "luxury prices hinge on trim and variant, which the data lacks"
    if verdict(row["abs_pct_error"]) == "Miss":
        return "the listing price is unusual for these specifications"
    return (
        f"it is a well-represented model ({training_rows} training cars), so the gap is "
        "ordinary variation between similar listings"
    )


def explain_example(row: pd.Series, factors: pd.DataFrame, training_rows: int) -> str:
    """One sentence: verdict, signed error, interval, two main drivers and the context."""
    inside = "inside" if row["inside_interval"] else "outside"
    drivers = " and ".join(
        f"{f.label.lower()} ({f.effect_pct:+.0f}%)" for f in factors.itertuples()
    )
    return (
        f"{verdict(row['abs_pct_error'])}: the estimate is {row['pct_error']:+.2f}% from the "
        f"listing and {inside} the 80% range; the biggest influences were {drivers}, and "
        f"{example_context(row, training_rows)}."
    )


def describe_examples(predictor: PricePredictor, examples: pd.DataFrame) -> pd.DataFrame:
    """Add the top drivers and the one-sentence explanation to each worked example."""
    grouped, _ = contributions(
        predictor.pipeline,
        examples[KEY_INPUTS].assign(
            km_was_capped=examples["km_driven"] >= get_config().cleaning.km_cap
        ),
    )
    brands = predictor.schema["brands"]
    sentences, drivers = [], []
    for index, row in examples.iterrows():
        factors = top_factors(grouped.loc[index], EXAMPLE_FACTORS)
        training_rows = brands.get(row["brand"], {}).get(row["model"], 0)
        sentences.append(explain_example(row, factors, training_rows))
        drivers.append("; ".join(f"{f.label} {f.effect_pct:+.1f}%" for f in factors.itertuples()))
    return examples.assign(top_drivers=drivers, explanation=sentences)


def price_text(value: float) -> str:
    """'₹ 5,50,000 (₹ 5.50 lakh)'."""
    return f"{format_inr_indian(value)} ({format_lakh(value)})"


def example_markdown(row: pd.Series, number: int) -> str:
    """Markdown block for one worked example."""
    specs = (
        f"{row['vehicle_age']} year{'' if row['vehicle_age'] == 1 else 's'}, "
        f"{row['km_driven']:,} km, {row['fuel_type']}, "
        f"{row['transmission_type']}, {row['engine']} cc, {row['max_power']:g} bhp, "
        f"{row['mileage']:g} kmpl, {row['seats']} seats, {row['seller_type']} seller"
    )
    interval = f"{format_inr_indian(row['lower_inr'])} to {format_inr_indian(row['upper_inr'])}"
    return "\n".join(
        [
            f"### {number}. {row['category']}: {row['brand']} {row['model']}",
            "",
            f"- **Inputs:** {specs}",
            f"- **Actual listing price:** {price_text(row['actual_inr'])}",
            f"- **Predicted:** {price_text(row['predicted_inr'])}, 80% range {interval}",
            f"- **Why:** {row['explanation']}",
            "",
        ]
    )


def write_worked_examples(examples: pd.DataFrame) -> Path:
    """reports/predictions/worked_examples.md, generated from worked_examples.csv."""
    intro = [
        "# Worked examples: ten real test cars",
        "",
        "Ten cars from the locked test set, chosen by fixed rules (D-031), not by hand. Eight are "
        "the *typical* car of a category: the one whose error is closest to that category's median "
        "error. The last two are the single best and worst predictions in the test set. Influence "
        "percentages are each feature's TreeSHAP effect on this car's predicted price relative to "
        "an average car. All numbers come from `worked_examples.csv` in this folder.",
        "",
    ]
    blocks = [example_markdown(row, i) for i, (_, row) in enumerate(examples.iterrows(), start=1)]
    path = predictions_dir() / "worked_examples.md"
    path.write_text("\n".join(intro + blocks), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Task 6.7: what-if sweep
# ---------------------------------------------------------------------------


def reference_car(model_specs: pd.DataFrame) -> CarSpec:
    """The most common model in train.csv with its typical specs, age, km and seller."""
    top = model_specs.sort_values("n_training_rows", ascending=False).iloc[0]
    spec = default_spec(model_specs, top["brand"], top["model"])
    return CarSpec.from_mapping(
        {
            **spec,
            "vehicle_age": round(spec["median_vehicle_age"]),
            "km_driven": round(spec["median_km_driven"]),
            "seller_type": "Individual",
        }
    )


def whatif_grid(predictor: PricePredictor, car: CarSpec) -> pd.DataFrame:
    """Predicted price for every (age, km) pair in the configured sweep."""
    settings = get_config().prediction
    frames = []
    for km in settings.whatif_km:
        base = CarSpec(**{**asdict(car), "km_driven": km})
        frames.append(
            what_if(predictor, base, "vehicle_age", settings.whatif_ages).assign(km_driven=km)
        )
    grid = pd.concat(frames, ignore_index=True)
    grid.insert(0, "reference_car", f"{car.brand} {car.model}")
    return grid.round({"predicted_inr": 0, "lower_inr": 0, "upper_inr": 0})


def nearest(options: list[int], target: float) -> int:
    """The option closest to target (first one on ties)."""
    return min(options, key=lambda option: abs(option - target))


def change_pct(grid: pd.DataFrame, fixed: str, fixed_value: int, swept: str) -> float:
    """% change in predicted price from the first to the last swept value."""
    line = grid[grid[fixed] == fixed_value].sort_values(swept)
    return float((line["predicted_inr"].iloc[-1] / line["predicted_inr"].iloc[0] - 1) * 100)


def plot_whatif(grid: pd.DataFrame, car: CarSpec) -> plt.Figure:
    """Two panels: price vs age at several km levels, price vs km at several ages."""
    apply_style()
    settings = get_config().prediction
    fig, (ax_age, ax_km) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for colour, km in zip(CATEGORICAL_PALETTE, settings.whatif_chart_km_lines, strict=False):
        line = grid[grid["km_driven"] == km]
        ax_age.plot(
            line["vehicle_age"],
            line["predicted_inr"] / 1e5,
            marker="o",
            markersize=3,
            color=colour,
            label=f"{km:,} km",
        )
    for colour, age in zip(CATEGORICAL_PALETTE, settings.whatif_chart_age_lines, strict=False):
        line = grid[grid["vehicle_age"] == age]
        ax_km.plot(
            line["km_driven"] / 1000,
            line["predicted_inr"] / 1e5,
            marker="o",
            markersize=3,
            color=colour,
            label=f"{age} yr",
        )
    ax_age.set(xlabel="Vehicle age (years)", ylabel="Predicted price (₹ lakh)")
    ax_km.set(xlabel="Kilometres driven (thousand km)")
    for ax, title in ((ax_age, "Kilometres driven"), (ax_km, "Vehicle age")):
        ax.legend(title=title, frameon=False)
        ax.grid(True, alpha=0.5)
    typical_km = nearest(settings.whatif_km, car.km_driven)
    typical_age = nearest(settings.whatif_ages, car.vehicle_age)
    age_drop = change_pct(grid, "km_driven", typical_km, "vehicle_age")
    km_drop = change_pct(grid, "vehicle_age", typical_age, "km_driven")
    fig.suptitle(
        f"For a typical {car.brand} {car.model}, age moves price more than km: "
        f"{age_drop:+.1f}% over 1-12 years at {typical_km:,} km vs {km_drop:+.1f}% "
        f"over 10k-150k km at {typical_age} years",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    fig.text(
        0.01,
        -0.03,
        f"Other inputs held at the model's most common specs: {car.engine} cc, "
        f"{car.max_power:g} bhp, {car.fuel_type}, {car.transmission_type}. Steps come from "
        "the tree model's split points. A 1-year-old car with high km implies very heavy use "
        "(km per year), which the model prices down sharply; such cars are rare in the data.",
        fontsize=9,
        color=MUTED_TEXT,
    )
    return fig


# ---------------------------------------------------------------------------
# Timing (acceptance check)
# ---------------------------------------------------------------------------

TIMING_SCRIPT = """
import json, statistics, sys, time
sys.path.insert(0, "src")
start = time.perf_counter()
import sklearn.compose, sklearn.impute, sklearn.pipeline, sklearn.preprocessing, xgboost
import carprice.features, carprice.formatting
from carprice.predict import PricePredictor
imported = time.perf_counter()
predictor = PricePredictor()
loaded = time.perf_counter()
spec = json.loads(sys.argv[1])
for _ in range(int(sys.argv[3])):
    predictor.predict_one(spec)
times = []
for _ in range(int(sys.argv[2])):
    t0 = time.perf_counter()
    predictor.predict_one(spec)
    times.append(1000 * (time.perf_counter() - t0))
print(json.dumps({"library_import_s": imported - start, "predictor_load_s": loaded - imported,
                  "single_median_ms": statistics.median(times),
                  "single_p95_ms": sorted(times)[int(0.95 * len(times)) - 1]}))
"""


def time_one_process(car: CarSpec) -> dict[str, float]:
    """Import, load and steady-state prediction timings from one fresh Python process."""
    settings = get_config().prediction
    args = [json.dumps(asdict(car)), str(settings.timing_repeats), str(settings.timing_warmup)]
    result = subprocess.run(
        [sys.executable, "-c", TIMING_SCRIPT, *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=get_config().paths.root_dir,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def measure_timing(car: CarSpec) -> pd.DataFrame:
    """Best-of-N fresh-process timings (timeit convention), with the worst run kept too (D-031).

    Returns:
        pd.DataFrame: One row per statistic ('best' and 'worst' run) with pass flags.
    """
    settings = get_config().prediction
    runs = pd.DataFrame([time_one_process(car) for _ in range(settings.timing_runs)])
    rows = []
    for label, values in (("best", runs.min()), ("worst", runs.max())):
        rows.append(
            {
                "statistic": f"{label} of {settings.timing_runs} runs",
                **values.round(3).to_dict(),
                "max_load_s": settings.max_load_seconds,
                "max_single_ms": settings.max_single_prediction_ms,
                "load_passes": values["predictor_load_s"] < settings.max_load_seconds,
                "single_passes": values["single_median_ms"] < settings.max_single_prediction_ms,
                "repeats": settings.timing_repeats,
                "warmup_calls": settings.timing_warmup,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_model_specs() -> pd.DataFrame:
    """models/model_specs.csv: default specs per brand/model from train.csv (GUI pre-fill)."""
    train = pd.read_csv(get_config().paths.processed_data_dir / "train.csv")
    specs = build_model_specs(train)
    specs.to_csv(get_config().paths.models_dir / "model_specs.csv", index=False)
    return specs


def run_prediction_reports() -> dict[str, Any]:
    """Build schema and spec lookup, then every Phase 6 output file."""
    write_input_schema()
    specs = write_model_specs()
    predictor = PricePredictor()
    out_dir, tables_dir = predictions_dir(), get_config().paths.tables_dir
    scored = score_test_set(predictor)
    scored.to_csv(out_dir / "test_predictions.csv", index=False)
    examples = describe_examples(predictor, select_worked_examples(scored))
    examples.to_csv(out_dir / "worked_examples.csv", index=False)
    write_worked_examples(examples)
    car = reference_car(specs)
    grid = whatif_grid(predictor, car)
    grid.to_csv(out_dir / "whatif_age_km.csv", index=False)
    save_figure(plot_whatif(grid, car), get_config().paths.figures_dir / "whatif_age_km.png")
    timing = measure_timing(car)
    timing.to_csv(tables_dir / "predictor_timing.csv", index=False)
    logger.info("Phase 6 outputs written to %s", out_dir)
    return {"scored": scored, "examples": examples, "grid": grid, "timing": timing}


def main() -> None:
    """Entry point for python -m carprice.prediction_reports."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_prediction_reports()


if __name__ == "__main__":
    main()
