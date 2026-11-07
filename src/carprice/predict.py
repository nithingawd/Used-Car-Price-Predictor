"""Phase 6: validated single and batch price prediction from the saved pipeline.

Typical use::

    predictor = PricePredictor()
    result = predictor.predict_one({"brand": "Maruti", "model": "Swift", ...})
    result.price_inr, result.lower_inr, result.upper_inr

Inputs are checked against ``models/input_schema.json``, which is built from
``data/processed/train.csv`` only (D-029, D-030). Invalid input raises ``InputError``
(a ``ValueError``) whose message is plain language, ready for the GUI or CLI to show.
This module avoids importing the training stack so that it loads quickly.
"""

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import json
import logging
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from carprice.config import get_config
from carprice.formatting import format_inr_indian, format_lakh

logger = logging.getLogger(__name__)

CATEGORICAL_INPUTS: tuple[str, ...] = (
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type",
)
INTEGER_INPUTS: tuple[str, ...] = ("vehicle_age", "km_driven", "engine", "seats")
FLOAT_INPUTS: tuple[str, ...] = ("mileage", "max_power")
RANGED_INPUTS: tuple[str, ...] = ("vehicle_age", "km_driven", "mileage", "engine", "max_power")
FIELD_LABELS: dict[str, str] = {
    "brand": "Brand",
    "model": "Model",
    "seller_type": "Seller type",
    "fuel_type": "Fuel type",
    "transmission_type": "Transmission",
    "vehicle_age": "Vehicle age (years)",
    "km_driven": "Kilometres driven",
    "mileage": "Mileage (kmpl)",
    "engine": "Engine size (cc)",
    "max_power": "Max power (bhp)",
    "seats": "Seats",
}
MAX_LISTED_CHOICES = 12
MAX_REPORTED_PROBLEMS = 10


class InputError(ValueError):
    """Invalid car specification. ``problems`` lists each issue in plain language."""

    def __init__(self, problems: list[str]) -> None:
        """Store the problems and build one readable message."""
        self.problems = problems
        super().__init__("Please check the car details: " + " ".join(problems))


def normalise_label(value: Any) -> str:
    """Trim and title-case a label exactly as the Phase 1 cleaning did ('  bmw ' -> 'Bmw')."""
    return str(value).strip().title()


def prediction_interval(
    pred_inr: np.ndarray, q_low: float, q_high: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply D-020 log-residual quantiles: [expm1(log1p(pred)+q_low), expm1(log1p(pred)+q_high)]."""
    log_pred = np.log1p(np.asarray(pred_inr, dtype=float))
    return np.expm1(log_pred + q_low), np.expm1(log_pred + q_high)


# ---------------------------------------------------------------------------
# Input schema (built from train.csv only)
# ---------------------------------------------------------------------------


def build_input_schema(train_df: pd.DataFrame) -> dict[str, Any]:
    """Allowed categories, brand -> model counts and numeric ranges seen in training.

    Args:
        train_df: The training partition (never the test set).

    Returns:
        dict[str, Any]: JSON-serialisable schema used by validate_spec().
    """
    brand_models = train_df.groupby(["brand", "model"]).size()
    return {
        "source": "data/processed/train.csv",
        "n_training_rows": len(train_df),
        "brands": {
            brand: {model: int(n) for model, n in group.droplevel(0).items()}
            for brand, group in brand_models.groupby(level=0)
        },
        "categories": {
            col: {str(k): int(v) for k, v in train_df[col].value_counts().items()}
            for col in ("seller_type", "fuel_type", "transmission_type")
        },
        "ranges": {
            col: {"min": float(train_df[col].min()), "max": float(train_df[col].max())}
            for col in RANGED_INPUTS
        },
        "seats_allowed": sorted(int(s) for s in train_df["seats"].unique()),
        "km_cap": get_config().cleaning.km_cap,
    }


def schema_path(models_dir: Path | None = None) -> Path:
    """Location of the input schema next to the saved pipeline."""
    directory = models_dir or get_config().paths.models_dir
    return directory / get_config().prediction.input_schema_file


def write_input_schema(models_dir: Path | None = None) -> Path:
    """Build the schema from train.csv and save it as JSON."""
    train_df = pd.read_csv(get_config().paths.processed_data_dir / "train.csv")
    path = schema_path(models_dir)
    path.write_text(json.dumps(build_input_schema(train_df), indent=2), encoding="utf-8")
    logger.info("Saved input schema to %s", path)
    return path


def read_json(path: Path, hint: str) -> dict[str, Any]:
    """Read a JSON file or fail with a message saying how to create it."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}. {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Typed input and output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarSpec:
    """One used car as a user describes it (units in FIELD_LABELS)."""

    brand: str
    model: str
    vehicle_age: int
    km_driven: int
    seller_type: str
    fuel_type: str
    transmission_type: str
    mileage: float
    engine: int
    max_power: float
    seats: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CarSpec":
        """Build a spec from a dict or row, normalising labels and converting numbers.

        Raises:
            InputError: If fields are missing or numbers cannot be read.
        """
        values, problems = parse_fields(data)
        if problems:
            raise InputError(problems)
        return cls(**values)

    def to_frame(self) -> pd.DataFrame:
        """One-row DataFrame in the column layout the pipeline was trained on."""
        row = asdict(self)
        row["km_was_capped"] = self.km_driven >= get_config().cleaning.km_cap
        return pd.DataFrame([row])


def parse_number(name: str, raw: Any, problems: list[str]) -> float | int | None:
    """Convert one numeric field, appending a plain-language problem if impossible."""
    label = FIELD_LABELS[name]
    try:
        number = float(str(raw).replace(",", "").strip())
    except ValueError:
        problems.append(f"{label} must be a number (got '{raw}').")
        return None
    if not np.isfinite(number):
        problems.append(f"{label} must be a finite number.")
        return None
    if name in INTEGER_INPUTS:
        if number != int(number):
            problems.append(f"{label} must be a whole number (got {raw}).")
            return None
        return int(number)
    return number


def is_blank(raw: Any) -> bool:
    """True for None, NaN or an empty string."""
    return raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == ""


def parse_fields(data: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Read every CarSpec field it can; return the parsed values and any read problems.

    Fields that are missing or unreadable are left out of the values, so the later rule
    checks can still run on the rest and the user sees every problem at once.
    """
    problems: list[str] = []
    values: dict[str, Any] = {}
    for name in (f.name for f in fields(CarSpec)):
        raw = data.get(name)
        if is_blank(raw):
            problems.append(f"{FIELD_LABELS[name]} is missing.")
        elif name in CATEGORICAL_INPUTS:
            values[name] = normalise_label(raw)
        else:
            number = parse_number(name, raw, problems)
            if number is not None:
                values[name] = number
    return values, problems


@dataclass(frozen=True)
class PredictionResult:
    """Estimated price with its 80% range, the model version and any cautions."""

    price_inr: float
    lower_inr: float
    upper_inr: float
    interval_level: float
    model_version: str
    cautions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def price_text(self) -> str:
        """'₹ 5,50,000 (₹ 5.50 lakh)' style text for the point estimate (A7)."""
        return f"{format_inr_indian(self.price_inr)} ({format_lakh(self.price_inr)})"

    @property
    def range_text(self) -> str:
        """Readable interval, e.g. '₹ 4,57,000 to ₹ 6,61,000 (80% range)'."""
        level = round(self.interval_level * 100)
        lower, upper = format_inr_indian(self.lower_inr), format_inr_indian(self.upper_inr)
        return f"{lower} to {upper} ({level}% range)"


# ---------------------------------------------------------------------------
# Validation rules (D-029) and brand/model consistency (D-030)
# ---------------------------------------------------------------------------


def choices_text(options: list[str]) -> str:
    """Comma-separated list, shortened with 'and N more' when long."""
    ordered = sorted(options)
    shown = ", ".join(ordered[:MAX_LISTED_CHOICES])
    extra = len(ordered) - MAX_LISTED_CHOICES
    return shown + (f" and {extra} more" if extra > 0 else "")


def brand_model_problems(values: Mapping[str, Any], schema: dict[str, Any]) -> list[str]:
    """Brand must be known; model must be a model of that brand in the training data (D-030)."""
    brands: dict[str, dict[str, int]] = schema["brands"]
    brand, model = values.get("brand"), values.get("model")
    if brand is None:
        return []
    if brand not in brands:
        return [
            f"Brand '{brand}' is not in the training data. Known brands: "
            f"{choices_text(list(brands))}."
        ]
    if model is None or model in brands[brand]:
        return []
    owners = [name for name, models in brands.items() if model in models]
    if owners:
        return [f"Model '{model}' is made by {owners[0]}, not {brand}."]
    return [
        f"Model '{model}' is not a known {brand} model. Known {brand} "
        f"models: {choices_text(list(brands[brand]))}."
    ]


def category_problems(values: Mapping[str, Any], schema: dict[str, Any]) -> list[str]:
    """Seller, fuel and transmission must be levels seen in training."""
    problems = []
    for name, allowed in schema["categories"].items():
        value = values.get(name)
        if value is not None and value not in allowed:
            problems.append(
                f"{FIELD_LABELS[name]} '{value}' is not recognised. "
                f"Choose one of: {choices_text(list(allowed))}."
            )
    return problems


def range_problems(values: Mapping[str, Any], schema: dict[str, Any]) -> list[str]:
    """Numbers must lie within the range seen in training; seats in the allowed set."""
    problems = []
    for name, bounds in schema["ranges"].items():
        value = values.get(name)
        if value is not None and not bounds["min"] <= value <= bounds["max"]:
            problems.append(
                f"{FIELD_LABELS[name]} must be between {bounds['min']:g} and "
                f"{bounds['max']:g} (the range seen in training); got {value:g}."
            )
    seats = values.get("seats")
    if seats is not None and seats not in schema["seats_allowed"]:
        allowed = ", ".join(str(s) for s in schema["seats_allowed"])
        problems.append(f"Seats must be one of {allowed}; got {seats}.")
    return problems


def validate_values(values: Mapping[str, Any], schema: dict[str, Any]) -> list[str]:
    """Rule checks on parsed values; fields absent from values are skipped (D-029)."""
    return (
        brand_model_problems(values, schema)
        + category_problems(values, schema)
        + range_problems(values, schema)
    )


def validate_spec(spec: CarSpec, schema: dict[str, Any]) -> list[str]:
    """Every rule problem with a complete spec, in plain language (empty list means valid)."""
    return validate_values(asdict(spec), schema)


def input_problems(data: CarSpec | Mapping[str, Any], schema: dict[str, Any]) -> list[str]:
    """All problems with raw input: unreadable fields and broken rules, reported together."""
    if isinstance(data, CarSpec):
        return validate_spec(data, schema)
    values, problems = parse_fields(data)
    return problems + validate_values(values, schema)


def caution_messages(spec: CarSpec, price_inr: float, schema: dict[str, Any]) -> tuple[str, ...]:
    """D-026 weak segments that apply to this car, as short plain-language cautions."""
    config = get_config()
    cautions = []
    if spec.vehicle_age >= config.prediction.old_car_age_years:
        cautions.append(
            f"Cars {config.prediction.old_car_age_years} or more years old are the "
            "model's weakest group: condition matters more than the data can show, "
            "so treat this estimate as rough."
        )
    if (
        spec.brand in config.features.luxury_brands
        or price_inr > config.evaluation.luxury_threshold_inr
    ):
        cautions.append(
            "Luxury and high-priced cars are priced less accurately: they are rare in "
            "the data, and trim and variant, which the data lacks, move their prices a lot."
        )
    model_rows = schema["brands"].get(spec.brand, {}).get(spec.model, 0)
    if model_rows < config.prediction.rare_training_rows:
        cautions.append(
            f"Only {model_rows} {spec.brand} {spec.model} cars were in the training "
            "data, so this estimate rests on few examples."
        )
    fuel_rows = schema["categories"]["fuel_type"].get(spec.fuel_type, 0)
    if fuel_rows < config.prediction.rare_training_rows:
        cautions.append(
            f"Only {fuel_rows} {spec.fuel_type} cars were in the training data, "
            "so fuel-specific pricing is uncertain."
        )
    return tuple(cautions)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class PricePredictor:
    """Load the final pipeline, model card and input schema once; predict many times."""

    def __init__(self, models_dir: Path | None = None) -> None:
        """Load artefacts from models_dir (default: config paths.models_dir).

        Raises:
            FileNotFoundError: With a hint naming the command that creates the missing file.
        """
        directory = models_dir or get_config().paths.models_dir
        pipeline_path = directory / "final_pipeline.joblib"
        if not pipeline_path.is_file():
            raise FileNotFoundError(f"{pipeline_path.name} not found. Run: python tasks.py train")
        self.pipeline = joblib.load(pipeline_path)
        self.card = read_json(directory / "model_card.json", "Run: python tasks.py train")
        self.schema = read_json(schema_path(directory), "Run: python tasks.py predict")
        interval = self.card["prediction_interval_80pct"]
        self.q_low, self.q_high = interval["residual_quantile_10"], interval["residual_quantile_90"]
        self.interval_level = get_config().evaluation.interval_nominal_coverage
        self.warm_up()

    def warm_up(self) -> None:
        """Run one throwaway prediction so the first real request is not slowed by lazy init."""
        brand, models = next(iter(self.schema["brands"].items()))
        sample = {
            "brand": brand,
            "model": next(iter(models)),
            "seller_type": "Dealer",
            "fuel_type": "Petrol",
            "transmission_type": "Manual",
            "seats": 5,
            **{name: bounds["min"] for name, bounds in self.schema["ranges"].items()},
        }
        self.pipeline.predict(pd.DataFrame([{**sample, "km_was_capped": False}]))

    @property
    def model_version(self) -> str:
        """Human-readable model identity from the model card."""
        card = self.card
        trained = card["timestamp_utc"][:10]
        return f"{card['model_name']}, commit {card['git_commit']}, trained {trained}"

    def check(self, spec: CarSpec | Mapping[str, Any]) -> CarSpec:
        """Validate raw or typed input and return the CarSpec.

        Raises:
            InputError: Listing every problem found, in plain language.
        """
        problems = input_problems(spec, self.schema)
        if problems:
            raise InputError(problems)
        return spec if isinstance(spec, CarSpec) else CarSpec.from_mapping(spec)

    def predict_one(self, spec: CarSpec | Mapping[str, Any]) -> PredictionResult:
        """Validate one car and return its estimate, 80% range and cautions."""
        car = self.check(spec)
        price = float(self.pipeline.predict(car.to_frame())[0])
        lower, upper = prediction_interval(np.array([price]), self.q_low, self.q_high)
        return PredictionResult(
            price_inr=price,
            lower_inr=float(lower[0]),
            upper_inr=float(upper[0]),
            interval_level=self.interval_level,
            model_version=self.model_version,
            cautions=caution_messages(car, price, self.schema),
        )

    def row_problems(self, df: pd.DataFrame) -> list[list[str]]:
        """Validation problems for every row (empty list for a valid row)."""
        return [input_problems(record, self.schema) for record in df.to_dict(orient="records")]

    def predict_batch(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Predict many cars at once; returns df plus predicted_inr, lower_inr, upper_inr.

        Args:
            df: One row per car with the CarSpec columns (extra columns are kept).
            validate: True (default) raises InputError naming the bad rows. False predicts
                every row anyway and adds ``passes_validation`` and ``validation_problems``.

        Raises:
            InputError: When validate is True and any row is invalid.
        """
        problems = self.row_problems(df)
        bad = [(i, p) for i, p in enumerate(problems) if p]
        if validate and bad:
            listed = [f"Row {i + 1}: {' '.join(p)}" for i, p in bad[:MAX_REPORTED_PROBLEMS]]
            more = [f"({len(bad) - MAX_REPORTED_PROBLEMS} more rows have problems.)"]
            raise InputError(listed + (more if len(bad) > MAX_REPORTED_PROBLEMS else []))
        return self.score_frame(df, problems)

    def score_frame(self, df: pd.DataFrame, problems: list[list[str]]) -> pd.DataFrame:
        """Run the pipeline on normalised inputs and attach interval and validation columns."""
        inputs = df.copy()
        for name in CATEGORICAL_INPUTS:
            inputs[name] = inputs[name].map(normalise_label)
        inputs["km_was_capped"] = inputs["km_driven"] >= get_config().cleaning.km_cap
        out = df.copy()
        out["predicted_inr"] = self.pipeline.predict(inputs)
        out["lower_inr"], out["upper_inr"] = prediction_interval(
            out["predicted_inr"], self.q_low, self.q_high
        )
        out["passes_validation"] = [not p for p in problems]
        out["validation_problems"] = [" ".join(p) for p in problems]
        return out


# ---------------------------------------------------------------------------
# Command-line interface: python -m carprice.predict {one,batch,build-schema}
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the three sub-commands."""
    parser = argparse.ArgumentParser(description="Used car price prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("one", help="Predict one car from command-line arguments")
    for name in (f.name for f in fields(CarSpec)):
        one.add_argument(
            f"--{name.replace('_', '-')}", dest=name, required=True, help=FIELD_LABELS[name]
        )
    batch = sub.add_parser("batch", help="Predict every row of a CSV file")
    batch.add_argument("--csv", type=Path, required=True, help="Input CSV with CarSpec columns")
    batch.add_argument("--out", type=Path, required=True, help="Output CSV path")
    batch.add_argument(
        "--no-validate",
        action="store_true",
        help="Predict invalid rows too and flag them instead of stopping",
    )
    sub.add_parser("build-schema", help="Rebuild models/input_schema.json from train.csv")
    return parser


def print_result(result: PredictionResult) -> None:
    """Write one prediction to stdout in plain language."""
    lines = [
        f"Estimated price: {result.price_text}",
        f"Likely range:    {result.range_text}",
        f"Model:           {result.model_version}",
    ]
    lines += [f"Caution: {text}" for text in result.cautions]
    sys.stdout.write("\n".join(lines) + "\n")


def run_command(args: argparse.Namespace) -> None:
    """Execute the parsed sub-command."""
    if args.command == "build-schema":
        sys.stdout.write(f"Schema written to {write_input_schema()}\n")
    elif args.command == "one":
        spec = {name: getattr(args, name) for name in (f.name for f in fields(CarSpec))}
        print_result(PricePredictor().predict_one(spec))
    else:
        scored = PricePredictor().predict_batch(
            pd.read_csv(args.csv), validate=not args.no_validate
        )
        scored.to_csv(args.out, index=False)
        sys.stdout.write(f"Predicted {len(scored)} cars -> {args.out}\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 for bad input or missing files (no traceback)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # ₹ on Windows consoles
    args = build_parser().parse_args(argv)
    try:
        run_command(args)
    except InputError as err:
        sys.stderr.write("Could not predict. " + "\n".join(["", *err.problems]) + "\n")
        return 2
    except (FileNotFoundError, KeyError, pd.errors.ParserError) as err:
        sys.stderr.write(f"Could not predict: {err}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
