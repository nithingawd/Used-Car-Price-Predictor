"""Compare the outputs of two checkouts of this project (reproducibility audit, Phase 8).

Usage:
    python scripts/compare_runs.py <reference_root> <candidate_root> [--out report.csv]

For every CSV under reports/tables and reports/predictions, every JSON under models/, the
cleaned data and the split files, it reports whether the two runs agree. Numeric columns
must match within ``RTOL``/``ATOL``; text columns must match exactly. Columns that measure
wall-clock time or record when or where something ran are skipped (listed in
``VOLATILE_COLUMNS`` / ``VOLATILE_JSON_KEYS``). PNG figures are compared byte for byte.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-6
ATOL = 1e-9
CSV_GLOBS = ("reports/tables/*.csv", "reports/predictions/*.csv")
DATA_FILES = (
    "data/interim/cars_clean.csv",
    "data/processed/train.csv",
    "data/processed/test.csv",
    "data/processed/split_indices.json",
)
FIGURE_GLOB = "reports/figures/*.png"
SKIPPED_TABLES = {"predictor_timing.csv"}
VOLATILE_COLUMNS = {
    "fit_time_sec",
    "pred_time_sec",
    "datetime_start",
    "datetime_complete",
    "duration",
}
VOLATILE_JSON_KEYS = {"timestamp_utc", "git_commit"}

logger = logging.getLogger("compare_runs")


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_numeric_gap(left: pd.Series, right: pd.Series) -> float:
    """Largest absolute difference between two numeric columns (NaN pairs count as equal)."""
    both_nan = left.isna() & right.isna()
    gap = (left - right).abs().where(~both_nan, 0.0)
    return float(gap.max()) if len(gap) else 0.0


def is_measure(column: pd.Series) -> bool:
    """True for numeric, non-boolean columns (compared with a tolerance)."""
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, float, str]:
    """Compare two tables column by column, skipping volatile columns.

    Returns:
        tuple: (match, largest numeric gap, note).
    """
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False, float("nan"), f"shape {left.shape} vs {right.shape}"
    worst, failed = 0.0, []
    for column in left.columns:
        if column in VOLATILE_COLUMNS:
            continue
        a, b = left[column], right[column]
        if is_measure(a) and is_measure(b):
            worst = max(worst, max_numeric_gap(a, b))
            if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                failed.append(column)
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            failed.append(column)
    return not failed, worst, ", ".join(failed)


def strip_keys(value: object) -> object:
    """Drop volatile keys from nested JSON data."""
    if isinstance(value, dict):
        return {k: strip_keys(v) for k, v in value.items() if k not in VOLATILE_JSON_KEYS}
    if isinstance(value, list):
        return [strip_keys(v) for v in value]
    return value


def compare_file(reference: Path, candidate: Path, relative: str) -> dict[str, object]:
    """Compare one file present in the reference run with its counterpart."""
    row: dict[str, object] = {"file": relative, "match": False, "max_abs_gap": np.nan, "note": ""}
    if not candidate.exists():
        row["note"] = "missing in candidate"
    elif relative.endswith(".csv"):
        match, gap, note = compare_frames(pd.read_csv(reference), pd.read_csv(candidate))
        row.update(match=match, max_abs_gap=gap, note=note)
    elif relative.endswith(".json"):
        left = strip_keys(json.loads(reference.read_text(encoding="utf-8")))
        right = strip_keys(json.loads(candidate.read_text(encoding="utf-8")))
        row["match"] = left == right
    else:
        row["match"] = file_digest(reference) == file_digest(candidate)
        row["note"] = "byte comparison"
    return row


def relative_paths(root: Path) -> list[str]:
    """Every file the audit compares, relative to the project root."""
    found = [p for pattern in CSV_GLOBS for p in sorted(root.glob(pattern))]
    found += sorted(root.glob("models/*.json")) + sorted(root.glob(FIGURE_GLOB))
    names = [p.relative_to(root).as_posix() for p in found if p.name not in SKIPPED_TABLES]
    return names + [name for name in DATA_FILES if (root / name).exists()]


def compare_runs(reference_root: Path, candidate_root: Path) -> pd.DataFrame:
    """Compare every audited file of the reference run with the candidate run."""
    rows = [
        compare_file(reference_root / name, candidate_root / name, name)
        for name in relative_paths(reference_root)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Parse arguments, compare, log a summary and optionally save the full comparison."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = compare_runs(args.reference, args.candidate)
    if args.out:
        result.to_csv(args.out, index=False)
    logger.info("compared %d files, %d match", len(result), int(result["match"].sum()))
    for row in result[~result["match"].astype(bool)].itertuples():
        logger.info("DIFFERS: %s (%s)", row.file, row.note)
    logger.info("largest numeric gap in matching tables: %s", result["max_abs_gap"].max())


if __name__ == "__main__":
    main()
