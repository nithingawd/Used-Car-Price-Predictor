"""Number provenance: check that every figure quoted in prose exists in a saved output file.

The written documents (EDA findings, evaluation summary, final report, viva notes) may only
quote numbers that appear in ``reports/tables/*.csv``, the Phase 6 prediction files in
``reports/predictions/*.csv``, the JSON files in ``models/`` or ``configs/config.yaml``
(A4: "do not invent numbers"). A quoted number matches a source
value when the source value, rounded to the number of decimals written, equals it. Percent
(x100), lakh (/1e5) and crore (/1e7) forms of a source value are accepted, because prices
are stored in rupees and shares as fractions in some tables.

Ignored when scanning prose: inline code and code blocks (file names, commands), link and
image targets, decision ids such as D-014, ISO dates, and whole numbers up to
``SMALL_INTEGER_LIMIT`` (list numbering, "5-fold", "Q1", "Phase 8").
"""

from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

NUMBER_PATTERN = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")
SMALL_INTEGER_LIMIT = 12
UNIT_SCALES = (1.0, 100.0, 1e-5, 1e-7)
IGNORED_PATTERNS = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"\bD-\d+\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.+]*)?\b"),
    re.compile(r"\bv\d+(?:\.\d+)*\b"),
)


@dataclass(frozen=True)
class SourceValues:
    """All numbers found in one source file.

    Attributes:
        name: File path relative to the project root, with forward slashes.
        values: Absolute values of every number in the file, in all accepted unit scales.
    """

    name: str
    values: np.ndarray


def numbers_in(value: object) -> list[float]:
    """Return every number inside a table cell, JSON/YAML leaf, list, dict or string.

    Args:
        value: Any scalar or nested container. Booleans and NaN contribute nothing.

    Returns:
        list[float]: The numbers found, in order.
    """
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [] if np.isnan(value) else [float(value)]
    if isinstance(value, dict):
        return [n for item in value.values() for n in numbers_in(item)]
    if isinstance(value, list | tuple):
        return [n for item in value for n in numbers_in(item)]
    return [float(m.replace(",", "")) for m in NUMBER_PATTERN.findall(str(value))]


def _scaled(values: Iterable[float]) -> np.ndarray:
    """Absolute values in every accepted unit scale (as is, percent, lakh, crore)."""
    base = np.abs(np.asarray(list(values), dtype=float))
    return np.concatenate([base * scale for scale in UNIT_SCALES])


def _relative_name(path: Path, root: Path) -> str:
    """Path relative to the project root with forward slashes, for stable reporting."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def table_source(path: Path, root: Path) -> SourceValues:
    """Read a CSV table (header and cells) into a SourceValues record."""
    table = pd.read_csv(path)
    found = numbers_in(list(table.columns))
    found += [n for cell in table.to_numpy().ravel() for n in numbers_in(cell)]
    return SourceValues(_relative_name(path, root), _scaled(found))


def structured_source(path: Path, root: Path) -> SourceValues:
    """Read a JSON or YAML file into a SourceValues record."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return SourceValues(_relative_name(path, root), _scaled(numbers_in(data)))


def load_sources(root: Path) -> list[SourceValues]:
    """Collect every allowed source, tables first so they are preferred when reporting.

    Args:
        root: Project root containing ``reports/``, ``models`` and ``configs``.

    Returns:
        list[SourceValues]: Tables, prediction files, model JSON files, then the config.
    """
    tables = sorted((root / "reports" / "tables").glob("*.csv"))
    tables += sorted((root / "reports" / "predictions").glob("*.csv"))
    model_files = sorted((root / "models").glob("*.json"))
    sources = [table_source(path, root) for path in tables]
    sources += [structured_source(path, root) for path in model_files]
    sources.append(structured_source(root / "configs" / "config.yaml", root))
    return sources


def strip_ignored(text: str) -> str:
    """Remove code, link targets, decision ids, dates and version tags before scanning."""
    for pattern in IGNORED_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def quoted_numbers(text: str) -> list[str]:
    """Number tokens in prose that need a source (small whole numbers and the appendix exempt)."""
    tokens = []
    for token in NUMBER_PATTERN.findall(strip_ignored(without_appendix(text))):
        value = float(token.replace(",", ""))
        if "." not in token and value <= SMALL_INTEGER_LIMIT:
            continue
        tokens.append(token)
    return tokens


def source_of(token: str, sources: list[SourceValues]) -> str | None:
    """Name of the first source containing the token at its written precision, else None."""
    value = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    for source in sources:
        if np.any(np.isclose(np.round(source.values, decimals), value, atol=1e-9)):
            return source.name
    return None


def unsupported_numbers(text: str, sources: list[SourceValues]) -> list[str]:
    """Number tokens in the text that match no source value."""
    return [token for token in quoted_numbers(text) if source_of(token, sources) is None]


def number_source_table(text: str, sources: list[SourceValues]) -> pd.DataFrame:
    """One row per distinct quoted number with the file it was matched to.

    Args:
        text: Markdown prose to scan.
        sources: Output of ``load_sources``.

    Returns:
        pd.DataFrame: Columns ``number`` and ``source_file`` (None when unsupported),
        in order of first appearance.
    """
    tokens = list(dict.fromkeys(quoted_numbers(text)))
    rows = [{"number": token, "source_file": source_of(token, sources)} for token in tokens]
    return pd.DataFrame(rows, columns=["number", "source_file"])


def notebook_markdown(path: Path) -> str:
    """Concatenated markdown cells of a Jupyter notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    return "\n".join("".join(cell["source"]) for cell in cells)


APPENDIX_START = "<!-- number-sources:start -->"
APPENDIX_END = "<!-- number-sources:end -->"


def without_appendix(text: str) -> str:
    """The document text with any generated number-source block removed."""
    if APPENDIX_START not in text:
        return text
    head, rest = text.split(APPENDIX_START, 1)
    return head + rest.split(APPENDIX_END, 1)[-1]


def appendix_markdown(table: pd.DataFrame) -> str:
    """Render the number-source table as a Markdown table between the appendix markers."""
    lines = [APPENDIX_START, "", "| Number as written | Source file |", "|---|---|"]
    for row in table.itertuples(index=False):
        source = f"`{row.source_file}`" if row.source_file else "**no source found**"
        lines.append(f"| {row.number} | {source} |")
    return "\n".join([*lines, "", APPENDIX_END])


def refresh_source_appendix(path: Path, root: Path) -> pd.DataFrame:
    """Rewrite the number-source block of a Markdown document in place.

    Args:
        path: Markdown file containing ``APPENDIX_START`` and ``APPENDIX_END`` markers.
        root: Project root used to load the sources.

    Returns:
        pd.DataFrame: The number-source table that was written.

    Raises:
        ValueError: if the markers are missing.
    """
    text = path.read_text(encoding="utf-8")
    if APPENDIX_START not in text or APPENDIX_END not in text:
        raise ValueError(f"{path} has no {APPENDIX_START} ... {APPENDIX_END} block")
    table = number_source_table(without_appendix(text), load_sources(root))
    head, rest = text.split(APPENDIX_START, 1)
    tail = rest.split(APPENDIX_END, 1)[1]
    path.write_text(head + appendix_markdown(table) + tail, encoding="utf-8")
    missing = int(table["source_file"].isna().sum())
    logger.info("%s: %d numbers, %d without a source", path, len(table), missing)
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or ["reports/final_report.md"]:
        refresh_source_appendix(Path(name), Path("."))
