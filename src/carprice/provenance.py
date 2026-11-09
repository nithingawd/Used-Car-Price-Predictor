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
# Measurements that change on every run (wall-clock times, timestamps, the git commit) are
# not reproducible results, so they can neither support a quoted number nor be quoted.
VOLATILE_FILES = frozenset({"predictor_timing.csv"})
VOLATILE_COLUMNS = frozenset(
    {"fit_time_sec", "pred_time_sec", "datetime_start", "datetime_complete", "duration"}
)
VOLATILE_KEYS = frozenset({"timestamp_utc", "git_commit"})


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
    table = table.drop(columns=[c for c in table.columns if c in VOLATILE_COLUMNS])
    found = numbers_in(list(table.columns))
    found += [n for cell in table.to_numpy().ravel() for n in numbers_in(cell)]
    return SourceValues(_relative_name(path, root), _scaled(found))


def structured_source(path: Path, root: Path) -> SourceValues:
    """Read a JSON or YAML file into a SourceValues record."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if isinstance(data, dict):
        data = {key: value for key, value in data.items() if key not in VOLATILE_KEYS}
    return SourceValues(_relative_name(path, root), _scaled(numbers_in(data)))


def load_sources(root: Path) -> list[SourceValues]:
    """Collect every allowed source, tables first so they are preferred when reporting.

    Args:
        root: Project root containing ``reports/``, ``models`` and ``configs``.

    Returns:
        list[SourceValues]: Tables, prediction files, model JSON files, then the config;
        volatile files, columns and keys (run times, timestamps, commit) are left out.
    """
    tables = sorted((root / "reports" / "tables").glob("*.csv"))
    tables += sorted((root / "reports" / "predictions").glob("*.csv"))
    tables = [path for path in tables if path.name not in VOLATILE_FILES]
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


CITATION_PATTERN = re.compile(r"`([^`\s]+\.(?:csv|json|yaml))`")
BLOCK_SEPARATOR = re.compile(r"\n\s*\n")


def cited_files(block: str) -> list[str]:
    """File names cited in backticks in a block of Markdown, e.g. `test_metrics.csv`."""
    return CITATION_PATTERN.findall(block)


def blocks_with_context(text: str) -> list[tuple[str, list[str]]]:
    """Paragraph blocks, each with the files cited in it or in a following table caption.

    A Markdown table's caption (a block starting with ": ") cites the files for the
    table's numbers, so it is attached to the block before it.
    """
    blocks = [block for block in strip_appendix_blocks(text) if block.strip()]
    result = []
    for i, block in enumerate(blocks):
        cited = cited_files(block)
        if i + 1 < len(blocks) and blocks[i + 1].lstrip().startswith(": "):
            cited += cited_files(blocks[i + 1])
        result.append((block, cited))
    return result


def strip_appendix_blocks(text: str) -> list[str]:
    """The document without its generated appendix, split into blank-line separated blocks."""
    return BLOCK_SEPARATOR.split(without_appendix(text))


def cited_source_of(token: str, sources: list[SourceValues], cited: list[str]) -> str | None:
    """First source among the cited files that contains the token, else None."""
    preferred = [s for s in sources if any(s.name.endswith(name) for name in cited)]
    return source_of(token, preferred)


def number_source_table(text: str, sources: list[SourceValues]) -> pd.DataFrame:
    """One row per distinct quoted number with the file it was matched to.

    A number is attributed to a file cited in the same paragraph (or table caption) when
    that file contains it ("cited"); otherwise to the first file that contains it
    ("search"), which shows the value exists but may not be its intended source.

    Args:
        text: Markdown prose to scan.
        sources: Output of ``load_sources``.

    Returns:
        pd.DataFrame: Columns ``number``, ``source_file`` (None when unsupported) and
        ``matched_by``, in order of first appearance.
    """
    rows: dict[str, dict[str, object]] = {}
    for block, cited in blocks_with_context(text):
        for token in quoted_numbers(block):
            if token in rows:
                continue
            source = cited_source_of(token, sources, cited)
            matched_by = "cited"
            if source is None:
                source, matched_by = source_of(token, sources), "search"
            rows[token] = {"number": token, "source_file": source, "matched_by": matched_by}
    return pd.DataFrame(list(rows.values()), columns=["number", "source_file", "matched_by"])


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


APPENDIX_PAIRS_PER_ROW = 2


def _source_cell(row: pd.Series) -> str:
    """Short source label for the appendix; a trailing * marks a search match."""
    if not row["source_file"]:
        return "**no source found**"
    name = str(row["source_file"]).removeprefix("reports/tables/")
    return f"`{name}`" + ("" if row["matched_by"] == "cited" else " *")


def appendix_markdown(table: pd.DataFrame) -> str:
    """Render the number-source table compactly, several number/source pairs per row."""
    pairs = [(str(row["number"]), _source_cell(row)) for _, row in table.iterrows()]
    width = APPENDIX_PAIRS_PER_ROW
    header = "| " + " | ".join(["Number", "Source"] * width) + " |"
    rule = "|" + "---|" * (2 * width)
    lines = [APPENDIX_START, "", header, rule]
    for start in range(0, len(pairs), width):
        chunk = pairs[start : start + width]
        chunk += [("", "")] * (width - len(chunk))
        lines.append("| " + " | ".join(cell for pair in chunk for cell in pair) + " |")
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
