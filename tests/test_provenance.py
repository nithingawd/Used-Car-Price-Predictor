"""Tests for provenance.py and report_facts.py, plus the number check on every written document."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carprice.config import get_config
from carprice.provenance import (
    SourceValues,
    appendix_markdown,
    load_sources,
    notebook_markdown,
    number_source_table,
    numbers_in,
    quoted_numbers,
    source_of,
    unsupported_numbers,
    without_appendix,
)
from carprice.report_facts import (
    build_key_figures,
    ratio,
    reduction_pct,
    share_pct,
)

TOY_SOURCES = [
    SourceValues("reports/tables/a.csv", np.array([94648.0, 0.5097, 550000.0, 5.5, 50.97])),
    SourceValues("configs/config.yaml", np.array([42.0, 500000.0])),
]

# Every prose document that quotes results. Markdown files and notebook markdown cells.
DOCUMENTS = [
    "reports/eda_findings.md",
    "reports/evaluation_summary.md",
    "reports/final_report.md",
    "reports/viva_notes.md",
    "reports/demo_script.md",
    "reports/requirements_traceability.md",
    "reports/reproducibility_check.md",
    "README.md",
]
NOTEBOOKS = [
    "notebooks/01_data_audit.ipynb",
    "notebooks/02_eda.ipynb",
    "notebooks/03_model_evaluation.ipynb",
]


def test_numbers_in_handles_nested_values() -> None:
    """Numbers are found in strings, dicts and lists; booleans and NaN are skipped."""
    value = {"a": [1, "₹ 5,50,000 and 0.25"], "b": True, "c": float("nan")}
    assert numbers_in(value) == [1.0, 550000.0, 0.25]


def test_strip_ignored_removes_code_links_ids_and_dates() -> None:
    """Inline code, link targets, decision ids, dates and versions are not scanned."""
    text = "See `eda_01.png` and [fig](reports/eda_22.png), D-014, 2026-11-08, v1.0, 777"
    assert quoted_numbers(text) == ["777"]


def test_small_whole_numbers_are_exempt() -> None:
    """List numbering and counts up to 12 need no source; decimals always do."""
    assert quoted_numbers("Phase 8, 5-fold, 12 cars, 13 cars, 0.5") == ["13", "0.5"]


def test_source_matches_at_written_precision_and_units() -> None:
    """A rounded value, a percent form and a lakh form all trace to their table."""
    assert source_of("94,648", TOY_SOURCES) == "reports/tables/a.csv"
    assert source_of("50.97", TOY_SOURCES) == "reports/tables/a.csv"
    assert source_of("5.50", TOY_SOURCES) == "reports/tables/a.csv"
    assert source_of("42", TOY_SOURCES) == "configs/config.yaml"
    assert source_of("94,649", TOY_SOURCES) is None


def test_unsupported_numbers_and_source_table() -> None:
    """Invented numbers are listed; the source table has one row per distinct number."""
    text = "MAE ₹ 94,648, 50.97% within 10%, seed 42, invented 123.45, again 94,648"
    assert unsupported_numbers(text, TOY_SOURCES) == ["123.45"]
    table = number_source_table(text, TOY_SOURCES)
    assert list(table["number"]) == ["94,648", "50.97", "42", "123.45"]
    assert table["source_file"].isna().sum() == 1


def test_share_ratio_and_reduction_helpers() -> None:
    """Hand-computed values for the key-figure helpers."""
    counts = pd.Series({"a": 1, "b": 3})
    assert share_pct(counts, "b") == 75.0
    assert ratio(1050000, 500000) == 2.1
    assert reduction_pct(115850, 94648) == 18.3


def test_key_figures_file_matches_recomputation() -> None:
    """The saved report_key_figures.csv equals a fresh computation from the saved tables."""
    config = get_config()
    saved = pd.read_csv(config.paths.tables_dir / "report_key_figures.csv")
    fresh = build_key_figures(config)
    assert list(saved["figure_id"]) == list(fresh["figure_id"])
    assert np.allclose(saved["value"], fresh["value"])


def test_key_figures_known_values() -> None:
    """Spot checks against values worked out by hand from the source tables."""
    figures = build_key_figures(get_config()).set_index("figure_id")["value"]
    assert figures["median_ratio_automatic_vs_manual"] == 2.1  # 10,50,000 / 5,00,000
    assert figures["test_mae_inr_reduction_vs_ridge_pct"] == 18.3  # (115850-94648)/115850
    assert figures["share_top_two_brands_pct"] == 51.73  # (4933+2952)/15244


@pytest.mark.parametrize("name", DOCUMENTS)
def test_document_numbers_have_a_source(name: str) -> None:
    """Every number quoted in a written document exists in a saved table, model file or config."""
    path = Path(name)
    assert path.exists(), f"{name} is missing"
    text = path.read_text(encoding="utf-8")
    assert unsupported_numbers(text, load_sources(Path("."))) == []


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_markdown_numbers_have_a_source(name: str) -> None:
    """Every number in a notebook's markdown cells exists in a saved output file."""
    text = notebook_markdown(Path(name))
    assert unsupported_numbers(text, load_sources(Path("."))) == []


def test_final_report_appendix_is_current_and_fully_cited() -> None:
    """Appendix D lists every number with a source the report itself cites nearby."""
    text = Path("reports/final_report.md").read_text(encoding="utf-8")
    table = number_source_table(without_appendix(text), load_sources(Path(".")))
    assert appendix_markdown(table) in text, "run: python -m carprice.provenance"
    assert table["source_file"].notna().all()
    assert (table["matched_by"] == "cited").all()


def test_citation_is_preferred_over_first_match() -> None:
    """When two files hold a value, the one cited beside it wins; uncited numbers search."""
    sources = [
        SourceValues("reports/tables/first.csv", np.array([7.25, 13.0])),
        SourceValues("configs/config.yaml", np.array([7.25])),
    ]
    text = "\n\n".join(["Threshold 7.25 [`config.yaml`].", "Count 13 with no citation."])
    table = number_source_table(text, sources)
    assert list(table["source_file"]) == ["configs/config.yaml", "reports/tables/first.csv"]
    assert list(table["matched_by"]) == ["cited", "search"]


def test_volatile_measurements_are_not_sources(tmp_path: Path) -> None:
    """Run times, timing tables and timestamps can neither support nor be quoted numbers."""
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    pd.DataFrame({"model_id": ["a"], "rmse": [0.1654], "fit_time_sec": [0.5098]}).to_csv(
        tables / "cv_results.csv", index=False
    )
    pd.DataFrame({"single_median_ms": [27.221]}).to_csv(
        tables / "predictor_timing.csv", index=False
    )
    sources = load_sources(tmp_path)
    assert unsupported_numbers("CV 0.1654, fit 50.98%, median 27.221 ms", sources) == [
        "50.98",
        "27.221",
    ]


def test_title_block_is_not_scanned() -> None:
    """Author USNs in the YAML title block are metadata, not quoted results."""
    text = '---\nauthor:\n  - "Student - 1RV24CS174"\n---\n\nScore 777.'
    assert quoted_numbers(text) == ["777"]


def test_title_block_is_skipped_in_the_appendix() -> None:
    """The appendix scans paragraph by paragraph; the title block must still be skipped."""
    text = '---\nauthor:\n  - "Student - 1RV24CS176"\n---\n\nScore 0.25 [`config.yaml`].'
    table = number_source_table(text, TOY_SOURCES)
    assert list(table["number"]) == ["0.25"]
