"""Export reports/final_report.md to .docx and .pdf with pandoc (Phase 8, D-040).

Usage:
    python scripts/export_report.py            # or: python tasks.py report / make report

Word: pandoc converts the Markdown directly (table of contents, numbered sections, figures
embedded). PDF: no LaTeX engine is assumed, so pandoc first writes a self-contained HTML file
styled by ``scripts/report_print.css`` and headless Chrome or Edge prints it to PDF. The
pandoc executable comes from the ``pypandoc_binary`` package in requirements.txt; a system
pandoc on PATH works the same way (equivalent shell commands are in the README).
"""

import logging
from pathlib import Path
import subprocess
import sys
import tempfile

from capture_screenshots import find_browser
import pypandoc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
SOURCE = REPORTS_DIR / "final_report.md"
CSS = Path(__file__).resolve().parent / "report_print.css"
COMMON_ARGS = ["--toc", "--toc-depth=2", "--number-sections", f"--resource-path={REPORTS_DIR}"]
PRINT_TIMEOUT_S = 180

logger = logging.getLogger("export_report")


def export_docx(source: Path, target: Path) -> Path:
    """Convert the Markdown report to Word with pandoc."""
    pypandoc.convert_file(str(source), "docx", outputfile=str(target), extra_args=COMMON_ARGS)
    logger.info("wrote %s", target)
    return target


def export_html(source: Path, target: Path) -> Path:
    """Convert the Markdown report to one self-contained, print-styled HTML file."""
    args = [*COMMON_ARGS, "--standalone", "--embed-resources", "--mathml", f"--css={CSS}"]
    pypandoc.convert_file(str(source), "html5", outputfile=str(target), extra_args=args)
    return target


def print_pdf(html: Path, target: Path) -> Path:
    """Print an HTML file to PDF with headless Chrome or Edge.

    Raises:
        RuntimeError: if the browser exits with an error or writes no file.
    """
    command = [
        str(find_browser()),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={target}",
        html.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, timeout=PRINT_TIMEOUT_S, check=False)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"PDF printing failed: {result.stderr.decode(errors='replace')}")
    logger.info("wrote %s", target)
    return target


def main() -> int:
    """Export the report to .docx and .pdf next to the Markdown source."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not SOURCE.exists():
        logger.error("missing %s", SOURCE)
        return 1
    export_docx(SOURCE, SOURCE.with_suffix(".docx"))
    with tempfile.TemporaryDirectory() as folder:
        html = export_html(SOURCE, Path(folder) / "final_report.html")
        print_pdf(html, SOURCE.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
