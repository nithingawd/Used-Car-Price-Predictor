"""Cross-platform task runner for the project (the Makefile calls these same targets).

Usage: python tasks.py <target>. Run ``python tasks.py --help`` for the list of targets.
"""

import argparse
from collections.abc import Callable
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe" if os.name == "nt" else VENV_DIR / "bin" / "python"
)
CACHE_DIR_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache")
GENERATED_DATA_DIRS = ("data/interim", "data/processed")
KEEP_FILE_NAME = ".gitkeep"

logger = logging.getLogger("tasks")


def python_exe() -> str:
    """Return the project venv interpreter if it exists, else the current interpreter."""
    return str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)


def run_command(cmd: list[str]) -> int:
    """Run a command from the project root and return its exit code.

    Args:
        cmd: Command and arguments.

    Returns:
        int: Process return code.
    """
    logger.info("run: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def run_python_code(code: str, *args: str) -> int:
    """Run a Python snippet in the project interpreter with ``src/`` on the import path.

    Args:
        code: Python source executed after ``src`` is inserted into ``sys.path``.
        *args: Extra command-line arguments visible to the snippet as ``sys.argv[1:]``.

    Returns:
        int: Process return code.
    """
    prefix = "import sys; sys.path.insert(0, 'src'); "
    return run_command([python_exe(), "-c", prefix + code, *args])


def run_module(module: str, *args: str) -> int:
    """Run a ``carprice`` module as a script, e.g. ``carprice.train``.

    Args:
        module: Dotted module name.
        *args: Extra command-line arguments for the module.

    Returns:
        int: Process return code.
    """
    code = f"import runpy; runpy.run_module('{module}', run_name='__main__')"
    return run_python_code(code, *args)


def setup() -> int:
    """Create ``.venv`` (if missing) with the current interpreter and install requirements."""
    if not VENV_DIR.exists():
        logger.info("creating virtual environment at %s", VENV_DIR)
        code = run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
        if code != 0:
            return code
    return run_command([python_exe(), "-m", "pip", "install", "-r", "requirements.txt"])


def remove_cache_dirs() -> None:
    """Delete Python, pytest and ruff caches outside ``.venv``."""
    for name in CACHE_DIR_NAMES:
        for path in PROJECT_ROOT.rglob(name):
            if VENV_DIR.name not in path.parts and path.is_dir():
                logger.info("removing %s", path.relative_to(PROJECT_ROOT))
                shutil.rmtree(path, ignore_errors=True)


def remove_generated_data() -> None:
    """Empty ``data/interim`` and ``data/processed``; the raw CSV is never touched."""
    for folder in GENERATED_DATA_DIRS:
        folder_path = PROJECT_ROOT / folder
        if not folder_path.exists():
            continue
        for item in folder_path.iterdir():
            if item.name == KEEP_FILE_NAME:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()


def clean() -> int:
    """Remove caches and generated interim/processed data so the pipeline starts fresh."""
    remove_cache_dirs()
    remove_generated_data()
    return 0


def clean_data() -> int:
    """Phase 1: audit the raw CSV and write ``data/interim/cars_clean.csv``."""
    return run_python_code(
        "import logging; logging.basicConfig(level=logging.INFO, format='%(message)s'); "
        "from carprice.data import run_data_audit, run_cleaning; "
        "run_data_audit(); run_cleaning()"
    )


def eda() -> int:
    """Phase 2: regenerate EDA figures and tables (the notebook holds the narrative)."""
    return run_python_code("from carprice.viz import run_eda_pipeline; run_eda_pipeline()")


def split() -> int:
    """Phase 3: write the stratified train/test split and the measured feature checks."""
    return run_module("carprice.features")


def train() -> int:
    """Phase 4: CV benchmark, Optuna tuning, final pipeline and encoder comparison."""
    return run_module("carprice.train", "--stage", "all")


def evaluate() -> int:
    """Phase 5: score the locked test set once, then permutation importance and SHAP."""
    return run_module("carprice.evaluate", "--stage", "all")


def predict() -> int:
    """Phase 6: input schema, spec lookup, test predictions, worked examples, what-if, timing."""
    return run_module("carprice.prediction_reports")


def run_all() -> int:
    """Run the full pipeline (Phases 1 to 6) in order, stopping at the first failure."""
    for step in (clean_data, eda, split, train, evaluate, predict):
        code = step()
        if code != 0:
            logger.error("step %s failed with exit code %d", step.__name__, code)
            return code
    return 0


def app() -> int:
    """Launch the Streamlit web application."""
    return run_command([python_exe(), "-m", "streamlit", "run", "app/Home.py"])


def test() -> int:
    """Run the pytest suite."""
    return run_command([python_exe(), "-m", "pytest", "tests"])


def coverage() -> int:
    """Run the pytest suite with line coverage of ``src/carprice``."""
    return run_command(
        [python_exe(), "-m", "pytest", "tests", "-q", "--cov=src/carprice", "--cov-report=term"]
    )


def lint() -> int:
    """Run ``ruff check`` and ``ruff format --check`` over the whole repository."""
    check_code = run_command([python_exe(), "-m", "ruff", "check", "."])
    format_code = run_command([python_exe(), "-m", "ruff", "format", "--check", "."])
    return check_code or format_code


def report() -> int:
    """Export ``reports/final_report.md`` to .docx and .pdf (see scripts/export_report.py)."""
    return run_command([python_exe(), "scripts/export_report.py"])


TARGETS: dict[str, Callable[[], int]] = {
    "setup": setup,
    "clean": clean,
    "clean-data": clean_data,
    "eda": eda,
    "split": split,
    "train": train,
    "evaluate": evaluate,
    "predict": predict,
    "all": run_all,
    "app": app,
    "test": test,
    "coverage": coverage,
    "lint": lint,
    "report": report,
}


def main() -> None:
    """Parse the target name and run it, exiting with its return code."""
    logging.basicConfig(level=logging.INFO, format="[tasks] %(message)s")
    parser = argparse.ArgumentParser(description="Used Car Price Predictor task runner")
    parser.add_argument("target", choices=list(TARGETS), help="task to run")
    args = parser.parse_args()
    sys.exit(TARGETS[args.target]())


if __name__ == "__main__":
    main()
