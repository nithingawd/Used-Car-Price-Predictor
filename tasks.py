"""Project automation tasks for Windows and cross-platform execution."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = (
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else PROJECT_ROOT / ".venv" / "bin" / "python"
)


def run_command(cmd: list[str]) -> int:
    """Execute a system command and return the exit code.

    Args:
        cmd: Command and arguments list.

    Returns:
        int: Process return code.
    """
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def setup() -> int:
    """Create virtual environment and install requirements."""
    if not (PROJECT_ROOT / ".venv").exists():
        print("Creating virtual environment at .venv...")
        ret = subprocess.run([sys.executable, "-m", "venv", str(PROJECT_ROOT / ".venv")]).returncode
        if ret != 0:
            return ret

    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    print("Installing requirements...")
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        return run_command([py, "-m", "pip", "install", "-r", str(req_file)])
    return 0


def clean() -> int:
    """Clean interim artifacts, caches, and generated files (raw data is preserved)."""
    print("Cleaning generated files and cache directories...")
    patterns = [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    ]
    for pattern in patterns:
        for path in PROJECT_ROOT.rglob(pattern):
            if ".venv" in path.parts:
                continue
            if path.is_dir():
                print(f"Removing {path}")
                import shutil

                shutil.rmtree(path, ignore_errors=True)

    # Clean interim and processed data files (preserve .gitkeep and raw data)
    for folder in ["data/interim", "data/processed"]:
        folder_path = PROJECT_ROOT / folder
        if folder_path.exists():
            for item in folder_path.iterdir():
                if item.name != ".gitkeep":
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil

                        shutil.rmtree(item, ignore_errors=True)
    return 0


def clean_data() -> int:
    """Run data audit and cleaning pipeline from raw data."""
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from carprice.data import run_data_audit, run_cleaning; "
        "run_data_audit(); run_cleaning()"
    )
    return run_command([py, "-c", code])


def test() -> int:
    """Run test suite using pytest."""
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    return run_command([py, "-m", "pytest", "tests"])


def lint() -> int:
    """Run ruff linter and formatting checks."""
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    print("Running ruff check...")
    ret_check = run_command([py, "-m", "ruff", "check", "src", "tests"])
    print("Running ruff format check...")
    ret_format = run_command([py, "-m", "ruff", "format", "--check", "src", "tests"])
    return ret_check or ret_format


def run_module(module: str, *args: str) -> int:
    """Run a carprice module as a script with src/ on the import path.

    Args:
        module: Dotted module name, e.g. 'carprice.train'.
        *args: Extra command-line arguments for the module.

    Returns:
        int: Process return code.
    """
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    code = (
        "import sys, runpy; sys.path.insert(0, 'src'); "
        f"runpy.run_module('{module}', run_name='__main__')"
    )
    return run_command([py, "-c", code, *args])


def eda() -> int:
    """Phase 2: regenerate EDA figures and tables (the notebook holds the narrative)."""
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from carprice.viz import run_eda_pipeline; run_eda_pipeline()"
    )
    return run_command([py, "-c", code])


def split() -> int:
    """Phase 3: write the stratified train/test split and the measured feature checks."""
    return run_module("carprice.features")


def train() -> int:
    """Phase 4: CV benchmark, Optuna tuning, final pipeline, and encoder comparison."""
    return run_module("carprice.train", "--stage", "all")


def evaluate() -> int:
    """Phase 5: score the locked test set once, then permutation importance and SHAP."""
    return run_module("carprice.evaluate", "--stage", "all")


def predict() -> int:
    """Phase 6: input schema, spec lookup, test predictions, worked examples, what-if, timing."""
    return run_module("carprice.prediction_reports")


def app() -> int:
    """Launch Streamlit web application."""
    py = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    return run_command([py, "-m", "streamlit", "run", "app/Home.py"])


def main() -> None:
    """Command-line entry point for project tasks."""
    parser = argparse.ArgumentParser(description="Used Car Price Predictor task runner")
    parser.add_argument(
        "target",
        choices=[
            "setup",
            "clean",
            "clean-data",
            "eda",
            "split",
            "train",
            "evaluate",
            "predict",
            "app",
            "test",
            "lint",
        ],
        help="Task target to execute",
    )
    args = parser.parse_args()

    targets = {
        "setup": setup,
        "clean": clean,
        "clean-data": clean_data,
        "eda": eda,
        "split": split,
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
        "app": app,
        "test": test,
        "lint": lint,
    }

    exit_code = targets[args.target]()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
