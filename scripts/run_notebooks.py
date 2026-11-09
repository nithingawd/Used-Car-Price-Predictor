"""Re-execute the project notebooks in place with nbclient (python tasks.py notebooks).

Each notebook runs with ``notebooks/`` as its working directory, as in Jupyter. The notebooks
only read saved outputs and call ``src/carprice``, so run ``make all`` first.
"""

import logging
from pathlib import Path
import sys

from nbclient import NotebookClient
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
CELL_TIMEOUT_S = 900

logger = logging.getLogger("run_notebooks")


def execute_notebook(path: Path) -> None:
    """Execute one notebook top to bottom and save it with its new outputs."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    logger.info("executed %s", path.relative_to(PROJECT_ROOT))


def main() -> int:
    """Execute every numbered notebook in order."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in sorted(NOTEBOOK_DIR.glob("0*.ipynb")):
        execute_notebook(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
