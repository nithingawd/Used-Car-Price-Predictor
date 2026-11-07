"""Reusable Streamlit components for the Used Car Price Predictor GUI (Phase 7).

Importing this package puts ``src/`` on the import path so pages can use ``carprice``.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
