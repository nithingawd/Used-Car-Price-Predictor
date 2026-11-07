"""Upload checks for the batch tab, kept free of Streamlit calls so they can be tested."""

from dataclasses import dataclass, field
import io

import pandas as pd

from carprice.config import get_config
from carprice.predict import FIELD_LABELS

INPUT_FIELDS = list(FIELD_LABELS)


@dataclass(frozen=True)
class UploadProblem:
    """A plain-language reason an uploaded file cannot be used."""

    title: str
    message: str
    items: list[str] = field(default_factory=list)


def check_upload(content: bytes) -> tuple[pd.DataFrame | None, UploadProblem | None]:
    """Parse an uploaded CSV and check its shape before any row is priced.

    Returns:
        (frame, None) with exactly the CarSpec columns, or (None, problem).
    """
    max_rows = get_config().gui.batch_max_rows
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return None, UploadProblem(
            "Unreadable file.",
            "This file could not be read as a CSV. Save it as comma-separated values and try "
            "again.",
        )
    missing = [column for column in INPUT_FIELDS if column not in frame.columns]
    if missing:
        return None, UploadProblem(
            "Missing columns.",
            "The file is missing columns the model needs. Start from the template.",
            missing,
        )
    if frame.empty or len(frame) > max_rows:
        return None, UploadProblem(
            "Wrong number of rows.", f"The file must contain between 1 and {max_rows:,} cars."
        )
    return frame[INPUT_FIELDS], None
