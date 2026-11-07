"""The A7 palette: the single source of colour values for static charts and the GUI.

Kept dependency-free so the Streamlit app can import it without loading matplotlib.
"""

BG_COLOR: str = "#F7F5F0"
SURFACE_COLOR: str = "#FFFFFF"
INK_COLOR: str = "#14213D"
MUTED_TEXT: str = "#6B7280"
PRIMARY_TEAL: str = "#0E7C7B"
ACCENT_AMBER: str = "#F2A541"
BORDER_COLOR: str = "#E3DED3"
POSITIVE_COLOR: str = "#2E7D32"
NEGATIVE_COLOR: str = "#C0392B"

CATEGORICAL_PALETTE: list[str] = [
    "#0E7C7B",  # Primary teal
    "#14213D",  # Ink
    "#F2A541",  # Amber accent
    "#7A9E9F",  # Soft teal
    "#B5651D",  # Ochre
    "#8D99AE",  # Muted slate ("other" / de-emphasised)
]
SLATE: str = CATEGORICAL_PALETTE[5]
SEQUENTIAL_STOPS: list[str] = ["#FFFFFF", "#0E7C7B"]
