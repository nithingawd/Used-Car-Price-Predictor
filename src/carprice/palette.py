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
# Gridlines: the border colour at 70% opacity over white, flattened to a solid hex so it can be
# a theme colour slot.
GRID_COLOR: str = "#ECE8E0"

# ---------------------------------------------------------------------------
# Dark mode (GUI only; report figures stay light). Derived from A7, see D-036.
# Text tokens reach at least 4.5:1 on both dark backgrounds (tests/test_viz.py).
# ---------------------------------------------------------------------------
DARK_BG_COLOR: str = "#0F1626"  # deep ink-navy page
DARK_SURFACE_COLOR: str = "#18233A"  # cards, one step lighter than the page
DARK_INK_COLOR: str = "#EEEAE1"  # warm off-white text, echoes the light background
DARK_MUTED_TEXT: str = "#A4ACBA"
DARK_TEAL: str = "#3FB1AF"  # brightened teal for lines, links and markers
DARK_BORDER_COLOR: str = "#2C3852"
DARK_GRID_COLOR: str = "#243049"
DARK_POSITIVE_COLOR: str = "#5DBB63"
DARK_NEGATIVE_COLOR: str = "#E8695B"
DARK_CATEGORICAL_PALETTE: list[str] = [
    DARK_TEAL,
    DARK_INK_COLOR,  # the "ink" series flips to light text
    ACCENT_AMBER,  # amber already reads well on navy (7.6:1)
    "#7A9E9F",
    "#D08A45",  # ochre, lightened from #B5651D for contrast on navy
    "#8D99AE",
]

# Ten Plotly colour slots, same roles in both themes: the six categorical series, then
# muted text, border, gridlines and card surface. They are written into .streamlit/config.toml
# as chartCategoricalColors; the GUI draws with placeholders that Streamlit swaps for these.
CHART_SLOT_ROLES: list[str] = [
    "teal",
    "ink",
    "amber",
    "soft_teal",
    "ochre",
    "slate",
    "muted",
    "border",
    "grid",
    "surface",
]
LIGHT_CHART_SLOTS: list[str] = [
    *CATEGORICAL_PALETTE,
    MUTED_TEXT,
    BORDER_COLOR,
    GRID_COLOR,
    SURFACE_COLOR,
]
DARK_CHART_SLOTS: list[str] = [
    *DARK_CATEGORICAL_PALETTE,
    DARK_MUTED_TEXT,
    DARK_BORDER_COLOR,
    DARK_GRID_COLOR,
    DARK_SURFACE_COLOR,
]
