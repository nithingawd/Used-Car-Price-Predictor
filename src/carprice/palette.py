"""Colour values for the static report figures (A7) and the Streamlit GUI (D-045).

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
# GUI theme (D-045): zinc neutrals, one blue action colour, amber as the single highlight
# per view. Report figures above keep the A7 palette they were published with.
# Text tokens reach at least 4.5:1 on page and cards in both modes (tests/test_theme.py).
# ---------------------------------------------------------------------------
GUI_BG_COLOR: str = "#FAFAFA"
GUI_SURFACE_COLOR: str = "#FFFFFF"
GUI_INK_COLOR: str = "#18181B"
GUI_MUTED_TEXT: str = "#71717A"
GUI_PRIMARY: str = "#2F5BEA"
GUI_HIGHLIGHT: str = "#D97706"
GUI_BORDER_COLOR: str = "#E4E4E7"
GUI_GRID_COLOR: str = "#F0F0F2"
GUI_POSITIVE_COLOR: str = "#15803D"
GUI_NEGATIVE_COLOR: str = "#DC2626"
GUI_CATEGORICAL_PALETTE: list[str] = [
    GUI_PRIMARY,
    GUI_INK_COLOR,
    GUI_HIGHLIGHT,
    "#0D9488",  # teal, fourth series
    "#BE185D",  # rose, fifth series
    "#8B8B94",  # neutral for "other" and de-emphasised marks
]

DARK_BG_COLOR: str = "#0B0B0D"
DARK_SURFACE_COLOR: str = "#141417"
DARK_INK_COLOR: str = "#EDEDEF"
DARK_MUTED_TEXT: str = "#A1A1AA"
DARK_PRIMARY: str = "#7C9BFF"  # lines, links and selected text; buttons keep GUI_PRIMARY
DARK_HIGHLIGHT: str = "#F59E0B"
DARK_BORDER_COLOR: str = "#27272A"
DARK_GRID_COLOR: str = "#1E1E22"
DARK_POSITIVE_COLOR: str = "#4ADE80"
DARK_NEGATIVE_COLOR: str = "#F87171"
DARK_CATEGORICAL_PALETTE: list[str] = [
    DARK_PRIMARY,
    DARK_INK_COLOR,
    DARK_HIGHLIGHT,
    "#2DD4BF",
    "#F472B6",
    "#71717A",
]

# Ten Plotly colour slots, same roles in both themes: the six categorical series, then
# muted text, border, gridlines and card surface. They are written into .streamlit/config.toml
# as chartCategoricalColors; the GUI draws with placeholders that Streamlit swaps for these.
CHART_SLOT_ROLES: list[str] = [
    "primary",
    "ink",
    "highlight",
    "series4",
    "series5",
    "slate",
    "muted",
    "border",
    "grid",
    "surface",
]
LIGHT_CHART_SLOTS: list[str] = [
    *GUI_CATEGORICAL_PALETTE,
    GUI_MUTED_TEXT,
    GUI_BORDER_COLOR,
    GUI_GRID_COLOR,
    GUI_SURFACE_COLOR,
]
DARK_CHART_SLOTS: list[str] = [
    *DARK_CATEGORICAL_PALETTE,
    DARK_MUTED_TEXT,
    DARK_BORDER_COLOR,
    DARK_GRID_COLOR,
    DARK_SURFACE_COLOR,
]
