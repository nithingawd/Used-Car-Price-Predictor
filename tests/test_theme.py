"""Light and dark GUI themes (D-036): config, CSS and chart colours stay in sync and readable."""

import json
from pathlib import Path
import re
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from carprice import palette  # noqa: E402
from components.palette import chart_slot, style_figure  # noqa: E402
from components.ui import range_bar  # noqa: E402

THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
CSS = (ROOT / "app" / "components" / "theme.css").read_text(encoding="utf-8")
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
TEXT_CONTRAST_MIN = 4.5  # WCAG AA body text (A7)
GRAPHIC_CONTRAST_MIN = 3.0  # WCAG 1.4.11 non-text graphics (chart marks)


def contrast(first: str, second: str) -> float:
    """WCAG contrast ratio between two #RRGGBB colours."""

    def luminance(colour: str) -> float:
        """WCAG relative luminance of one #RRGGBB colour."""
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted([luminance(first), luminance(second)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_light_theme_is_the_a7_palette() -> None:
    """The light theme in config.toml uses the exact A7 colours."""
    light = THEME["light"]
    assert light["backgroundColor"] == palette.BG_COLOR
    assert light["secondaryBackgroundColor"] == palette.SURFACE_COLOR
    assert light["textColor"] == palette.INK_COLOR
    assert light["primaryColor"] == palette.PRIMARY_TEAL
    assert light["chartCategoricalColors"] == palette.LIGHT_CHART_SLOTS
    assert light["chartCategoricalColors"][:6] == palette.CATEGORICAL_PALETTE


def test_dark_theme_matches_palette_module() -> None:
    """The dark theme in config.toml matches palette.py and follows the system setting."""
    dark = THEME["dark"]
    assert dark["backgroundColor"] == palette.DARK_BG_COLOR
    assert dark["secondaryBackgroundColor"] == palette.DARK_SURFACE_COLOR
    assert dark["textColor"] == palette.DARK_INK_COLOR
    assert dark["borderColor"] == palette.DARK_BORDER_COLOR
    assert dark["chartCategoricalColors"] == palette.DARK_CHART_SLOTS
    assert "base" not in THEME, "a fixed base would stop the app following the system theme"


def test_slot_lists_cover_every_role() -> None:
    """Both chart slot lists have one colour per role."""
    assert len(palette.LIGHT_CHART_SLOTS) == len(palette.CHART_SLOT_ROLES) == 10
    assert len(palette.DARK_CHART_SLOTS) == len(palette.CHART_SLOT_ROLES)


@pytest.mark.parametrize(
    "text", [palette.DARK_INK_COLOR, palette.DARK_MUTED_TEXT, palette.DARK_TEAL]
)
@pytest.mark.parametrize("background", [palette.DARK_BG_COLOR, palette.DARK_SURFACE_COLOR])
def test_dark_text_contrast(text: str, background: str) -> None:
    """Dark-mode text colours reach 4.5:1 on the page and on cards."""
    assert contrast(text, background) >= TEXT_CONTRAST_MIN


def test_dark_negative_readable_on_cards() -> None:
    """The dark error colour is readable as text on cards."""
    assert contrast(palette.DARK_NEGATIVE_COLOR, palette.DARK_SURFACE_COLOR) >= TEXT_CONTRAST_MIN


@pytest.mark.parametrize("series", palette.DARK_CATEGORICAL_PALETTE)
def test_dark_chart_series_visible_on_cards(series: str) -> None:
    """Each dark chart series reaches 3:1 against the card colour."""
    assert contrast(series, palette.DARK_SURFACE_COLOR) >= GRAPHIC_CONTRAST_MIN


def test_button_text_readable_in_both_themes() -> None:
    """White button text is readable on the primary colour in both themes."""
    for mode in ("light", "dark"):
        assert contrast("#FFFFFF", THEME[mode]["primaryColor"]) >= TEXT_CONTRAST_MIN


def test_css_dark_tokens_match_palette() -> None:
    """theme.css light-dark() tokens use the palette.py values."""
    pairs = {
        "--cp-bg": (palette.BG_COLOR, palette.DARK_BG_COLOR),
        "--cp-surface": (palette.SURFACE_COLOR, palette.DARK_SURFACE_COLOR),
        "--cp-ink": (palette.INK_COLOR, palette.DARK_INK_COLOR),
        "--cp-muted": (palette.MUTED_TEXT, palette.DARK_MUTED_TEXT),
        "--cp-teal": (palette.PRIMARY_TEAL, palette.DARK_TEAL),
        "--cp-border": (palette.BORDER_COLOR, palette.DARK_BORDER_COLOR),
        "--cp-negative": (palette.NEGATIVE_COLOR, palette.DARK_NEGATIVE_COLOR),
    }
    for token, (light, dark) in pairs.items():
        assert f"{token}: light-dark({light}, {dark});" in CSS, token


def test_chart_slots_are_streamlit_placeholders() -> None:
    """Chart slot names map to Streamlit's #0000NN colour placeholders."""
    assert chart_slot("teal") == "#000001"
    assert chart_slot("surface") == "#000010"


def test_figures_use_only_theme_placeholders() -> None:
    """Literal hex colours would not follow the theme; only #000001-#000010 may appear."""
    figure = style_figure(range_bar(400000, 500000, 600000), 96).to_plotly_json()
    figure["layout"].pop("template", None)  # Streamlit's own template, themed by Streamlit
    colours = set(HEX.findall(json.dumps(figure, default=str)))
    allowed = {chart_slot(role) for role in palette.CHART_SLOT_ROLES}
    assert colours, "expected the figure to carry colour placeholders"
    assert colours <= allowed
