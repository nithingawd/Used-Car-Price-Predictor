"""The only place the GUI's Python code takes colours from (A7, light and dark). CSS uses theme.css.

Chart colours are theme placeholders, not hex values (D-036). Streamlit replaces "#000001"
to "#000010" in a Plotly figure, in the browser, with slots 1 to 10 of the active theme's
`chartCategoricalColors` (.streamlit/config.toml, built from carprice.palette *_CHART_SLOTS).
So one figure draws in the A7 light palette or its dark counterpart, and switches instantly
when the visitor changes theme, without a rerun.
"""

from typing import Any

import plotly.graph_objects as go

from carprice.palette import CHART_SLOT_ROLES

__all__ = [
    "ACCENT_AMBER",
    "BORDER_COLOR",
    "CATEGORICAL_PALETTE",
    "GRID_COLOR",
    "INK_COLOR",
    "MUTED_TEXT",
    "PRIMARY_TEAL",
    "SLATE",
    "SURFACE_COLOR",
    "TEAL_TINT",
    "PLOTLY_CONFIG",
    "chart_slot",
    "style_figure",
]


def chart_slot(role: str) -> str:
    """Placeholder colour for a chart role ('teal', 'ink', ..., 'surface'); see CHART_SLOT_ROLES."""
    return f"#{CHART_SLOT_ROLES.index(role) + 1:06d}"


PRIMARY_TEAL = chart_slot("teal")
INK_COLOR = chart_slot("ink")
ACCENT_AMBER = chart_slot("amber")
SLATE = chart_slot("slate")
MUTED_TEXT = chart_slot("muted")
BORDER_COLOR = chart_slot("border")
GRID_COLOR = chart_slot("grid")
SURFACE_COLOR = chart_slot("surface")
CATEGORICAL_PALETTE = [chart_slot(role) for role in CHART_SLOT_ROLES[:6]]
# Translucent teal reads on both white and navy cards, so it needs no slot.
TEAL_TINT = "rgba(14, 124, 123, 0.14)"
FONT_UI = "Inter, system-ui, sans-serif"
FONT_SIZE = 14
PLOTLY_CONFIG: dict[str, Any] = {"displayModeBar": False, "responsive": True}


def style_figure(fig: go.Figure, height: int, show_legend: bool = False) -> go.Figure:
    """Apply the A7 chart style: Inter 14px, light grid, card-coloured plot, no chart junk."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font={"family": FONT_UI, "size": FONT_SIZE, "color": INK_COLOR},
        showlegend=show_legend,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "title": None},
        hoverlabel={
            "bgcolor": SURFACE_COLOR,
            "bordercolor": BORDER_COLOR,
            "font": {"family": FONT_UI, "size": FONT_SIZE, "color": INK_COLOR},
        },
        transition={"duration": 300, "easing": "cubic-in-out"},
    )
    axis = {
        "gridcolor": GRID_COLOR,
        "zeroline": False,
        "linecolor": BORDER_COLOR,
        "tickfont": {"color": MUTED_TEXT, "size": FONT_SIZE},
        "title": {"font": {"color": INK_COLOR, "size": FONT_SIZE}},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
