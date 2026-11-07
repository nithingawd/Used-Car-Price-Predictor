"""The only place the GUI's Python code takes colours from (A7). CSS uses theme.css."""

from typing import Any

import plotly.graph_objects as go

from carprice.palette import (
    ACCENT_AMBER,
    BG_COLOR,
    BORDER_COLOR,
    CATEGORICAL_PALETTE,
    INK_COLOR,
    MUTED_TEXT,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    PRIMARY_TEAL,
    SLATE,
    SURFACE_COLOR,
)

__all__ = [
    "ACCENT_AMBER",
    "BG_COLOR",
    "BORDER_COLOR",
    "CATEGORICAL_PALETTE",
    "INK_COLOR",
    "MUTED_TEXT",
    "NEGATIVE_COLOR",
    "POSITIVE_COLOR",
    "PRIMARY_TEAL",
    "SLATE",
    "SURFACE_COLOR",
    "TEAL_TINT",
    "PLOTLY_CONFIG",
    "style_figure",
]

TEAL_TINT = "rgba(14, 124, 123, 0.14)"
GRID_COLOR = "rgba(227, 222, 211, 0.7)"
FONT_UI = "Inter, system-ui, sans-serif"
FONT_SIZE = 14
PLOTLY_CONFIG: dict[str, Any] = {"displayModeBar": False, "responsive": True}


def style_figure(fig: go.Figure, height: int, show_legend: bool = False) -> go.Figure:
    """Apply the A7 chart style: Inter 14px, light grid, white plot, no chart junk."""
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
