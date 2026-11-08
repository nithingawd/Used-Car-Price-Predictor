"""Reusable presentation components. Pages build their UI from these only (7.1)."""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from html import escape
from pathlib import Path
import re
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from carprice.formatting import format_inr_indian, format_lakh
from components.palette import (
    ACCENT_AMBER,
    BORDER_COLOR,
    INK_COLOR,
    MUTED_TEXT,
    PLOTLY_CONFIG,
    TEAL_TINT,
    style_figure,
)

__all__ = [
    "chart_card",
    "empty_state",
    "format_inr",
    "format_lakh",
    "info_note",
    "load_theme",
    "metric_card",
    "page_header",
    "panel",
    "range_bar",
    "section_header",
    "safely",
]

THEME_CSS = Path(__file__).with_name("theme.css")
NOTE_KINDS = {
    "info": "cp-note",
    "caution": "cp-note cp-note-caution",
    "error": "cp-note cp-note-error",
}
RANGE_BAR_HEIGHT = 96


def format_inr(value: float) -> str:
    """₹ with Indian digit grouping, e.g. 550000 -> '₹ 5,50,000' (A7)."""
    return format_inr_indian(value)


@st.cache_data(show_spinner=False)
def _theme_css() -> str:
    """Read theme.css once per server process."""
    return THEME_CSS.read_text(encoding="utf-8")


def load_theme() -> None:
    """Inject the A7 stylesheet (called once per run by the app entry point)."""
    st.html(f"<style>{_theme_css()}</style>")


def page_header(title: str, lede: str) -> None:
    """The page's single H1 plus a one-sentence introduction."""
    st.html(f'<h1 class="cp-title">{escape(title)}</h1><p class="cp-lede">{escape(lede)}</p>')


def section_header(text: str) -> None:
    """A short sentence-case section label with a hairline underneath."""
    st.html(f'<p class="cp-section">{escape(text)}</p>')


def metric_card(label: str, value: str, caption: str) -> None:
    """A big number with its label above and a muted plain-English caption below."""
    st.html(
        f'<div class="cp-card"><p class="cp-metric-label">{escape(label)}</p>'
        f'<p class="cp-metric-value">{escape(value)}</p>'
        f'<p class="cp-caption">{escape(caption)}</p></div>'
    )


def info_note(
    message: str, kind: str = "info", title: str | None = None, items: Iterable[str] = ()
) -> None:
    """A bordered note: 'info' (teal), 'caution' or 'error' (red edge, A7 negative colour)."""
    heading = f"<strong>{escape(title)}</strong> " if title else ""
    bullets = "".join(f"<li>{escape(item)}</li>" for item in items)
    listing = f"<ul>{bullets}</ul>" if bullets else ""
    st.html(
        f'<div class="{NOTE_KINDS[kind]}" role="note">{heading}{escape(message)}{listing}</div>'
    )


def empty_state(title: str, message: str) -> None:
    """A calm placeholder card for 'nothing to show yet'."""
    st.html(
        f'<div class="cp-card cp-empty"><p class="cp-section">{escape(title)}</p>'
        f'<p class="cp-caption">{escape(message)}</p></div>'
    )


def panel(name: str) -> AbstractContextManager[Any]:
    """A bordered card container. Its key adds the class `st-key-card-<name>`, which theme.css
    uses to give every card the surface colour in light and dark mode (D-036).
    `name` must be unique on the page."""
    return st.container(border=True, key=f"card-{name}")


def chart_card(fig: go.Figure, title: str, takeaway: str) -> None:
    """A surface card holding a finding-style title, the chart and its text takeaway (A7)."""
    with panel(re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")):
        section_header(title)
        st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")
        st.html(f'<p class="cp-caption">{escape(takeaway)}</p>')


def range_bar(lower: float, estimate: float, upper: float) -> go.Figure:
    """The 80% range as a teal band with the estimate as the page's one amber marker."""
    pad = (upper - lower) * 0.35
    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=lower - pad,
        x1=upper + pad,
        y0=0,
        y1=0,
        line={"color": BORDER_COLOR, "width": 2},
    )
    fig.add_shape(
        type="rect",
        x0=lower,
        x1=upper,
        y0=-0.35,
        y1=0.35,
        fillcolor=TEAL_TINT,
        line={"width": 0},
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[0, 0],
            mode="markers",
            marker={"color": INK_COLOR, "size": 14, "symbol": "line-ns-open", "line": {"width": 2}},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[-0.95, -0.95],
            mode="text",
            text=[f"Low {format_lakh(lower)}", f"High {format_lakh(upper)}"],
            textfont={"color": MUTED_TEXT},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[estimate],
            y=[0],
            mode="markers",
            hovertemplate=f"Estimate {format_inr(estimate)}<extra></extra>",
            marker={
                "color": ACCENT_AMBER,
                "size": 18,
                "symbol": "diamond",
                "line": {"color": INK_COLOR, "width": 1.5},
            },
        )
    )
    style_figure(fig, RANGE_BAR_HEIGHT)
    fig.update_xaxes(visible=False, range=[lower - pad, upper + pad])
    fig.update_yaxes(visible=False, range=[-1.4, 0.8])
    return fig


def safely(render: Callable[[], Any], what: str) -> Any:
    """Run a render step; on a missing file or bad data show a plain message, never a traceback."""
    try:
        return render()
    except FileNotFoundError as err:
        info_note(f"{what} is not available yet: {err}", kind="error", title="Missing file.")
    except (KeyError, ValueError) as err:
        info_note(
            f"{what} could not be shown because the data looks wrong ({err}).",
            kind="error",
            title="Something went wrong.",
        )
    return None
