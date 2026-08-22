"""Tactical console theme for the Smart Scan dashboard.

The visual language follows the Code-Raptors SIH 2026 front end: a dark operations
console with a dotted grid ground, monospace type, neon-green readouts and cyan accents.
Everything visual lives here so the app module stays about the data, and so the palette
is defined once rather than repeated as literals across a few hundred lines of markup.

Colour roles, rather than colour names, are what the rest of the code refers to:

* :data:`ACCENT_HIT` -- a successful intercept, and every positive readout.
* :data:`ACCENT_INFO` -- headings, table headers, the score callout.
* :data:`ACCENT_MISS` -- a failed scan.
* :data:`TRUTH` -- ground truth emitter activity, deliberately muted so the receiver's
  own path reads on top of it rather than competing with it.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# --- palette -------------------------------------------------------------------------
BACKGROUND = "#0E1117"
PANEL = "#151B28"
GRID = "#2C374A"
TEXT_MUTED = "#8A9BB3"
TEXT_BRIGHT = "#C8D1E0"

ACCENT_HIT = "#00FF43"
ACCENT_INFO = "#00D2FF"
ACCENT_MISS = "#FF2A2A"
ACCENT_NEUTRAL = "#4B5563"
TRUTH = "#568288"
ACCENT_WARN = "#FFB020"

MONO = "'Courier New', Courier, monospace"

#: Grade colours for the scorecard badges, keyed by letter.
GRADE_COLOURS = {
    "A": ACCENT_HIT,
    "B": "#7FE05B",
    "C": ACCENT_WARN,
    "D": "#FF7A2A",
    "E": ACCENT_MISS,
    "-": TEXT_MUTED,
}

_CSS = f"""
<style>
    .block-container {{ padding-top: 2rem !important; }}

    /* Dotted grid ground, the signature of the reference console. */
    .stApp {{
        background-color: {BACKGROUND};
        background-image: radial-gradient({GRID} 1px, transparent 0);
        background-size: 30px 30px;
        background-position: -15px -15px;
    }}

    [data-testid="stMetricValue"] {{
        font-family: {MONO};
        color: {ACCENT_HIT};
        text-shadow: 0px 0px 4px rgba(0, 255, 67, 0.4);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 0.8rem;
    }}
    [data-testid="stMetricDelta"] {{ font-family: {MONO}; }}

    hr {{ border-color: {GRID}; }}
    th {{
        text-transform: uppercase;
        font-size: 0.85rem;
        color: {ACCENT_INFO} !important;
    }}

    h4 {{
        font-family: {MONO};
        letter-spacing: 1.6px;
        color: {TEXT_BRIGHT};
        text-transform: uppercase;
    }}

    /* Sidebar reads as an instrument panel rather than a web form. */
    [data-testid="stSidebar"] {{
        background-color: {PANEL};
        border-right: 1px solid {GRID};
    }}
    [data-testid="stSidebar"] label {{
        color: {TEXT_MUTED} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.78rem;
    }}

    .stExpander {{ border: 1px solid {GRID} !important; border-radius: 6px; }}
    .stProgress > div > div > div > div {{ background-color: {ACCENT_HIT}; }}
</style>
"""


def inject_css() -> None:
    """Apply the console stylesheet. Call once, immediately after ``set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    """Render the banner header."""
    st.markdown(
        f"""
        <div style="background-color:{PANEL};padding:20px;border-radius:8px;
                    margin-bottom:20px;text-align:center;border:1px solid {GRID};">
          <h1 style="font-family:{MONO};letter-spacing:2px;font-size:2.2rem;margin:0;
                     color:{TEXT_BRIGHT};">{title}</h1>
          <p style="color:{TEXT_MUTED};font-family:{MONO};font-size:0.95rem;
                    margin:10px 0 0 0;letter-spacing:1.5px;text-transform:uppercase;">
            {subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer(lines: list[str]) -> None:
    """Render the banner footer."""
    body = "<br>".join(lines)
    st.markdown(
        f"""
        <div style="background-color:{PANEL};padding:15px;border-radius:8px;
                    margin-top:40px;text-align:center;border:1px solid {GRID};">
          <p style="color:{TEXT_MUTED};font-family:{MONO};font-size:0.85rem;margin:0;
                    letter-spacing:1.2px;text-transform:uppercase;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(label: str, value: str, colour: str = ACCENT_INFO, height: int = 105) -> None:
    """Render a bordered readout box for a single headline figure."""
    st.markdown(
        f"""
        <div style="padding:20px;background-color:{PANEL};border-left:4px solid {colour};
                    margin-top:10px;height:{height}px;display:flex;flex-direction:column;
                    justify-content:center;">
          <span style="color:{TEXT_MUTED};font-size:0.9rem;text-transform:uppercase;
                       letter-spacing:1px;">{label}</span>
          <span style="color:{colour};font-size:2.5rem;font-family:{MONO};
                       font-weight:bold;margin-top:5px;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_badge(label: str, grade: str, score: float, caption: str) -> None:
    """Render a scorecard grade badge in the console style."""
    colour = GRADE_COLOURS.get(grade, TEXT_MUTED)
    st.markdown(
        f"""
        <div style="border:1px solid {GRID};border-radius:8px;padding:14px 18px;
                    background-color:{PANEL};">
          <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1.2px;
                      color:{TEXT_MUTED};font-family:{MONO};">{label}</div>
          <div style="display:flex;align-items:baseline;gap:12px;margin-top:6px;">
            <span style="font-size:2.6rem;font-weight:700;color:{colour};
                         font-family:{MONO};text-shadow:0 0 6px {colour}55;">{grade}</span>
            <span style="font-size:1.5rem;font-weight:600;color:{TEXT_BRIGHT};
                         font-family:{MONO};">{score:.0f}<span
                  style="font-size:0.9rem;color:{TEXT_MUTED};">/100</span></span>
          </div>
          <div style="font-size:0.82rem;color:{TEXT_MUTED};margin-top:6px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_line(text: str, colour: str = ACCENT_INFO) -> None:
    """Render a single-line status strip, used for alerts and notes."""
    st.markdown(
        f"""
        <div style="background-color:{PANEL};border-left:4px solid {colour};
                    padding:10px 14px;margin:6px 0;font-family:{MONO};
                    font-size:0.85rem;color:{TEXT_BRIGHT};letter-spacing:0.6px;">
          {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """Return the shared Plotly layout for every chart on the console.

    Transparent backgrounds let the dotted grid show through, which is what makes the
    charts sit *in* the console rather than on top of it.
    """
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": MONO, "color": TEXT_BRIGHT, "size": 12},
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "right",
            "x": 1,
            "font": {"color": TEXT_BRIGHT},
        },
    }
    layout.update(overrides)
    return layout


def axis(title: str, **overrides: Any) -> dict[str, Any]:
    """Return a grid-styled axis definition."""
    definition: dict[str, Any] = {
        "title": title,
        "showgrid": True,
        "gridwidth": 1,
        "gridcolor": GRID,
        "zeroline": False,
    }
    definition.update(overrides)
    return definition
