"""
Dashboard Styling Components

Centralized CSS and styling helpers for the Q23 Dashboard.
Provides consistent theming, color-coded components, and professional UI elements.
"""

from __future__ import annotations

from typing import Optional, Literal
import streamlit as st


# =============================================================================
# COLOR PALETTE
# =============================================================================

# Factor category colors
COLORS = {
    # Factor categories
    "defensive": "#3498db",     # Blue - stability, protection
    "alpha": "#2ecc71",         # Green - growth, returns
    "quality": "#f1c40f",       # Gold - excellence, premium
    "combo": "#9b59b6",         # Purple - combination, synergy
    "neutral": "#95a5a6",       # Gray - neutral, all
    "reset": "#e74c3c",         # Red - reset, danger
    
    # Metric colors
    "positive": "#2ecc71",      # Green
    "negative": "#e74c3c",      # Red
    "warning": "#f39c12",       # Orange
    "info": "#3498db",          # Blue
    
    # Container colors
    "card_bg": "#1e1e1e",
    "card_border": "#333333",
    "header_gradient_start": "#667eea",
    "header_gradient_end": "#764ba2",
}


# =============================================================================
# GLOBAL CSS INJECTION
# =============================================================================

DASHBOARD_CSS = """
<style>
/* Factor Category Buttons */
div[data-testid="stButton"] button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #333;
    margin-bottom: 1rem;
}

.metric-card-header {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-card-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.metric-card-value.positive { color: #2ecc71; }
.metric-card-value.negative { color: #e74c3c; }
.metric-card-value.neutral { color: #ecf0f1; }

/* Section Headers */
.section-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #333;
}

/* KPI Container */
.kpi-container {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #16213e;
}

/* Performance Badge */
.perf-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.perf-badge.excellent { background: #2ecc71; color: #000; }
.perf-badge.good { background: #3498db; color: #fff; }
.perf-badge.warning { background: #f39c12; color: #000; }
.perf-badge.poor { background: #e74c3c; color: #fff; }

/* Table Styling */
.styled-table {
    border-collapse: collapse;
    width: 100%;
}

.styled-table th {
    background: #2d2d2d;
    color: #ecf0f1;
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

.styled-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #333;
}

.styled-table tr:nth-child(even) {
    background: #1e1e1e;
}

.styled-table tr:hover {
    background: #2a2a2a;
}

/* Strategy Header */
.strategy-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid #0f3460;
}

.strategy-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.strategy-subtitle {
    color: #888;
    font-size: 1.1rem;
}

/* Risk Container */
.risk-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d1a1a 100%);
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 1rem;
}

/* Win Container */
.win-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #1a2d1a 100%);
    border-left: 4px solid #2ecc71;
    border-radius: 8px;
    padding: 1rem;
}

/* Factor Selection Styling */
.factor-section {
    background: #1e1e1e;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.factor-section.defensive { border-left: 4px solid #3498db; }
.factor-section.alpha { border-left: 4px solid #2ecc71; }
.factor-section.quality { border-left: 4px solid #f1c40f; }
.factor-section.combo { border-left: 4px solid #9b59b6; }
</style>
"""


def inject_custom_css() -> None:
    """Inject all custom CSS at page load. Call once at the start of the app."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


# =============================================================================
# STYLED BUTTON HELPERS
# =============================================================================

def styled_button_css(button_key: str, color: str) -> str:
    """Generate CSS to style a specific button by its key."""
    return f"""
    <style>
    div[data-testid="stButton"]:has(button[kind="secondary"]) button[key="{button_key}"] {{
        background-color: {color} !important;
        color: white !important;
        border: none !important;
    }}
    </style>
    """


def factor_button_row() -> None:
    """
    Inject CSS for the factor category buttons in Live Strategy page.
    Call this before rendering the buttons.
    """
    css = """
    <style>
    /* All Factors button - Gray */
    div.row-widget.stButton:nth-of-type(1) button {
        background: linear-gradient(135deg, #636e72 0%, #95a5a6 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Defensive button - Blue */
    div.row-widget.stButton:nth-of-type(2) button {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Alpha button - Green */
    div.row-widget.stButton:nth-of-type(3) button {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Quality button - Gold */
    div.row-widget.stButton:nth-of-type(4) button {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%) !important;
        color: #1a1a1a !important;
        border: none !important;
        font-weight: 700 !important;
    }
    
    /* Reset button - Red */
    div.row-widget.stButton:nth-of-type(5) button {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# METRIC DISPLAY HELPERS
# =============================================================================

def metric_value_class(value: float, threshold: float = 0.0, 
                       good_direction: Literal["up", "down"] = "up") -> str:
    """Determine CSS class for metric value based on direction and threshold."""
    if good_direction == "up":
        if value > threshold:
            return "positive"
        elif value < threshold:
            return "negative"
    else:  # good_direction == "down" (e.g., drawdown)
        if value < threshold:
            return "positive"
        elif value > threshold:
            return "negative"
    return "neutral"


def styled_metric_card(
    label: str,
    value: str,
    value_class: str = "neutral",
    sublabel: Optional[str] = None,
) -> None:
    """
    Render a styled metric card.
    
    Args:
        label: The metric label
        value: The formatted value string
        value_class: CSS class for coloring (positive, negative, neutral)
        sublabel: Optional additional context
    """
    sublabel_html = f'<div style="color: #666; font-size: 0.8rem;">{sublabel}</div>' if sublabel else ''
    
    html = f"""
    <div class="metric-card">
        <div class="metric-card-header">{label}</div>
        <div class="metric-card-value {value_class}">{value}</div>
        {sublabel_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def performance_badge(sharpe: float) -> str:
    """Generate HTML for a performance badge based on Sharpe ratio."""
    if sharpe >= 2.0:
        badge_class = "excellent"
        text = "Excellent"
    elif sharpe >= 1.0:
        badge_class = "good"
        text = "Good"
    elif sharpe >= 0.5:
        badge_class = "warning"
        text = "Fair"
    else:
        badge_class = "poor"
        text = "Poor"
    
    return f'<span class="perf-badge {badge_class}">{text}</span>'


# =============================================================================
# SECTION HEADER HELPERS
# =============================================================================

def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f'<div class="section-header">{icon_html}{title}</div>', unsafe_allow_html=True)


def strategy_header(
    title: str,
    strategy_name: str,
    tag: str,
    date_range: str,
    sharpe: Optional[float] = None,
) -> None:
    """
    Render the main strategy header with title, metadata, and performance badge.
    """
    badge_html = performance_badge(sharpe) if sharpe is not None else ""
    
    html = f"""
    <div class="strategy-header">
        <div class="strategy-title">{title}</div>
        <div class="strategy-subtitle">
            <strong>{strategy_name}</strong> | Tag: <code>{tag}</code> | {date_range}
            {badge_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CONTAINER HELPERS
# =============================================================================

def risk_container_start() -> None:
    """Start a risk-styled container (red accent)."""
    st.markdown('<div class="risk-container">', unsafe_allow_html=True)


def risk_container_end() -> None:
    """End a risk-styled container."""
    st.markdown('</div>', unsafe_allow_html=True)


def win_container_start() -> None:
    """Start a win-styled container (green accent)."""
    st.markdown('<div class="win-container">', unsafe_allow_html=True)


def win_container_end() -> None:
    """End a win-styled container."""
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TABLE STYLING HELPERS
# =============================================================================

def style_dataframe(df, numeric_cols: Optional[list] = None):
    """
    Apply professional styling to a pandas DataFrame for display.
    
    Args:
        df: DataFrame to style
        numeric_cols: List of columns to apply numeric formatting/coloring
    
    Returns:
        Styled DataFrame
    """
    import pandas as pd
    
    def color_negative_red(val):
        """Color negative values red, positive green."""
        try:
            num = float(str(val).replace('%', '').replace(',', ''))
            if num < 0:
                return 'color: #e74c3c'
            elif num > 0:
                return 'color: #2ecc71'
        except (ValueError, TypeError):
            pass
        return ''
    
    styled = df.style.set_properties(**{
        'background-color': '#1e1e1e',
        'color': '#ecf0f1',
        'border': '1px solid #333',
        'text-align': 'right',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#2d2d2d'),
            ('color', '#ecf0f1'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding', '10px'),
        ]},
        {'selector': 'td', 'props': [
            ('padding', '8px 12px'),
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#252525'),
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#2a2a2a'),
        ]},
    ])
    
    # Apply conditional formatting to numeric columns
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                styled = styled.applymap(color_negative_red, subset=[col])
    
    return styled


# =============================================================================
# LIVE STRATEGY PAGE SPECIFIC STYLES
# =============================================================================

LIVE_STRATEGY_BUTTON_CSS = """
<style>
/* Style for All Factors button */
[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
    background: linear-gradient(135deg, #636e72 0%, #95a5a6 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Style for Defensive button */
[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
    background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Style for Alpha button */
[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Style for Quality/Reset button in position 4 */
[data-testid="stHorizontalBlock"] > div:nth-child(4) button {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Expander headers with color coding */
div[data-testid="stExpander"]:has(p:contains("Defensive")) {
    border-left: 4px solid #3498db !important;
}

div[data-testid="stExpander"]:has(p:contains("Alpha")) {
    border-left: 4px solid #2ecc71 !important;
}

div[data-testid="stExpander"]:has(p:contains("Quality")) {
    border-left: 4px solid #f1c40f !important;
}

div[data-testid="stExpander"]:has(p:contains("Interaction")) {
    border-left: 4px solid #9b59b6 !important;
}
</style>
"""


def inject_live_strategy_css() -> None:
    """Inject CSS specific to the Live Strategy page."""
    st.markdown(LIVE_STRATEGY_BUTTON_CSS, unsafe_allow_html=True)


# =============================================================================
# OVERVIEW PAGE SPECIFIC STYLES  
# =============================================================================

OVERVIEW_CSS = """
<style>
/* Primary KPI styling */
.primary-kpi {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #0f3460;
}

.primary-kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
}

.primary-kpi-label {
    color: #888;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Metric group containers */
.metric-group {
    background: #1e1e1e;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.metric-group-title {
    font-size: 1rem;
    font-weight: 600;
    color: #888;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #333;
}

/* Return coloring */
.return-positive { color: #2ecc71 !important; }
.return-negative { color: #e74c3c !important; }

/* Sharpe coloring */
.sharpe-excellent { color: #2ecc71 !important; }
.sharpe-good { color: #3498db !important; }
.sharpe-fair { color: #f39c12 !important; }
.sharpe-poor { color: #e74c3c !important; }
</style>
"""


def inject_overview_css() -> None:
    """Inject CSS specific to the Overview page."""
    st.markdown(OVERVIEW_CSS, unsafe_allow_html=True)
