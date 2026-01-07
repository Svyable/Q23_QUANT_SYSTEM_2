"""
Dashboard Styling Components

Centralized CSS and styling helpers for the Q23 Dashboard.
Provides consistent theming, color-coded components, and professional UI elements.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
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
# STRATEGY COLOR PALETTE - For consistent colors across charts and UI elements
# =============================================================================

# Primary strategy colors - vibrant, distinct colors for strategies
STRATEGY_COLORS = [
    "#2ecc71",  # Green (Emerald)
    "#e74c3c",  # Red (Alizarin)
    "#3498db",  # Blue (Peter River)
    "#f39c12",  # Orange (Orange)
    "#9b59b6",  # Purple (Amethyst)
    "#1abc9c",  # Teal (Turquoise)
    "#e67e22",  # Dark Orange (Carrot)
    "#00cec9",  # Cyan (Robin Egg)
    "#fd79a8",  # Pink (Pico)
    "#a29bfe",  # Light Purple (Periwinkle)
    "#ffeaa7",  # Yellow (Pale Gold)
    "#74b9ff",  # Light Blue (Picton)
    "#fab1a0",  # Salmon (Coral)
    "#55efc4",  # Mint (Light Greenish)
    "#636e72",  # Gray Blue (Gloomy)
]

# Benchmark colors - muted grays for visual distinction from strategies
BENCHMARK_COLORS = [
    "#95a5a6",  # Gray (Concrete) - NYSE EW
    "#7f8c8d",  # Dark Gray (Asbestos) - NYSE MC
    "#34495e",  # Navy Gray (Wet Asphalt) - NASDAQ EW
    "#2c3e50",  # Dark Navy (Midnight Blue) - NASDAQ MC
    "#566573",  # Blue Gray - SP500 EW
    "#5d6d7e",  # Steel Blue - SP500 MC
    "#717d7e",  # Pewter - NAS100 EW
    "#626567",  # Charcoal - NAS100 MC
]

# Benchmark strategy IDs for identification
BENCHMARK_STRATEGY_IDS = [
    # Exchange-based benchmarks
    "benchmark_nys_ew",
    "benchmark_nys_mc",
    "benchmark_nas_ew",
    "benchmark_nas_mc",
    # Index-based benchmarks (SP500, NAS100)
    "benchmark_sp500_ew",
    "benchmark_sp500_mc",
    "benchmark_nas100_ew",
    "benchmark_nas100_mc",
]


def get_strategy_color(
    strategy_id: str,
    strategy_index: int = 0,
    all_strategies: Optional[List[str]] = None,
) -> str:
    """
    Get a consistent color for a strategy.
    
    Uses deterministic assignment to ensure the same strategy always gets
    the same color when displayed in the same context (e.g., charts + tags).
    
    Args:
        strategy_id: The strategy identifier
        strategy_index: Index position among non-benchmark strategies
        all_strategies: Full list of strategies for deterministic ordering
    
    Returns:
        Hex color string
    """
    # Benchmarks get gray tones
    if strategy_id in BENCHMARK_STRATEGY_IDS:
        bench_idx = BENCHMARK_STRATEGY_IDS.index(strategy_id)
        return BENCHMARK_COLORS[bench_idx % len(BENCHMARK_COLORS)]
    
    # Regular strategies get vibrant colors
    return STRATEGY_COLORS[strategy_index % len(STRATEGY_COLORS)]


# =============================================================================
# STRATEGY COLOR MANAGEMENT SYSTEM
# =============================================================================

class StrategyColorManager:
    """
    Centralized strategy color management system.

    Provides deterministic color assignment based on strategy IDs,
    user customization, and persistence across sessions.
    """

    def __init__(self):
        self._custom_colors: Dict[str, str] = {}
        self._load_custom_colors()

    def _load_custom_colors(self) -> None:
        """Load custom color assignments from persistent storage."""
        try:
            import json
            from pathlib import Path

            color_file = Path("src/q23/dashboard/strategy_colors.json")
            if color_file.exists():
                with open(color_file, 'r') as f:
                    self._custom_colors = json.load(f)
        except Exception:
            # If loading fails, start with empty custom colors
            self._custom_colors = {}

    def _save_custom_colors(self) -> None:
        """Save custom color assignments to persistent storage."""
        try:
            import json
            from pathlib import Path

            color_file = Path("src/q23/dashboard/strategy_colors.json")
            color_file.parent.mkdir(parents=True, exist_ok=True)

            with open(color_file, 'w') as f:
                json.dump(self._custom_colors, f, indent=2)
        except Exception:
            # If saving fails, just continue
            pass

    def get_strategy_color(self, strategy_id: str) -> str:
        """
        Get the color for a strategy, with deterministic fallback.

        Priority:
        1. User-customized color
        2. Deterministic color based on strategy ID hash

        Args:
            strategy_id: The strategy identifier

        Returns:
            Hex color string
        """
        # Check for custom color first
        if strategy_id in self._custom_colors:
            return self._custom_colors[strategy_id]

        # Use deterministic color assignment based on strategy ID
        return self._get_deterministic_color(strategy_id)

    def _get_deterministic_color(self, strategy_id: str) -> str:
        """
        Generate a deterministic color based on strategy ID hash.

        Uses a hash of the strategy ID to consistently assign colors
        from the appropriate palette (strategy vs benchmark).
        """
        import hashlib

        # Create hash of strategy ID for deterministic assignment
        hash_obj = hashlib.md5(strategy_id.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Use different palettes for benchmarks vs strategies
        if strategy_id in BENCHMARK_STRATEGY_IDS:
            color_palette = BENCHMARK_COLORS
        else:
            color_palette = STRATEGY_COLORS

        # Use hash to select color deterministically
        color_idx = hash_int % len(color_palette)
        return color_palette[color_idx]

    def set_custom_color(self, strategy_id: str, color: str) -> None:
        """
        Set a custom color for a strategy.

        Args:
            strategy_id: The strategy identifier
            color: Hex color string (e.g., '#FF5733')
        """
        self._custom_colors[strategy_id] = color
        self._save_custom_colors()

    def remove_custom_color(self, strategy_id: str) -> None:
        """
        Remove custom color for a strategy (revert to default).

        Args:
            strategy_id: The strategy identifier
        """
        if strategy_id in self._custom_colors:
            del self._custom_colors[strategy_id]
            self._save_custom_colors()

    def get_all_colors(self, strategy_ids: List[str]) -> Dict[str, str]:
        """
        Get color mapping for all provided strategy IDs.

        Args:
            strategy_ids: List of strategy IDs

        Returns:
            Dict mapping strategy_id -> hex color
        """
        return {sid: self.get_strategy_color(sid) for sid in strategy_ids}

    def get_custom_colors(self) -> Dict[str, str]:
        """Get all custom color assignments."""
        return self._custom_colors.copy()

    def reset_all_custom_colors(self) -> None:
        """Reset all custom colors to defaults."""
        self._custom_colors = {}
        self._save_custom_colors()

    def get_color_palette_options(self) -> Dict[str, str]:
        """
        Get available color options for customization.

        Returns:
            Dict of color_name -> hex_color for UI selection
        """
        # Combine all palettes for user selection
        all_colors = {}

        # Strategy colors
        for i, color in enumerate(STRATEGY_COLORS):
            all_colors[f"Strategy {i+1}"] = color

        # Benchmark colors
        for i, color in enumerate(BENCHMARK_COLORS):
            all_colors[f"Benchmark {i+1}"] = color

        # Add some additional nice colors
        extra_colors = {
            "Crimson Red": "#DC143C",
            "Forest Green": "#228B22",
            "Royal Blue": "#4169E1",
            "Dark Orange": "#FF8C00",
            "Purple": "#9370DB",
            "Teal": "#008080",
            "Coral": "#FF7F50",
            "Steel Blue": "#4682B4",
            "Olive": "#808000",
            "Slate Gray": "#708090",
            "Tomato": "#FF6347",
            "Medium Sea Green": "#3CB371",
            "Dodger Blue": "#1E90FF",
            "Orange Red": "#FF4500",
            "Medium Purple": "#9370DB",
        }
        all_colors.update(extra_colors)

        return all_colors


# Global color manager instance
_color_manager = StrategyColorManager()

def get_strategy_color(strategy_id: str) -> str:
    """
    Get the color for a strategy using the global color manager.

    Args:
        strategy_id: The strategy identifier

    Returns:
        Hex color string
    """
    return _color_manager.get_strategy_color(strategy_id)

def set_strategy_color(strategy_id: str, color: str) -> None:
    """
    Set a custom color for a strategy.

    Args:
        strategy_id: The strategy identifier
        color: Hex color string
    """
    _color_manager.set_custom_color(strategy_id, color)

def get_strategy_color_manager() -> StrategyColorManager:
    """Get the global strategy color manager instance."""
    return _color_manager

def build_strategy_color_map(
    strategy_ids: List[str],
) -> Dict[str, str]:
    """
    Build a consistent color mapping for a list of strategies.

    Uses the global color manager to ensure consistent colors across
    all components (charts, UI elements, etc.).

    Args:
        strategy_ids: List of strategy IDs to map

    Returns:
        Dict mapping strategy_id -> hex color
    """
    return _color_manager.get_all_colors(strategy_ids)


def inject_multiselect_colors(
    color_map: Dict[str, str],
    display_name_func: Optional[callable] = None,
) -> None:
    """
    Inject CSS to color multiselect tags based on strategy colors.
    
    This creates CSS rules that target the multiselect tag elements
    and applies the corresponding strategy colors.
    
    Args:
        color_map: Dict mapping strategy_id -> hex color
        display_name_func: Function to convert strategy_id to display name.
                          If None, uses the raw strategy_id.
    """
    if not color_map:
        return
    
    css_rules = []
    
    for strategy_id, color in color_map.items():
        # Get display name for matching the tag text
        display_name = strategy_id
        if display_name_func:
            try:
                display_name = display_name_func(strategy_id)
            except Exception:
                pass
        
        # Create CSS rule targeting multiselect tags by their text content
        # Streamlit multiselect tags have a specific structure we can target
        # The tag text is in a span inside the tag element
        css_rules.append(f"""
        /* Strategy: {strategy_id} - {display_name} */
        span[data-baseweb="tag"]:has(span[title="{display_name}"]) {{
            background-color: {color} !important;
            border-color: {color} !important;
        }}
        span[data-baseweb="tag"]:has(span[title="{display_name}"]) span {{
            color: white !important;
        }}
        /* Also match truncated names */
        span[data-baseweb="tag"]:has(span[title^="{display_name[:20]}"]) {{
            background-color: {color} !important;
            border-color: {color} !important;
        }}
        span[data-baseweb="tag"]:has(span[title^="{display_name[:20]}"]) span {{
            color: white !important;
        }}
        """)
    
    # Combine all rules into a single style block
    full_css = f"""
    <style>
    /* Strategy Comparison Multiselect Color Coding */
    {chr(10).join(css_rules)}
    
    /* Improve tag visibility */
    span[data-baseweb="tag"] {{
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    
    span[data-baseweb="tag"]:hover {{
        filter: brightness(1.1) !important;
        transform: scale(1.02) !important;
    }}
    
    /* Remove button styling */
    span[data-baseweb="tag"] svg {{
        color: rgba(255, 255, 255, 0.8) !important;
    }}
    span[data-baseweb="tag"]:hover svg {{
        color: white !important;
    }}
    </style>
    """
    
    st.markdown(full_css, unsafe_allow_html=True)


# =============================================================================
# GLOBAL CSS INJECTION
# =============================================================================

DASHBOARD_CSS = """
<style>
/* =============================================================================
   GLOBAL ENHANCEMENTS - Subtle polish for app-like feel
   ============================================================================= */

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Improved spacing and padding throughout */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Better data table styling */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #2d2d2d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* Enhanced chart containers - Updated to prevent clipping */
div[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

/* 1) Plotly wrapper is literally "overflow:hidden" inline -> override it */
div[data-testid="stPlotlyChart"] .js-plotly-plot {
    overflow: visible !important;
}

/* 2) If Streamlit puts the element in a scroll box, unclip it too */
div[data-testid="stElementContainer"] {
    overflow: visible !important;
}

/* 3) Internal Plotly SVG containers also need this */
div[data-testid="stPlotlyChart"] .plot-container,
div[data-testid="stPlotlyChart"] .svg-container {
    overflow: visible !important;
}

/* Improved metric display */
[data-testid="stMetricValue"] {
    font-weight: 600;
    letter-spacing: -0.02em;
}

[data-testid="stMetricLabel"] {
    font-size: 0.875rem;
    opacity: 0.85;
    font-weight: 500;
}

/* Better button styling */
button[kind="primary"] {
    box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
    transition: all 0.2s ease;
}

button[kind="primary"]:hover {
    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
    transform: translateY(-1px);
}

/* Enhanced expander styling */
div[data-testid="stExpander"] {
    border-radius: 8px;
    border: 1px solid #2d2d2d;
    background: rgba(26, 26, 26, 0.5);
    transition: all 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: #3a3a3a;
    background: rgba(30, 30, 30, 0.6);
}

/* Improved selectbox and input styling */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    border-radius: 6px;
    transition: all 0.2s ease;
}

div[data-testid="stSelectbox"] > div:focus-within,
div[data-testid="stNumberInput"] > div:focus-within {
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.1);
}

/* Better divider styling */
hr {
    border: none;
    border-top: 1px solid #2d2d2d;
    margin: 1.5rem 0;
}

/* Enhanced caption styling */
[data-testid="stCaption"] {
    opacity: 0.75;
    font-size: 0.875rem;
    font-weight: 400;
}

/* =============================================================================
   SIDEBAR ADMIN DOCK (fixed bottom strip)
   ============================================================================= */

/* Ensure sidebar content doesn't get hidden behind the fixed dock */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-bottom: 3.75rem !important;
    padding-top: 3.25rem !important;
}

/* Fixed dock container (compact pill in top-left corner) */
section[data-testid="stSidebar"] .q23-admin-dock {
    position: fixed;
    left: 0.6rem;
    top: 0.6rem;
    width: fit-content;
    z-index: 10000;
    background: rgba(20, 20, 20, 0.95);
    backdrop-filter: blur(12px);
    border: 1px solid #2d2d2d;
    border-radius: 14px;
    padding: 0.25rem 0.3rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05);
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .q23-admin-dock:hover {
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.08);
}

/* Dock inner layout */
section[data-testid="stSidebar"] .q23-admin-dock .dock-row {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.25rem;
}

section[data-testid="stSidebar"] .q23-admin-dock a.dock-item {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.15rem;
    height: 2.15rem;
    padding: 0;
    border-radius: 12px;
    text-decoration: none;
    color: #ecf0f1;
    border: 1px solid transparent;
    font-weight: 700;
    font-size: 1.05rem;
    transition: all 0.18s ease;
}

section[data-testid="stSidebar"] .q23-admin-dock a.dock-item:hover {
    background: #1e1e1e;
    border-color: #333;
    transform: translateY(-1px);
}

section[data-testid="stSidebar"] .q23-admin-dock a.dock-item.active {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-color: #0f3460;
    color: #ffffff;
}

/* Hide labels in compact mode (icon-only) */
section[data-testid="stSidebar"] .q23-admin-dock .dock-label {
    display: none !important;
}

@media (max-width: 768px) {
    section[data-testid="stSidebar"] .q23-admin-dock {
        left: 0.5rem;
        top: 0.5rem;
    }
}

/* Factor Category Buttons */
div[data-testid="stButton"] button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid transparent;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    border-color: rgba(255, 255, 255, 0.1);
}

div[data-testid="stButton"] button:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #2d2d2d;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #3a3a3a;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transform: translateY(-1px);
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
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #2d2d2d;
}

.styled-table th {
    background: linear-gradient(135deg, #2d2d2d 0%, #252525 100%);
    color: #ecf0f1;
    padding: 14px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 2px solid #3a3a3a;
}

.styled-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #2d2d2d;
    transition: background-color 0.15s ease;
}

.styled-table tr:nth-child(even) {
    background: #1e1e1e;
}

.styled-table tr:hover {
    background: #2a2a2a;
    cursor: pointer;
}

.styled-table tr:last-child td {
    border-bottom: none;
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
# GRADIENT COLOR CONSTANTS
# =============================================================================

# Long positions: Green to black gradient
LONG_GRADIENT_COLORS = [
    "#2ecc71",  # Bright green (max weight)
    "#27ae60",  # Dark green
    "#1e8449",  # Darker green
    "#145a32",  # Very dark green
    "#000000",  # Black (min weight/null)
]

# Short positions: Red to black gradient
SHORT_GRADIENT_COLORS = [
    "#e74c3c",  # Bright red (max absolute weight)
    "#c0392b",  # Dark red
    "#a93226",  # Darker red
    "#7b241c",  # Very dark red
    "#000000",  # Black (min weight/null)
]


# =============================================================================
# TABLE STYLING HELPERS
# =============================================================================

def _create_long_gradient_colormap() -> list:
    """Create a custom colormap for long weights (green to black)."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    
    colors = LONG_GRADIENT_COLORS
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('long_gradient', colors, N=n_bins)
    return cmap


def _create_short_gradient_colormap() -> list:
    """Create a custom colormap for short weights (red to black)."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    
    colors = SHORT_GRADIENT_COLORS
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('short_gradient', colors, N=n_bins)
    return cmap


def style_long_weights_gradient(
    df, 
    column: str = "weight",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "pd.Styler":
    """
    Apply green-to-black gradient to long weights.
    
    Args:
        df: DataFrame with weight column
        column: Name of the weight column
        vmin: Minimum value for normalization (defaults to df min)
        vmax: Maximum value for normalization (defaults to df max)
    
    Returns:
        Styled DataFrame
    """
    import pandas as pd
    import numpy as np
    
    if df.empty or column not in df.columns:
        return df.style
    
    # For long weights, use positive values only
    weights = df[column]
    positive_weights = weights[weights > 0]
    
    if vmin is None:
        vmin = positive_weights.min() if not positive_weights.empty else 0.0
    if vmax is None:
        vmax = positive_weights.max() if not positive_weights.empty else 1.0
    
    # Normalize to 0-1 range
    if vmax > vmin:
        normalized = (weights - vmin) / (vmax - vmin)
    else:
        normalized = pd.Series([0.0] * len(weights), index=weights.index)
    
    # Create gradient function
    def apply_long_gradient(val):
        try:
            weight = float(val)
            if pd.isna(weight) or weight <= 0:
                return 'background-color: #000000; color: #888888'
            
            # Normalize this specific weight value
            norm_val = (weight - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            norm_val = max(0.0, min(1.0, norm_val))
            
            # Map to gradient (0 = black, 1 = bright green)
            if norm_val < 0.25:
                color = LONG_GRADIENT_COLORS[4]  # Black
            elif norm_val < 0.5:
                color = LONG_GRADIENT_COLORS[3]  # Very dark green
            elif norm_val < 0.75:
                color = LONG_GRADIENT_COLORS[2]  # Darker green
            else:
                color = LONG_GRADIENT_COLORS[1] if norm_val < 0.9 else LONG_GRADIENT_COLORS[0]  # Bright green
            
            # Determine text color (light for dark backgrounds, dark for light)
            text_color = '#ecf0f1' if norm_val < 0.5 else '#000000'
            
            return f'background-color: {color}; color: {text_color}'
        except (ValueError, TypeError):
            return 'background-color: #000000; color: #888888'
    
    styled = df.style.map(
        apply_long_gradient,
        subset=[column]
    ).format({column: "{:.2%}"})
    
    return styled


def style_short_weights_gradient(
    df,
    column: str = "weight",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "pd.Styler":
    """
    Apply red-to-black gradient to short weights (absolute values).
    
    Args:
        df: DataFrame with weight column (should contain negative values)
        column: Name of the weight column
        vmin: Minimum absolute value for normalization (defaults to df min abs)
        vmax: Maximum absolute value for normalization (defaults to df max abs)
    
    Returns:
        Styled DataFrame
    """
    import pandas as pd
    import numpy as np
    
    if df.empty or column not in df.columns:
        return df.style
    
    # For short weights, use absolute values of negative weights
    weights = df[column]
    negative_weights = weights[weights < 0]
    abs_negative = negative_weights.abs()
    
    if vmin is None:
        vmin = abs_negative.min() if not abs_negative.empty else 0.0
    if vmax is None:
        vmax = abs_negative.max() if not abs_negative.empty else 1.0
    
    # Normalize to 0-1 range
    if vmax > vmin:
        normalized = (abs_negative - vmin) / (vmax - vmin)
    else:
        normalized = pd.Series([0.0] * len(abs_negative), index=abs_negative.index)
    
    # Create gradient function
    def apply_short_gradient(val):
        try:
            weight = float(val)
            if pd.isna(weight) or weight >= 0:
                return 'background-color: #000000; color: #888888'
            
            # Use absolute value for normalization
            abs_weight = abs(weight)
            norm_val = (abs_weight - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            norm_val = max(0.0, min(1.0, norm_val))
            
            # Map to gradient (0 = black, 1 = bright red)
            if norm_val < 0.25:
                color = SHORT_GRADIENT_COLORS[4]  # Black
            elif norm_val < 0.5:
                color = SHORT_GRADIENT_COLORS[3]  # Very dark red
            elif norm_val < 0.75:
                color = SHORT_GRADIENT_COLORS[2]  # Darker red
            else:
                color = SHORT_GRADIENT_COLORS[1] if norm_val < 0.9 else SHORT_GRADIENT_COLORS[0]  # Bright red
            
            # Determine text color (light for dark backgrounds, dark for light)
            text_color = '#ecf0f1' if norm_val < 0.5 else '#ffffff'
            
            return f'background-color: {color}; color: {text_color}'
        except (ValueError, TypeError):
            return 'background-color: #000000; color: #888888'
    
    styled = df.style.map(
        apply_short_gradient,
        subset=[column]
    ).format({column: "{:.2%}"})
    
    return styled


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
                styled = styled.map(color_negative_red, subset=[col])
    
    return styled


def style_weights_diverging_gradient(
    df,
    column: str = "weight",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "pd.Styler":
    """
    Apply diverging gradient to weights: green (positive) -> black (zero) -> red (negative).
    
    Positive values fade from green to black as they approach zero.
    Negative values fade from red to black as they approach zero.
    Zero/null values are black.
    
    Args:
        df: DataFrame with weight column (can contain both positive and negative values)
        column: Name of the weight column
        vmin: Minimum value for normalization (defaults to df min)
        vmax: Maximum value for normalization (defaults to df max)
    
    Returns:
        Styled DataFrame
    """
    import pandas as pd
    import numpy as np
    
    if df.empty or column not in df.columns:
        return df.style
    
    weights = df[column]
    
    # Determine normalization bounds
    if vmin is None:
        vmin = weights.min() if not weights.empty else -1.0
    if vmax is None:
        vmax = weights.max() if not weights.empty else 1.0
    
    # Ensure symmetric bounds for diverging scale
    abs_max = max(abs(vmin), abs(vmax)) if (vmin < 0 and vmax > 0) else max(abs(vmin), abs(vmax))
    if vmin < 0 and vmax > 0:
        # Diverging: symmetric around zero
        vmin_normalized = -abs_max
        vmax_normalized = abs_max
    else:
        # One-sided: use actual bounds
        vmin_normalized = vmin
        vmax_normalized = vmax
    
    # Create gradient function
    def apply_diverging_gradient(val):
        try:
            weight = float(val)
            if pd.isna(weight):
                return 'background-color: #000000; color: #888888'
            
            # Handle zero
            if abs(weight) < 1e-10:
                return 'background-color: #000000; color: #888888'
            
            # Normalize to 0-1 range for positive, 0-1 for negative (based on absolute value)
            if vmax_normalized > vmin_normalized:
                if weight > 0:
                    # Positive: normalize to 0-1, where 1 = max, 0 = zero
                    norm_val = weight / vmax_normalized if vmax_normalized > 0 else 0.0
                    norm_val = max(0.0, min(1.0, norm_val))
                else:
                    # Negative: normalize to 0-1, where 1 = min (most negative), 0 = zero
                    norm_val = abs(weight) / abs(vmin_normalized) if vmin_normalized < 0 else 0.0
                    norm_val = max(0.0, min(1.0, norm_val))
            else:
                norm_val = 0.0
            
            # Map to gradient
            if weight > 0:
                # Long: green to black (norm_val 1 = bright green, 0 = black)
                if norm_val < 0.25:
                    color = LONG_GRADIENT_COLORS[4]  # Black
                elif norm_val < 0.5:
                    color = LONG_GRADIENT_COLORS[3]  # Very dark green
                elif norm_val < 0.75:
                    color = LONG_GRADIENT_COLORS[2]  # Darker green
                else:
                    color = LONG_GRADIENT_COLORS[1] if norm_val < 0.9 else LONG_GRADIENT_COLORS[0]  # Bright green
                
                text_color = '#ecf0f1' if norm_val < 0.5 else '#000000'
            else:
                # Short: red to black (norm_val 1 = bright red, 0 = black)
                if norm_val < 0.25:
                    color = SHORT_GRADIENT_COLORS[4]  # Black
                elif norm_val < 0.5:
                    color = SHORT_GRADIENT_COLORS[3]  # Very dark red
                elif norm_val < 0.75:
                    color = SHORT_GRADIENT_COLORS[2]  # Darker red
                else:
                    color = SHORT_GRADIENT_COLORS[1] if norm_val < 0.9 else SHORT_GRADIENT_COLORS[0]  # Bright red
                
                text_color = '#ecf0f1' if norm_val < 0.5 else '#ffffff'
            
            return f'background-color: {color}; color: {text_color}'
        except (ValueError, TypeError):
            return 'background-color: #000000; color: #888888'
    
    styled = df.style.map(
        apply_diverging_gradient,
        subset=[column]
    ).format({column: "{:.2%}"})
    
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

OVERVIEW_PAGE_CSS = """
<style>
/* Strategy Header */
.strategy-header-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #0f3460;
}
.strategy-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.strategy-meta {
    color: #888;
    font-size: 0.95rem;
}
.strategy-meta code {
    background: #2d2d2d;
    padding: 2px 8px;
    border-radius: 4px;
    color: #3498db;
}
.perf-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 1rem;
}
.badge-excellent { background: #2ecc71; color: #000; }
.badge-good { background: #3498db; color: #fff; }
.badge-fair { background: #f39c12; color: #000; }
.badge-poor { background: #e74c3c; color: #fff; }

/* Section Headers */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ecf0f1;
    margin: 2rem 0 1.25rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #2d2d2d;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    letter-spacing: -0.01em;
    position: relative;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #3498db, transparent);
}
.section-title .icon {
    font-size: 1.2rem;
}

/* KPI Cards */
.kpi-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.kpi-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    flex: 1;
    border: 1px solid #333;
    text-align: center;
}
.kpi-card.primary {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    box-shadow: 0 4px 12px rgba(15, 52, 96, 0.2);
}

.kpi-card.primary:hover {
    box-shadow: 0 6px 16px rgba(15, 52, 96, 0.3);
    border-color: #1a4a7a;
}
.kpi-card.risk {
    border-left: 3px solid #e74c3c;
}
.kpi-card.win {
    border-left: 3px solid #2ecc71;
}
.kpi-label {
    font-size: 0.75rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
    font-weight: 500;
    opacity: 0.9;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.kpi-value.positive { color: #2ecc71; }
.kpi-value.negative { color: #e74c3c; }
.kpi-value.neutral { color: #ecf0f1; }
.kpi-value.highlight { color: #3498db; }

/* Metric Table */
.metric-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}
.metric-table th {
    background: #2d2d2d;
    color: #ecf0f1;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
}
.metric-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #333;
    font-size: 0.9rem;
}
.metric-table tr:nth-child(even) { background: #1e1e1e; }
.metric-table tr:hover { background: #2a2a2a; }
.metric-table .value-positive { color: #2ecc71; font-weight: 600; }
.metric-table .value-negative { color: #e74c3c; font-weight: 600; }
</style>
"""

# Legacy alias for backward compatibility
OVERVIEW_CSS = OVERVIEW_PAGE_CSS


def inject_overview_css() -> None:
    """Inject CSS specific to the Overview page."""
    st.markdown(OVERVIEW_PAGE_CSS, unsafe_allow_html=True)


# =============================================================================
# REUSABLE KPI CARD COMPONENTS
# =============================================================================

from typing import List, Tuple, Union
from dataclasses import dataclass


@dataclass
class KPIMetric:
    """A single KPI metric for rendering."""
    label: str
    value: Union[str, float]
    value_class: str = "neutral"  # positive, negative, neutral, highlight
    format_spec: str = ""  # e.g., ".2%", ".3f"
    
    def formatted_value(self) -> str:
        """Return formatted value string."""
        if isinstance(self.value, str):
            return self.value
        if self.format_spec:
            return f"{self.value:{self.format_spec}}"
        return str(self.value)


def render_kpi_row(
    metrics: List[KPIMetric],
    card_class: str = "",
) -> None:
    """
    Render a row of KPI cards.
    
    Args:
        metrics: List of KPIMetric objects to render
        card_class: Additional CSS class for cards (primary, risk, win)
    """
    cards_html = []
    for m in metrics:
        card_cls = f"kpi-card {card_class}".strip()
        cards_html.append(f"""
        <div class="{card_cls}">
            <div class="kpi-label">{m.label}</div>
            <div class="kpi-value {m.value_class}">{m.formatted_value()}</div>
        </div>
        """)
    
    html = f'<div class="kpi-row">{"".join(cards_html)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_section_title(title: str, icon: str = "") -> None:
    """
    Render a styled section title with optional icon.
    
    Args:
        title: Section title text
        icon: Optional emoji icon
    """
    icon_html = f'<span class="icon">{icon}</span>' if icon else ""
    html = f'<div class="section-title">{icon_html} {title}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_strategy_header_v2(
    title: str,
    strategy_name: str,
    tag: str,
    date_range: Tuple[str, str],
    sharpe: float,
) -> None:
    """
    Render the strategy header with performance badge.
    
    Args:
        title: Main title
        strategy_name: Strategy display name
        tag: Run tag
        date_range: (start, end) date tuple
        sharpe: Sharpe ratio for badge
    """
    if sharpe >= 2.0:
        badge_class, badge_text = "badge-excellent", "Excellent"
    elif sharpe >= 1.0:
        badge_class, badge_text = "badge-good", "Good"
    elif sharpe >= 0.5:
        badge_class, badge_text = "badge-fair", "Fair"
    else:
        badge_class, badge_text = "badge-poor", "Needs Work"
    
    html = f"""
    <div class="strategy-header-container">
        <div class="strategy-title">{title}</div>
        <div class="strategy-meta">
            <strong>{strategy_name}</strong> &nbsp;|&nbsp; 
            Tag: <code>{tag}</code> &nbsp;|&nbsp; 
            {date_range[0]} to {date_range[1]}
            <span class="perf-badge {badge_class}">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_value_class(
    value: float,
    threshold: float = 0.0,
    invert: bool = False,
) -> str:
    """
    Get CSS class based on value relative to threshold.
    
    Args:
        value: The value to evaluate
        threshold: Comparison threshold
        invert: If True, negative values are "positive" (e.g., for costs)
    
    Returns:
        CSS class: "positive", "negative", or "neutral"
    """
    if invert:
        if value < threshold:
            return "positive"
        elif value > threshold:
            return "negative"
    else:
        if value > threshold:
            return "positive"
        elif value < threshold:
            return "negative"
    return "neutral"


def render_kpi_grid(
    rows: List[List[KPIMetric]],
    card_classes: Optional[List[str]] = None,
) -> None:
    """
    Render multiple rows of KPI cards.
    
    Args:
        rows: List of rows, each row is a list of KPIMetric
        card_classes: Optional list of card classes for each row
    """
    if card_classes is None:
        card_classes = [""] * len(rows)
    
    for row, cls in zip(rows, card_classes):
        render_kpi_row(row, card_class=cls)
