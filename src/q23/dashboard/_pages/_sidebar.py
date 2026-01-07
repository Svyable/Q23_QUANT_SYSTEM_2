"""
Sidebar Component

Handles all sidebar navigation and selection logic.
Returns a SidebarState dataclass containing all user selections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Protocol

import pandas as pd
import streamlit as st

from q23.dashboard.core import (
    _project_root,
    _base_output_dir,
    _discover_tags,
    discover_available_strategies,
    discover_strategy_tags,
    get_strategy_output_dir,
    get_strategy_display_name,
    get_strategy_configs,
    get_date_preset_range,
    get_default_date_range,
    get_enabled_strategy_ids,
)
from q23.dashboard.components.styles import (
    build_strategy_color_map,
    inject_multiselect_colors,
    get_strategy_color,
)
from q23.shared.config import cfg, TransactionCostConfig, TransactionCostScheme
from q23.dashboard.admin_settings import get_session_tc_config, set_session_tc_config


# =============================================================================
# CONSTANTS - Following DRY principle
# =============================================================================

PAGE_OPTIONS = [
    "Overview",
    "Weights",
    "Factors/IC",
    "Attribution",
    "Bias",
    "Diagnostics",
    "What-If",
    "Live Strategy",
    "Strategy Warehouse",
    "Blotter",
    "Rebalance",
    "Strategy Comparison",
    "Position Stack",
    "Stock Analysis",
    "Stock Elite",
    "Calendar Heatmap",
]

ELITE_PAGE_OPTIONS = [
    "Summary",
    "PM Risk Dashboard",
    "Risk Attribution",
    "Brinson Attribution",
    "Ex-Ante Risk",
    "Sector Analysis",
    "Correlation Analysis",
    "Beta Analysis",
    "Regime Analysis",
    "Tail Risk & Stress",
    "Convexity & Gamma",
    "Alpha Decay",
    "Capacity Estimation",
    "Factor Timing",
    "Rotation Velocity",
    "Rank IC & Quintiles",
]


# =============================================================================
# SOLID DESIGN: Protocols and Interfaces
# =============================================================================

class ISidebarSection(Protocol):
    """Interface for sidebar sections following Interface Segregation Principle."""

    def render(self, sidebar_state: 'SidebarState') -> 'SidebarState':
        """Render this section and return updated state."""
        ...


class IStrategySelector(ABC):
    """Abstract base for strategy selection logic."""

    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        pass

    @abstractmethod
    def get_default_strategy(self, strategies: List[str]) -> str:
        """Get the default strategy to select."""
        pass

    @abstractmethod
    def get_strategy_configs(self) -> dict:
        """Get configuration for all strategies."""
        pass


class ITagResolver(ABC):
    """Abstract base for tag resolution logic."""

    @abstractmethod
    def get_tags_for_strategy(self, strategy_id: str) -> List[str]:
        """Get available tags for a strategy."""
        pass

    @abstractmethod
    def get_default_tag(self, tags: List[str], strategy_id: str) -> str:
        """Get the default tag for a strategy."""
        pass


class IDateRangeSelector(ABC):
    """Abstract base for date range selection."""

    @abstractmethod
    def get_available_ranges(self) -> dict:
        """Get available date range options."""
        pass

    @abstractmethod
    def get_default_range(self) -> Tuple[str, str]:
        """Get default date range."""
        pass


# =============================================================================
# SOLID DESIGN: Concrete Implementations
# =============================================================================

class StrategySelector(IStrategySelector):
    """Concrete implementation of strategy selection following SRP."""

    def get_available_strategies(self) -> List[str]:
        """Get enabled strategies for sidebar display."""
        return get_enabled_strategy_ids(include_benchmarks=False)

    def get_default_strategy(self, strategies: List[str]) -> str:
        """Get strategy with most recent run or admin default."""
        if not strategies:
            return ""

        # Check admin default first
        default_strategy_id = st.session_state.get("admin_default_strategy_id")
        if default_strategy_id in strategies:
            return default_strategy_id

        # Find strategy with latest run
        latest_strategy, _ = _find_strategy_with_latest_run(strategies)
        return latest_strategy or strategies[0]

    def get_strategy_configs(self) -> dict:
        """Get configuration dictionary for strategies."""
        return get_strategy_configs()


class TagResolver(ITagResolver):
    """Concrete implementation of tag resolution following SRP."""

    def get_tags_for_strategy(self, strategy_id: str) -> List[str]:
        """Get tags for a specific strategy."""
        return discover_strategy_tags(strategy_id)

    def get_default_tag(self, tags: List[str], strategy_id: str) -> str:
        """Get default tag, preferring most recent."""
        if not tags:
            return ""

        # Check session state for previously selected tag
        prev_strategy = st.session_state.get("_prev_strategy")
        if prev_strategy == strategy_id:
            selected_tag = st.session_state.get("q23_selected_tag")
            if selected_tag in tags:
                return selected_tag

        # Default to most recent tag
        return tags[0]


class DateRangeSelector(IDateRangeSelector):
    """Concrete implementation of date range selection following SRP."""

    def get_available_ranges(self) -> dict:
        """Get available date range presets."""
        return {
            "1M": get_date_preset_range("1M"),
            "3M": get_date_preset_range("3M"),
            "6M": get_date_preset_range("6M"),
            "1Y": get_date_preset_range("1Y"),
            "2Y": get_date_preset_range("2Y"),
            "5Y": get_date_preset_range("5Y"),
            "10Y": get_date_preset_range("10Y"),
            "YTD": get_date_preset_range("YTD"),
            "MAX": get_date_preset_range("MAX"),
        }

    def get_default_range(self) -> Tuple[str, str]:
        """Get default date range from configuration."""
        return get_default_date_range()


# =============================================================================
# SOLID DESIGN: Sidebar Section Classes
# =============================================================================

def hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB tuple string for CSS."""
    hex_color = hex_color.lstrip('#')
    try:
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"
    except (ValueError, IndexError):
        return "128, 128, 128"  # Default to gray if invalid hex


class SidebarCSSInjector:
    """Handles CSS injection for sidebar styling. Follows SRP."""

    @staticmethod
    def inject_wide_dropdown_css() -> None:
        """Inject CSS to make the strategy dropdown wider than sidebar."""
        st.markdown(
            """
            <style>
            /* Make strategy dropdown pop out wider than sidebar */
            div[data-testid="stSidebar"] div[data-baseweb="select"] > div {
                min-width: 100%;
            }
            div[data-testid="stSidebar"] div[data-baseweb="popover"] {
                min-width: 380px !important;
                max-width: 450px !important;
            }
            div[data-testid="stSidebar"] ul[role="listbox"] {
                min-width: 380px !important;
                max-width: 450px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def inject_strategy_colors(color_map: dict) -> None:
        """Inject CSS for strategy color indicators in selectbox."""
        if not color_map:
            return

        css_rules = []

        for strategy_id, color in color_map.items():
            display_name = get_strategy_display_name(strategy_id)

            # Target selectbox dropdown options with multiple CSS selectors
            # Streamlit selectbox options use different structures
            css_rules.append(f"""
            /* Strategy: {strategy_id} - {display_name} */
            /* Target the dropdown option text */
            div[data-testid="stSidebar"] div[data-baseweb="menu"] div[data-baseweb="menu-item"]:has-text("{display_name}"),
            div[data-testid="stSidebar"] div[role="listbox"] div[role="option"]:has-text("{display_name}"),
            div[data-testid="stSidebar"] li[data-baseweb="menu-item"]:has-text("{display_name}") {{
                border-left: 4px solid {color} !important;
                padding-left: 12px !important;
                background: linear-gradient(90deg, rgba({hex_to_rgb(color)}, 0.1) 0%, rgba({hex_to_rgb(color)}, 0.02) 30%, transparent 50%) !important;
                position: relative !important;
            }}
            """)

            # Add a color indicator dot/pill
            css_rules.append(f"""
            div[data-testid="stSidebar"] div[data-baseweb="menu"] div[data-baseweb="menu-item"]:has-text("{display_name}")::before,
            div[data-testid="stSidebar"] div[role="listbox"] div[role="option"]:has-text("{display_name}")::before,
            div[data-testid="stSidebar"] li[data-baseweb="menu-item"]:has-text("{display_name}")::before {{
                content: "";
                position: absolute;
                left: 6px;
                top: 50%;
                transform: translateY(-50%);
                width: 8px;
                height: 8px;
                background-color: {color};
                border-radius: 50%;
                border: 2px solid rgba(255, 255, 255, 0.8);
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
                z-index: 1;
            }}
            """)

        # Add general styling for better dropdown appearance
        css_rules.append("""
        /* General dropdown styling */
        div[data-testid="stSidebar"] div[data-baseweb="popover"] {
            z-index: 1000 !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        }

        div[data-testid="stSidebar"] div[data-baseweb="menu"] div[data-baseweb="menu-item"],
        div[data-testid="stSidebar"] div[role="listbox"] div[role="option"],
        div[data-testid="stSidebar"] li[data-baseweb="menu-item"] {
            padding: 10px 16px 10px 28px !important;
            margin: 2px 4px !important;
            border-radius: 6px !important;
            transition: all 0.2s ease !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
        }

        div[data-testid="stSidebar"] div[data-baseweb="menu"] div[data-baseweb="menu-item"]:hover,
        div[data-testid="stSidebar"] div[role="listbox"] div[role="option"]:hover,
        div[data-testid="stSidebar"] li[data-baseweb="menu-item"]:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            transform: translateX(2px) !important;
        }
        """)

        if css_rules:
            full_css = f"""
            <style>
            /* Strategy Selectbox Color Indicators */
            {chr(10).join(css_rules)}
            </style>
            """
            st.markdown(full_css, unsafe_allow_html=True)


class StrategyDisplayFormatter:
    """Handles strategy display formatting. Follows SRP."""

    @staticmethod
    def format_strategy_option(strategy_id: str, configs: dict) -> str:
        """Format strategy display with key stats for dropdown."""
        if strategy_id not in configs:
            return get_strategy_display_name(strategy_id)

        cfg_info = configs[strategy_id]
        display = cfg_info.display_name

        # Build compact stat string
        stats_parts = []
        stats_parts.append(f"{len(cfg_info.factors)}F")  # Factor count
        stats_parts.append(f"{cfg_info.long_seats}L")    # Long seats
        if cfg_info.short_seats > 0:
            stats_parts.append(f"{cfg_info.short_seats}S")  # Short seats if any
        stats_parts.append(f"{cfg_info.target_vol:.0%}σ")   # Target vol

        return f"{display} ({' | '.join(stats_parts)})"


class StrategySelectionSection(ISidebarSection):
    """Handles strategy selection UI. Follows SRP."""

    def __init__(self, strategy_selector: IStrategySelector):
        self.strategy_selector = strategy_selector

    def render(self, sidebar_state: 'SidebarState') -> 'SidebarState':
        """Render strategy selection and return updated state."""
        available_strategies = self.strategy_selector.get_available_strategies()

        if not available_strategies:
            st.warning("No strategies registered. Falling back to legacy mode.")
            return replace(sidebar_state, selected_strategy=None)

        # Build color map for consistent strategy colors
        color_map = build_strategy_color_map(available_strategies)

        # Inject CSS for selectbox styling
        SidebarCSSInjector.inject_strategy_colors(color_map)

        # Get default strategy
        default_strategy = self.strategy_selector.get_default_strategy(available_strategies)

        # Initialize session state
        if "q23_active_strategy" not in st.session_state:
            st.session_state.q23_active_strategy = default_strategy

        selected_strategy = st.session_state.q23_active_strategy
        if selected_strategy not in available_strategies:
            selected_strategy = default_strategy
            st.session_state.q23_active_strategy = selected_strategy

        # Derive index for selectbox
        try:
            idx = available_strategies.index(selected_strategy)
        except ValueError:
            idx = 0

        # Render selectbox
        configs = self.strategy_selector.get_strategy_configs()
        selected_strategy = st.selectbox(
            "Active Strategy",
            options=available_strategies,
            format_func=lambda x: StrategyDisplayFormatter.format_strategy_option(x, configs),
            index=idx,
            key="q23_active_strategy",
            help="Select strategy • Stats: Factors | Longs | Vol target",
        )

        # Show strategy info
        if selected_strategy and selected_strategy in configs:
            self._render_strategy_info(configs[selected_strategy])

        return replace(sidebar_state, selected_strategy=selected_strategy)

    def _render_strategy_info(self, cfg_info) -> None:
        """Render strategy information panel."""
        st.caption(f"📊 v{cfg_info.version} | {len(cfg_info.factors)} factors | {cfg_info.min_date}+")

        # Position configuration
        pos_info = []
        if cfg_info.long_seats > 0:
            pos_info.append(f"Long: {cfg_info.long_seats}")
        if cfg_info.short_seats > 0:
            pos_info.append(f"Short: {cfg_info.short_seats}")
        if cfg_info.long_only:
            pos_info.append("Long Only")

        if pos_info:
            st.caption(f"📈 {' | '.join(pos_info)}")

        # Risk parameters
        st.caption(f"🎯 Target Vol: {cfg_info.target_vol:.0%} | Max Pos: {cfg_info.max_pos:.0%}")


class TagSelectionSection(ISidebarSection):
    """Handles tag selection UI. Follows SRP."""

    def __init__(self, tag_resolver: ITagResolver):
        self.tag_resolver = tag_resolver

    def render(self, sidebar_state: 'SidebarState') -> 'SidebarState':
        """Render tag selection and return updated state."""
        if not sidebar_state.selected_strategy:
            return sidebar_state._replace(tag="")

        tags = self.tag_resolver.get_tags_for_strategy(sidebar_state.selected_strategy)

        if not tags:
            st.warning(f"No runs found for {sidebar_state.selected_strategy}")
            return replace(sidebar_state, tag="")

        # Get default tag
        default_tag = self.tag_resolver.get_default_tag(tags, sidebar_state.selected_strategy)

        # Initialize session state
        if "q23_selected_tag" not in st.session_state or \
           st.session_state.get("_prev_strategy") != sidebar_state.selected_strategy:
            st.session_state.q23_selected_tag = default_tag
            st.session_state._prev_strategy = sidebar_state.selected_strategy

        selected_tag = st.session_state.q23_selected_tag
        if selected_tag not in tags:
            selected_tag = default_tag
            st.session_state.q23_selected_tag = selected_tag

        # Render selectbox
        selected_tag = st.selectbox(
            "Run Tag",
            options=tags,
            index=tags.index(selected_tag),
            key="q23_selected_tag",
            help="Select strategy run • Newest runs appear first"
        )

        return replace(sidebar_state, tag=selected_tag)


class DateRangeSelectionSection(ISidebarSection):
    """Handles date range selection UI. Follows SRP."""

    def __init__(self, date_selector: IDateRangeSelector):
        self.date_selector = date_selector

    def render(self, sidebar_state: 'SidebarState') -> 'SidebarState':
        """Render date range selection and return updated state."""
        available_ranges = self.date_selector.get_available_ranges()

        # Date range selection
        st.subheader("📅 Date Range")

        range_options = list(available_ranges.keys())
        default_range = self.date_selector.get_default_range()

        # Find current selection
        current_range_key = None
        for key, (start, end) in available_ranges.items():
            if (start, end) == sidebar_state.date_range:
                current_range_key = key
                break

        selected_range_key = st.selectbox(
            "Quick Ranges",
            options=range_options,
            index=range_options.index(current_range_key) if current_range_key else 0,
            key="date_range_preset",
            help="Select predefined date ranges"
        )

        selected_range = available_ranges[selected_range_key]

        # Custom date range
        use_custom = st.checkbox("Use Custom Dates", key="use_custom_dates")

        if use_custom:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime(selected_range[0]),
                    key="custom_start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=pd.to_datetime(selected_range[1]),
                    key="custom_end_date"
                )

            if start_date and end_date:
                selected_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        return replace(sidebar_state, date_range=selected_range)


@dataclass
class SidebarState:
    """Container for all sidebar selections."""
    # Strategy selection
    use_multi_strategy: bool
    selected_strategy: Optional[str]
    
    # Date range
    date_range: Tuple[str, str]
    
    # Run selection
    tag: str
    base_dir: Path
    base_name: str
    
    # Navigation
    page: str
    elite_page: str
    
    # Transaction costs
    tc_config: Optional[TransactionCostConfig] = None


# Page options (Performance deprecated - merged into Overview)
MAIN_PAGES = [
    "Overview",
    "Weights",
    "Rebalance",  # Moved to position #3 for easier access
    "Factors/IC",
    "Attribution",
    "Bias",
    "Diagnostics",
    "What-If",
    "Live Strategy",
    "Strategy Warehouse",
    "Blotter",
    "Strategy Comparison",
    "Position Stack",
    "Stock Analysis",
    "Stock Elite",
    "Calendar Heatmap",
]

ELITE_PAGES = [
    "Summary",
    "PM Risk Dashboard",
    "Risk Attribution",
    "Brinson Attribution",
    "Ex-Ante Risk",
    "Sector Analysis",
    "Correlation Analysis",
    "Beta Analysis",
    "Regime Analysis",
    "Tail Risk & Stress",
    "Convexity & Gamma",
    "Alpha Decay",
    "Capacity Estimation",
    "Factor Timing",
    "Rotation Velocity",
    "Rank IC & Quintiles",
]


def _find_strategy_with_latest_run(strategies: List[str]) -> Tuple[Optional[str], int]:
    """
    Find the strategy with the most recent run and return (strategy_id, index).
    
    This ensures the dashboard auto-selects the strategy that was most recently
    executed, providing a better default experience for PMs.
    
    Args:
        strategies: List of available strategy IDs
        
    Returns:
        Tuple of (strategy_id, index) for the strategy with latest run,
        or (first_strategy, 0) if no runs found
    """
    if not strategies:
        return None, 0
    
    latest_time = None
    latest_idx = 0
    
    for idx, sid in enumerate(strategies):
        tags = discover_strategy_tags(sid)
        if tags:
            try:
                # Tags are sorted newest-first by discover_strategy_tags
                tag_time = pd.to_datetime(tags[0])
                if latest_time is None or tag_time > latest_time:
                    latest_time = tag_time
                    latest_idx = idx
            except (ValueError, TypeError):
                # Tag couldn't be parsed as date, skip
                pass
    
    return strategies[latest_idx], latest_idx


class SidebarRenderer:
    """Main sidebar renderer using composition and dependency injection. Follows SOLID principles."""

    def __init__(self):
        # Dependency injection following DIP
        self.strategy_selector = StrategySelector()
        self.tag_resolver = TagResolver()
        self.date_selector = DateRangeSelector()

        # Section composition following OCP
        self.sections = [
            StrategySelectionSection(self.strategy_selector),
            TagSelectionSection(self.tag_resolver),
            DateRangeSelectionSection(self.date_selector),
        ]

    def render_sidebar(self) -> Optional[SidebarState]:
        """
        Render the sidebar using composed sections.

        Returns:
            SidebarState with all selections, or None if configuration is invalid
        """
        # Initialize state
        sidebar_state = self._initialize_sidebar_state()

        if not sidebar_state:
            return None

        # Render sections using composition
        with st.sidebar:
            SidebarCSSInjector.inject_wide_dropdown_css()

            for section in self.sections:
                sidebar_state = section.render(sidebar_state)

            # Render remaining sections (transaction costs, navigation, etc.)
            sidebar_state = self._render_additional_sections(sidebar_state)

        return sidebar_state

    def _initialize_sidebar_state(self) -> Optional[SidebarState]:
        """Initialize the sidebar state based on available strategies."""
        available_strategies = self.strategy_selector.get_available_strategies()

        if not available_strategies:
            # Legacy mode fallback
            base_dir = _base_output_dir()
            base_name = cfg.paths.BASE_NAME
            tags = _discover_tags(base_dir, base_name)

            return SidebarState(
                use_multi_strategy=False,
                selected_strategy=None,
                date_range=get_default_date_range(),
                tag=tags[0] if tags else "",
                base_dir=base_dir,
                base_name=base_name,
                tc_config=get_session_tc_config(),
                page="Overview",
                elite_page="Summary",
            )

        # Multi-strategy mode
        default_strategy = self.strategy_selector.get_default_strategy(available_strategies)

        # Initialize session state
        if "q23_active_strategy" not in st.session_state:
            st.session_state.q23_active_strategy = default_strategy

        selected_strategy = st.session_state.q23_active_strategy
        if selected_strategy not in available_strategies:
            selected_strategy = default_strategy
            st.session_state.q23_active_strategy = selected_strategy

        # Get strategy-specific paths and tags
        strategy_dir = get_strategy_output_dir(selected_strategy)
        tags = self.tag_resolver.get_tags_for_strategy(selected_strategy)
        default_tag = self.tag_resolver.get_default_tag(tags, selected_strategy) if tags else ""

        return SidebarState(
            use_multi_strategy=True,
            selected_strategy=selected_strategy,
            date_range=self.date_selector.get_default_range(),
            tag=default_tag,
            base_dir=strategy_dir,
            base_name=selected_strategy,
            tc_config=get_session_tc_config(),
            page="Overview",
            elite_page="Summary",
        )

    def _render_additional_sections(self, sidebar_state: SidebarState) -> SidebarState:
        """Render additional sidebar sections (transaction costs, navigation, etc.)."""
        # Transaction cost configuration
        sidebar_state = self._render_transaction_cost_section(sidebar_state)

        # Navigation sections
        sidebar_state = self._render_navigation_section(sidebar_state)

        return sidebar_state

    def _render_transaction_cost_section(self, sidebar_state: SidebarState) -> SidebarState:
        """Render transaction cost configuration section."""
        st.subheader("💰 Transaction Costs")

        tc_config = sidebar_state.tc_config

        # Scheme selection with user-friendly names
        scheme_display_map = {
            TransactionCostScheme.QUANTIACS_ATR.value: "Quantiacs ATR (5% of ATR(14))",
            TransactionCostScheme.FLAT_BPS.value: "Flat Basis Points",
            TransactionCostScheme.PERCENTAGE.value: "Percentage of Trade Value",
            TransactionCostScheme.TIERED.value: "Tiered by Trade Size",
            TransactionCostScheme.CUSTOM.value: "Custom Function",
        }

        scheme_options = list(scheme_display_map.keys())
        display_options = [scheme_display_map[opt] for opt in scheme_options]

        current_idx = scheme_options.index(tc_config.scheme.value) if tc_config.scheme.value in scheme_options else 0

        selected_display = st.selectbox(
            "Cost Model",
            options=display_options,
            index=current_idx,
            key="tc_scheme",
            help="Transaction cost calculation method"
        )

        # Get the actual enum value from display selection
        selected_scheme = scheme_options[display_options.index(selected_display)]

        # Parameters based on scheme - use correct attribute names
        if selected_scheme == TransactionCostScheme.FLAT_BPS.value:
            flat_bps = st.number_input(
                "Basis Points",
                value=tc_config.flat_bps,
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key="tc_flat_bps",
                help="Transaction cost in basis points"
            )
            # Create updated config with correct attribute
            updated_tc_config = TransactionCostConfig(
                scheme=TransactionCostScheme.FLAT_BPS,
                flat_bps=flat_bps,
                atr_multiplier=tc_config.atr_multiplier,
                percentage=tc_config.percentage,
            )

        elif selected_scheme == TransactionCostScheme.QUANTIACS_ATR.value:
            atr_multiplier = st.number_input(
                "ATR Multiplier",
                value=tc_config.atr_multiplier,
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                key="tc_atr_mult",
                help="ATR multiplier for transaction costs (default: 0.05 = 5% of ATR(14))"
            )
            # Create updated config with correct attribute
            updated_tc_config = TransactionCostConfig(
                scheme=TransactionCostScheme.QUANTIACS_ATR,
                atr_multiplier=atr_multiplier,
                flat_bps=tc_config.flat_bps,
                percentage=tc_config.percentage,
            )

        elif selected_scheme == TransactionCostScheme.PERCENTAGE.value:
            percentage = st.number_input(
                "Percentage",
                value=tc_config.percentage * 100,  # Convert to percentage for display
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                key="tc_percentage",
                help="Transaction cost as percentage of trade value"
            ) / 100  # Convert back to decimal

            # Create updated config with correct attribute
            updated_tc_config = TransactionCostConfig(
                scheme=TransactionCostScheme.PERCENTAGE,
                percentage=percentage,
                flat_bps=tc_config.flat_bps,
                atr_multiplier=tc_config.atr_multiplier,
            )

        elif selected_scheme == TransactionCostScheme.TIERED.value:
            st.info("Tiered transaction cost scheme uses predefined thresholds")
            # For now, keep existing config for tiered scheme
            updated_tc_config = tc_config._replace(scheme=TransactionCostScheme.TIERED)

        elif selected_scheme == TransactionCostScheme.CUSTOM.value:
            st.info("Custom transaction cost function will be used")
            updated_tc_config = tc_config._replace(scheme=TransactionCostScheme.CUSTOM)

        else:
            updated_tc_config = tc_config

        # Save to session if changed
        if updated_tc_config != tc_config:
            set_session_tc_config(updated_tc_config)

        return replace(sidebar_state, tc_config=updated_tc_config)

    def _render_navigation_section(self, sidebar_state: SidebarState) -> SidebarState:
        """Render navigation section for page selection."""
        st.subheader("🧭 Navigation")

        # Main page selection - using radio buttons as requested
        st.markdown("**Main Page**")
        selected_page = st.radio(
            "Select main analysis page",
            options=PAGE_OPTIONS,
            index=PAGE_OPTIONS.index(sidebar_state.page) if sidebar_state.page in PAGE_OPTIONS else 0,
            key="main_page_radio",
            label_visibility="collapsed",
            help="Select main analysis page"
        )

        # Elite analytics page selection - using radio buttons as requested
        st.markdown("**Elite Analytics**")
        selected_elite_page = st.radio(
            "Select elite analytics overlay",
            options=ELITE_PAGE_OPTIONS,
            index=ELITE_PAGE_OPTIONS.index(sidebar_state.elite_page) if sidebar_state.elite_page in ELITE_PAGE_OPTIONS else 0,
            key="elite_page_radio",
            label_visibility="collapsed",
            help="Select elite analytics overlay"
        )

        return replace(sidebar_state, page=selected_page, elite_page=selected_elite_page)


# Global instance following Singleton pattern for consistent behavior
_sidebar_renderer = SidebarRenderer()

def render_sidebar() -> Optional[SidebarState]:
    """
    Render the sidebar using SOLID-compliant architecture.

    Returns:
        SidebarState with all selections, or None if configuration is invalid
    """
    return _sidebar_renderer.render_sidebar()
    
    with st.sidebar:
        # Inject CSS for wider dropdown
        _inject_wide_dropdown_css()
        
        # =====================================================================
        # STRATEGY SELECTION
        # =====================================================================
        st.header("Strategy Selection")
        
        if not available_strategies:
            st.warning("No strategies registered. Falling back to legacy mode.")
            selected_strategy = None
        else:
            # Build color map for consistent strategy colors
            color_map = build_strategy_color_map(available_strategies)

            # Inject CSS for selectbox option colors
            _inject_strategy_selectbox_colors(color_map)

            # Derive index from current state
            try:
                idx = available_strategies.index(st.session_state.q23_active_strategy)
            except Exception:
                idx = 0

            selected_strategy = st.selectbox(
                "Active Strategy",
                options=available_strategies,
                format_func=lambda x: _format_strategy_option(x, strategy_configs),
                index=idx,
                key="q23_active_strategy",
                help="Select strategy • Stats: Factors | Longs | Vol target",
            )
            
            # Enhanced mini info section with more stats
            if selected_strategy and selected_strategy in strategy_configs:
                cfg_info = strategy_configs[selected_strategy]
                # Row 1: Version and core info
                st.caption(f"📊 v{cfg_info.version} | {len(cfg_info.factors)} factors | {cfg_info.min_date}+")
                # Row 2: Position config
                seats_info = f"{cfg_info.long_seats}L"
                if cfg_info.short_seats > 0:
                    seats_info += f"/{cfg_info.short_seats}S"
                st.caption(f"🎯 {seats_info} | {cfg_info.target_vol:.0%} vol | {cfg_info.max_pos:.0%} max pos")
                # Row 3: Show active run tag (hints user can change in Ops)
                if active_tag:
                    st.caption(f"📁 Run: `{active_tag}`")
        
        st.divider()
        
        # =====================================================================
        # DATE RANGE
        # =====================================================================
        st.header("Date Range")
        
        preset_options = ["2025", "2024", "2023", "2022", "2021", "2020", "2024-25", "All"]
        default_preset = st.session_state.get("admin_default_date_preset", "2025")
        if "q23_date_preset" not in st.session_state:
            st.session_state.q23_date_preset = default_preset if default_preset in preset_options else preset_options[0]

        date_preset = st.radio(
            "Quick Select",
            preset_options,
            index=preset_options.index(st.session_state.q23_date_preset) if st.session_state.q23_date_preset in preset_options else 0,
            horizontal=True,
            key="q23_date_preset",
        )
        
        preset_start, preset_end = get_date_preset_range(date_preset)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "Start",
                value=pd.to_datetime(preset_start),
                min_value=pd.to_datetime("2004-01-01"),
            )
        with col_d2:
            end_date = st.date_input(
                "End",
                value=pd.to_datetime(preset_end),
            )
        
        date_range = (str(start_date), str(end_date))
        
        st.divider()
        
        # =====================================================================
        # RUN SELECTION (tag already computed above, just verify)
        # =====================================================================
        if not tags:
            _show_no_tags_error(base_dir, base_name)
            return None
        
        tag = active_tag
        
        # =====================================================================
        # OPS SECTION (collapsible - Run Selection near the top for visibility)
        # =====================================================================
        with st.expander("🔧 Ops / Run Selection", expanded=False):
            # Tag selector with count info - prominent at top
            st.markdown("**📁 Select Run**")
            tag_label = f"Available Runs ({len(tags)})"
            selected_tag = st.selectbox(
                tag_label,
                options=tags,
                index=tags.index(tag) if tag in tags else 0,
                key="ops_tag_selector",
                help="Select a specific run • Latest is auto-selected on boot",
            )
            
            # Update session state if changed
            if selected_tag != st.session_state.q23_selected_tag:
                st.session_state.q23_selected_tag = selected_tag
                tag = selected_tag
                st.rerun()
            
            st.caption(f"✅ Active: `{tag}`")
            
            st.divider()
            
            # Strategy info
            st.markdown("**📂 Output Location**")
            if use_multi_strategy and selected_strategy:
                st.caption(f"Strategy: `{selected_strategy}`")
                st.caption(f"Dir: `{strategy_dir.name}`")
            else:
                st.caption(f"Project: `{proj.name}`")
                st.caption(f"BASE_NAME: `{base_name}`")
        
        st.divider()
        
        # =====================================================================
        # PAGE NAVIGATION
        # =====================================================================
        page = st.radio(
            "Page",
            MAIN_PAGES,
            index=0,
        )
        
        st.divider()
        
        # =====================================================================
        # TRANSACTION COSTS (collapsible)
        # =====================================================================
        tc_config = get_session_tc_config()
        
        with st.expander("💰 Transaction Costs", expanded=False):
            st.caption("Session-level TC override (admin defaults in Ops Console)")
            
            # Quick scheme selector
            scheme_options = [
                TransactionCostScheme.QUANTIACS_ATR,
                TransactionCostScheme.FLAT_BPS,
            ]
            scheme_labels = {
                TransactionCostScheme.QUANTIACS_ATR: "Quantiacs ATR (5% × ATR(14))",
                TransactionCostScheme.FLAT_BPS: "Flat BPS",
            }
            
            current_idx = 0
            for i, s in enumerate(scheme_options):
                if s == tc_config.scheme:
                    current_idx = i
                    break
            
            selected_scheme = st.selectbox(
                "TC Model",
                options=scheme_options,
                format_func=lambda x: scheme_labels.get(x, x.value),
                index=current_idx,
                key="sidebar_tc_scheme",
                help="Override TC model for this session",
            )
            
            # Show relevant parameter
            if selected_scheme == TransactionCostScheme.QUANTIACS_ATR:
                col1, col2 = st.columns(2)
                with col1:
                    atr_mult = st.number_input(
                        "ATR Mult",
                        min_value=0.01,
                        max_value=0.20,
                        value=tc_config.atr_multiplier,
                        step=0.01,
                        format="%.2f",
                        key="sidebar_tc_atr_mult",
                        help="5% = Quantiacs default",
                    )
                with col2:
                    atr_win = st.number_input(
                        "ATR Win",
                        min_value=5,
                        max_value=30,
                        value=tc_config.atr_window,
                        step=1,
                        key="sidebar_tc_atr_win",
                        help="14 = Quantiacs default",
                    )
                flat_bps = tc_config.flat_bps
            else:
                flat_bps = st.number_input(
                    "Basis Points",
                    min_value=0.0,
                    max_value=50.0,
                    value=tc_config.flat_bps,
                    step=1.0,
                    key="sidebar_tc_flat_bps",
                )
                atr_mult = tc_config.atr_multiplier
                atr_win = tc_config.atr_window
            
            # Apply button
            if st.button("Apply TC Settings", key="sidebar_tc_apply", type="secondary"):
                new_tc = TransactionCostConfig(
                    scheme=selected_scheme,
                    atr_window=atr_win,
                    atr_multiplier=atr_mult,
                    flat_bps=flat_bps,
                )
                set_session_tc_config(new_tc)
                tc_config = new_tc
                st.success("TC settings applied!")
                st.rerun()
            
            # Show effective rate
            if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
                st.caption(f"Est. ~{tc_config.atr_multiplier * 1.5 * 100:.1f} bps/trade (1.5% ATR)")
            else:
                st.caption(f"Fixed {tc_config.flat_bps:.1f} bps/trade")
        
        st.divider()
        
        # =====================================================================
        # ELITE ANALYTICS
        # =====================================================================
        show_elite = bool(st.session_state.get("admin_show_elite_overlay", True))
        if show_elite:
            elite_section = st.expander("⚡ ELITE ANALYTICS", expanded=True)
            with elite_section:
                elite_page = st.radio(
                    "Elite Features",
                    ELITE_PAGES,
                    index=0,
                    key="elite_page_selector",
                )
        else:
            elite_page = "Summary"

        # =====================================================================
        # ADMIN DOCK (fixed bottom)
        # =====================================================================
        try:
            qp = st.query_params
            view = str(qp.get("view", "main"))
        except Exception:
            view = "main"

        active_main = "active" if view != "admin" else ""
        active_ops = "active" if view == "admin" else ""

        st.markdown(
            f"""
            <div class="q23-admin-dock">
              <div class="dock-row">
                <a class="dock-item {active_main}" href="?view=main" title="Main">
                  <span aria-hidden="true">🏠</span>
                </a>
                <a class="dock-item {active_ops}" href="?view=admin" title="Ops Console">
                  <span aria-hidden="true">🛠</span>
                </a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Store base_dir for Live Strategy page
    if "strategy_base_dir" not in st.session_state:
        st.session_state.strategy_base_dir = base_dir
    
    # Get final TC config (may have been updated in sidebar)
    final_tc_config = get_session_tc_config()
    
    return SidebarState(
        use_multi_strategy=use_multi_strategy,
        selected_strategy=selected_strategy,
        date_range=date_range,
        tag=tag,
        base_dir=base_dir,
        base_name=base_name,
        page=page,
        elite_page=elite_page,
        tc_config=final_tc_config,
    )


def _show_no_tags_error(base_dir: Path, base_name: str) -> None:
    """Display helpful error message when no tags are found."""
    debug_info = []
    if base_dir.exists():
        all_csvs = list(base_dir.glob("*_wide_weights_*.csv"))
        if all_csvs:
            debug_info.append(f"Found {len(all_csvs)} wide_weights files:")
            for f in sorted(all_csvs)[:5]:
                debug_info.append(f"  - {f.name}")
            if len(all_csvs) > 5:
                debug_info.append(f"  ... and {len(all_csvs) - 5} more")
        else:
            debug_info.append(f"No `*_wide_weights_*.csv` files found in `{base_dir}`")
    else:
        debug_info.append(f"Directory `{base_dir}` does not exist")
    
    error_msg = (
        f"No tags found. Expected files like `{base_name}_wide_weights_<tag>.csv` under:\n"
        f"`{base_dir}`\n\n"
    )
    if debug_info:
        error_msg += "\n".join(debug_info) + "\n\n"
    error_msg += "Run strategies first."
    
    st.error(error_msg)
