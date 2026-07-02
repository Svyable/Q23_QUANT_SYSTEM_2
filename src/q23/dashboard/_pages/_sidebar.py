"""
Sidebar Component

Handles all sidebar navigation and selection logic.
Returns a SidebarState dataclass containing all user selections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
from q23.shared.config import cfg, TransactionCostConfig, TransactionCostScheme
from q23.dashboard.admin_settings import get_session_tc_config, set_session_tc_config


def _inject_wide_dropdown_css() -> None:
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
        /* Strategy option styling for better readability */
        div[data-testid="stSidebar"] li[role="option"] {
            padding: 8px 12px !important;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        /* Selected option styling */
        div[data-testid="stSidebar"] div[data-baseweb="select"] span {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_strategy_option(strategy_id: str, configs: dict) -> str:
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


def render_sidebar() -> Optional[SidebarState]:
    """
    Render the sidebar and return the current state of all selections.
    
    Returns:
        SidebarState with all selections, or None if configuration is invalid
        (e.g., no tags found)
    """
    proj = _project_root()
    date_defaults = get_default_date_range()
    
    # =========================================================================
    # PRE-COMPUTE: Discover strategies and tags before rendering UI
    # =========================================================================
    # Filter to only enabled strategies for sidebar (Strategy Warehouse shows all)
    available_strategies = get_enabled_strategy_ids(include_benchmarks=False)
    
    if not available_strategies:
        use_multi_strategy = False
        selected_strategy_id = None
        strategy_configs = {}
        base_dir = _base_output_dir()
        base_name = cfg.paths.BASE_NAME
        tags = _discover_tags(base_dir, base_name)
        strategy_dir = base_dir
    else:
        use_multi_strategy = True
        strategy_configs = get_strategy_configs()
        
        # Determine which strategy is selected
        default_strategy_id = st.session_state.get("admin_default_strategy_id")
        latest_strategy, latest_idx = _find_strategy_with_latest_run(available_strategies)
        
        if "q23_active_strategy" not in st.session_state:
            if default_strategy_id in available_strategies:
                st.session_state.q23_active_strategy = default_strategy_id
            else:
                st.session_state.q23_active_strategy = latest_strategy
        
        selected_strategy_id = st.session_state.q23_active_strategy
        if selected_strategy_id not in available_strategies:
            selected_strategy_id = latest_strategy
            st.session_state.q23_active_strategy = selected_strategy_id
        
        strategy_dir = get_strategy_output_dir(selected_strategy_id)
        base_name = selected_strategy_id
        base_dir = strategy_dir
        tags = discover_strategy_tags(selected_strategy_id)
    
    # Compute active tag
    if tags:
        if "q23_selected_tag" not in st.session_state or st.session_state.get("_prev_strategy") != selected_strategy_id:
            st.session_state.q23_selected_tag = tags[0]
            st.session_state._prev_strategy = selected_strategy_id
        if st.session_state.q23_selected_tag not in tags:
            st.session_state.q23_selected_tag = tags[0]
        active_tag = st.session_state.q23_selected_tag
    else:
        active_tag = None
    
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
        
        # Build preset options dynamically based on current year
        today = pd.Timestamp.now()
        current_year = today.year
        
        preset_options = []
        
        # Add current year YTD if we're in that year
        if current_year >= 2026:
            preset_options.append("2026")
        
        # Add full years in reverse chronological order (most recent first)
        for year in range(current_year - 1, 2019, -1):
            preset_options.append(str(year))
        
        # Add multi-year and All options at the end
        preset_options.extend(["All"])
        
        # Determine default preset
        default_preset = st.session_state.get("admin_default_date_preset", "2025")
        if default_preset not in preset_options:
            # Try to find a sensible default
            if current_year >= 2026 and "2026" in preset_options:
                default_preset = "2026"
            elif "2025" in preset_options:
                default_preset = "2025"
            else:
                default_preset = preset_options[0] if preset_options else "All"
        
        if "q23_date_preset" not in st.session_state:
            st.session_state.q23_date_preset = default_preset

        date_preset = st.radio(
            "Quick Select",
            preset_options,
            index=preset_options.index(st.session_state.q23_date_preset) if st.session_state.q23_date_preset in preset_options else 0,
            horizontal=True,
            key="q23_date_preset",
            help="Quick date range presets: Full years show only that calendar year. YTD shows year-to-date up to today. Use custom dates below for more control.",
        )
        
        preset_start, preset_end = get_date_preset_range(date_preset)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(preset_start),
                min_value=pd.to_datetime("2004-01-01"),
                help="Start date for analysis period",
            )
        with col_d2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(preset_end),
                help="End date for analysis period",
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
