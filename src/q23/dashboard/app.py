"""
Q23 ELITE PM Dashboard

Main application entry point with page routing.
"""

from __future__ import annotations

from typing import Callable
import streamlit as st


# =============================================================================
# ERROR BOUNDARY
# =============================================================================

def safe_render(render_fn: Callable, page_name: str = "Page") -> None:
    """
    Safely execute a page render function with error handling.
    
    Args:
        render_fn: The page render function (lambda or callable)
        page_name: Name of the page for error messages
    """
    try:
        render_fn()
    except Exception as e:
        st.error(f"Error rendering {page_name}: {str(e)}")
        with st.expander("Error Details", expanded=False):
            st.exception(e)
        st.info("Try refreshing the page or selecting a different view.")

from q23.dashboard.analytics import summary_exposure
from q23.dashboard.core import (
    load_dashboard_data,
    load_strategy_dashboard_data,
    get_strategy_display_name,
)
from q23.dashboard.components import inject_custom_css
from q23.dashboard.admin_settings import initialize_admin_defaults
from q23.dashboard._pages import (
    # Sidebar
    SidebarState,
    render_sidebar,
    # Admin
    render_admin_panel,
    # Main pages
    render_overview_page,
    # render_performance_page,  # Deprecated - merged into Overview
    render_weights_page,
    render_factors_page,
    render_attribution_page,
    render_diagnostics_page,
    render_whatif_page,
    render_blotter_page,
    render_rebalance_page,
    render_live_strategy_page,
    render_strategy_warehouse,
    render_strategy_comparison,
    render_stock_analysis_page,
    render_stock_elite_analytics_page,
    # Elite analytics
    render_regime_analysis_page,
    render_tail_risk_page,
    render_convexity_page,
    render_alpha_decay_page,
    render_capacity_page,
    render_ab_testing_page,
    render_factor_timing_page,
    render_rotation_page,
    render_rank_ic_page,
    # Risk analytics
    render_risk_attribution_page,
    render_brinson_attribution_page,
    render_ex_ante_risk_page,
    render_sector_analysis_page,
    render_correlation_analysis_page,
    render_beta_analysis_page,
    render_pm_executive_summary,
    render_pm_risk_dashboard,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="Q23 ELITE PM Dashboard", layout="wide")

# Inject global CSS for consistent styling across all pages
inject_custom_css()

st.title("Quantiacs Q23 PM Dashboard")

# Seed persisted admin defaults before any widgets render
initialize_admin_defaults()

# =============================================================================
# SIDEBAR - Navigation and Selection
# =============================================================================

sidebar_state = render_sidebar()

if sidebar_state is None:
    # No tags found - error displayed in sidebar
    st.stop()

# =============================================================================
# VIEW ROUTING (main vs admin)
# =============================================================================

try:
    qp = st.query_params
    view = str(qp.get("view", "main"))
    panel = str(qp.get("panel", "settings"))
except Exception:
    view, panel = "main", "settings"

# =============================================================================
# DATA LOADING
# =============================================================================

if sidebar_state.use_multi_strategy and sidebar_state.selected_strategy:
    data = load_strategy_dashboard_data(
        sidebar_state.selected_strategy,
        sidebar_state.tag,
        date_range=sidebar_state.date_range,
    )
else:
    data = load_dashboard_data(
        sidebar_state.tag,
        sidebar_state.base_dir,
        sidebar_state.base_name,
    )

if data.weights is None or data.weights.empty:
    st.error("Could not load weights for this tag.")
    st.stop()

# Admin view renders its own panel and stops
if view == "admin":
    # panel is accepted for backward compatibility with older links
    safe_render(lambda: render_admin_panel(data, sidebar_state, default_panel=panel), page_name="Admin")
    st.stop()

# Compute exposure for pages that need it
expos = summary_exposure(data.w_last)

# Get display name for overview
strategy_display_name = (
    get_strategy_display_name(sidebar_state.selected_strategy)
    if sidebar_state.selected_strategy
    else "Strategy"
)

# =============================================================================
# PAGE ROUTING (Registry Pattern)
# =============================================================================

# Build page registry with callables
# Using lambdas to defer execution and capture current scope
PAGE_REGISTRY = {
    "Overview": lambda: render_overview_page(
        data,
        tag=sidebar_state.tag,
        date_range=sidebar_state.date_range,
        strategy_display_name=strategy_display_name,
    ),
    # Performance page deprecated - merged into Overview
    "Weights": lambda: render_weights_page(data),
    "Factors/IC": lambda: render_factors_page(data),
    "Attribution": lambda: render_attribution_page(data),
    "Diagnostics": lambda: render_diagnostics_page(data),
    "What-If": lambda: render_whatif_page(data, sidebar_state.tag),
    "Live Strategy": lambda: render_live_strategy_page(
        st.session_state.get("strategy_base_dir", sidebar_state.base_dir)
    ),
    "Strategy Warehouse": lambda: render_strategy_warehouse(),
    "Blotter": lambda: render_blotter_page(data, expos),
    "Rebalance": lambda: render_rebalance_page(data, sidebar_state.base_name),
    "Strategy Comparison": lambda: render_strategy_comparison(
        current_strategy=(
            sidebar_state.selected_strategy
            if sidebar_state.use_multi_strategy
            else ""
        ),
        date_range=sidebar_state.date_range,
    ),
    "Stock Analysis": lambda: render_stock_analysis_page(data),
    "Stock Elite": lambda: render_stock_elite_analytics_page(data),
}

# Execute the selected page with error boundary
page = sidebar_state.page
if page in PAGE_REGISTRY:
    safe_render(PAGE_REGISTRY[page], page_name=page)
else:
    st.error(f"Unknown page: {page}")

# =============================================================================
# ELITE ANALYTICS OVERLAY
# =============================================================================

elite_page = sidebar_state.elite_page

# Elite analytics routing
ELITE_REGISTRY = {
    "Summary": render_pm_executive_summary,
    "PM Risk Dashboard": render_pm_risk_dashboard,
    "Risk Attribution": render_risk_attribution_page,
    "Brinson Attribution": render_brinson_attribution_page,
    "Ex-Ante Risk": render_ex_ante_risk_page,
    "Sector Analysis": render_sector_analysis_page,
    "Correlation Analysis": render_correlation_analysis_page,
    "Beta Analysis": render_beta_analysis_page,
    "Regime Analysis": render_regime_analysis_page,
    "Tail Risk & Stress": render_tail_risk_page,
    "Convexity & Gamma": render_convexity_page,
    "Alpha Decay": render_alpha_decay_page,
    "Capacity Estimation": render_capacity_page,
    "Factor Timing": render_factor_timing_page,
    "Rotation Velocity": render_rotation_page,
    "Rank IC & Quintiles": render_rank_ic_page,
}

if bool(st.session_state.get("admin_show_elite_overlay", True)):
    if elite_page in ELITE_REGISTRY:
        safe_render(lambda: ELITE_REGISTRY[elite_page](data), page_name=elite_page)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "Q23 ELITE Dashboard — Multi-Strategy + What-If + "
    "Advanced Visualizations + Elite Analytics"
)
