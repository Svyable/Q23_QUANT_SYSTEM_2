"""
Dashboard Pages

All page render functions for the Q23 Dashboard.
"""

# Sidebar
from q23.dashboard._pages._sidebar import (
    SidebarState,
    render_sidebar,
    MAIN_PAGES,
    ELITE_PAGES,
)

# Main pages
from q23.dashboard._pages._overview import render_overview_page
# from q23.dashboard._pages._performance import render_performance_page  # Deprecated - merged into Overview
from q23.dashboard._pages._weights import render_weights_page
from q23.dashboard._pages._factors import render_factors_page
from q23.dashboard._pages._attribution import render_attribution_page
from q23.dashboard._pages._diagnostics import render_diagnostics_page
from q23.dashboard._pages._bias import render_bias_page
from q23.dashboard._pages._whatif import render_whatif_page
from q23.dashboard._pages._blotter import render_blotter_page
from q23.dashboard._pages._rebalance import render_rebalance_page
from q23.dashboard._pages._live_strategy import render_live_strategy_page
from q23.dashboard._pages._strategy_warehouse import render_strategy_warehouse
from q23.dashboard._pages._strategy_comparison import render_strategy_comparison
from q23.dashboard._pages._position_stack import render_position_stack
from q23.dashboard._pages._stock_analysis import render_stock_analysis_page
from q23.dashboard._pages._stock_elite_analytics import render_stock_elite_analytics_page
from q23.dashboard._pages._calendar_heatmap import render_calendar_heatmap_page

# Admin panel
from q23.dashboard._pages._admin import render_admin_panel

# Elite analytics pages
from q23.dashboard._pages._elite_analytics import (
    render_regime_analysis_page,
    render_tail_risk_page,
    render_convexity_page,
    render_alpha_decay_page,
    render_capacity_page,
    render_ab_testing_page,
    render_factor_timing_page,
    render_rotation_page,
    render_rank_ic_page,
)

# Risk analytics pages
from q23.dashboard._pages._risk_analytics import (
    render_risk_attribution_page,
    render_brinson_attribution_page,
    render_ex_ante_risk_page,
    render_sector_analysis_page,
    render_correlation_analysis_page,
    render_beta_analysis_page,
    render_pm_executive_summary,
    render_pm_risk_dashboard,
)

__all__ = [
    # Sidebar
    "SidebarState",
    "render_sidebar",
    "MAIN_PAGES",
    "ELITE_PAGES",
    # Main pages
    "render_overview_page",
    # "render_performance_page",  # Deprecated
    "render_weights_page",
    "render_factors_page",
    "render_attribution_page",
    "render_diagnostics_page",
    "render_bias_page",
    "render_whatif_page",
    "render_blotter_page",
    "render_rebalance_page",
    "render_live_strategy_page",
    "render_strategy_warehouse",
    "render_strategy_comparison",
    "render_position_stack",
    "render_stock_analysis_page",
    "render_stock_elite_analytics_page",
    "render_calendar_heatmap_page",
    # Admin
    "render_admin_panel",
    # Elite analytics
    "render_regime_analysis_page",
    "render_tail_risk_page",
    "render_convexity_page",
    "render_alpha_decay_page",
    "render_capacity_page",
    "render_ab_testing_page",
    "render_factor_timing_page",
    "render_rotation_page",
    "render_rank_ic_page",
    # Risk analytics
    "render_risk_attribution_page",
    "render_brinson_attribution_page",
    "render_ex_ante_risk_page",
    "render_sector_analysis_page",
    "render_correlation_analysis_page",
    "render_beta_analysis_page",
    "render_pm_executive_summary",
    "render_pm_risk_dashboard",
]
