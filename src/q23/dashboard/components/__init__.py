"""
Dashboard Components

Provides reusable UI components:
- Interactive charts (Plotly-based)
- PM insight panels
- Metric cards
- Summary widgets
"""
from q23.dashboard.components.charts import (
    PM_COLORS,
    PM_COLORSCALE,
    PM_DIVERGING,
    PLOTLY_AVAILABLE,
    get_plotly_layout,
    create_risk_gauge,
    create_waterfall_chart,
    create_treemap_chart,
    create_correlation_heatmap,
    create_risk_contribution_chart,
    create_time_series_chart,
    create_scatter_with_regression,
    create_bullet_chart,
    create_sparkline,
    create_multi_metric_card,
)

from q23.dashboard.components.insights import (
    InsightSeverity,
    InsightCategory,
    PMInsight,
    PMInsightEngine,
    render_insights_panel,
    generate_executive_summary,
)

from q23.dashboard.components.styles import (
    COLORS,
    inject_custom_css,
    inject_live_strategy_css,
    inject_overview_css,
    styled_metric_card,
    section_header,
    strategy_header,
    performance_badge,
    metric_value_class,
    style_dataframe,
)

__all__ = [
    # Chart utilities
    "PM_COLORS",
    "PM_COLORSCALE",
    "PM_DIVERGING",
    "PLOTLY_AVAILABLE",
    "get_plotly_layout",
    # Chart functions
    "create_risk_gauge",
    "create_waterfall_chart",
    "create_treemap_chart",
    "create_correlation_heatmap",
    "create_risk_contribution_chart",
    "create_time_series_chart",
    "create_scatter_with_regression",
    "create_bullet_chart",
    "create_sparkline",
    "create_multi_metric_card",
    # Insights
    "InsightSeverity",
    "InsightCategory",
    "PMInsight",
    "PMInsightEngine",
    "render_insights_panel",
    "generate_executive_summary",
    # Styles
    "COLORS",
    "inject_custom_css",
    "inject_live_strategy_css",
    "inject_overview_css",
    "styled_metric_card",
    "section_header",
    "strategy_header",
    "performance_badge",
    "metric_value_class",
    "style_dataframe",
]
