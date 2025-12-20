from q23.dashboard.analytics.attribution import (
    compute_factor_attribution,
    compute_rolling_factor_regression,
    compute_factor_vs_idio_decomposition,
    compute_single_stock_attribution,
)
from q23.dashboard.analytics.metrics import (
    summary_exposure,
    top_positions,
    turnover_series,
    select_return_series,
    period_return,
    period_windows,
    drawdown,
    compute_trade_list,
    rolling_sharpe,
    hit_rate,
    winner_loser_sizing,
)
from q23.dashboard.analytics.viz import (
    setup_plot_style,
    create_cumulative_return_chart,
    create_drawdown_chart,
    create_rolling_metrics_chart,
    create_factor_ic_chart,
    create_calendar_heatmap,
    create_exposure_time_series,
    create_factor_attribution_chart,
)
from q23.dashboard.analytics.whatif import (
    WhatIfEngine,
    WhatIfParams,
    WhatIfResult,
    compute_metric_deltas,
)
from q23.dashboard.analytics.advanced import (
    RegimeDetector,
    TailRiskAnalyzer,
    ConvexityAnalyzer,
    AlphaDecayAnalyzer,
    CapacityEstimator,
    TransactionCostAnalyzer,
    MicrostructureAnalyzer,
)
from q23.dashboard.analytics.scenarios import (
    ScenarioAnalyzer,
    ABTestingEngine,
    ParameterSurfaceExplorer,
    RankICAnalyzer,
    CrossSectionalMomentumTracker,
    FactorTimingAnalyzer,
)
from q23.dashboard.analytics.performance import (
    compute_comprehensive_performance,
    format_performance_table,
    compute_tc_comparison,
)
from q23.dashboard.analytics.risk_analytics import (
    # Data classes
    RiskAttributionResult,
    BrinsonResult,
    ExAnteRiskResult,
    ForwardRiskProjection,
    ExpectedReturnEstimate,
    # Risk Attribution
    compute_risk_attribution,
    compute_rolling_risk_attribution,
    # Brinson Attribution
    brinson_attribution,
    # Ex-Ante Risk
    compute_ex_ante_risk,
    compute_factor_covariance,
    # Forward Risk Projections (Risk Cones)
    compute_forward_risk_cone,
    # Expected Return Estimation
    compute_expected_return_ic,
    compute_expected_return_drift,
    compute_signal_freshness,
    compute_expected_return_estimate,
    # Sector/Industry Attribution
    compute_sector_attribution,
    compute_sector_exposure_timeseries,
    compute_sector_risk_contribution,
    compute_industry_concentration,
    # Correlation Analysis
    compute_correlation_analysis,
    compute_correlation_matrix,
    compute_beta_analysis,
    compute_up_down_capture,
    # Utilities
    estimate_specific_risk,
)
from q23.dashboard.analytics.stock_analytics import (
    # Data classes
    StockSummary,
    TradeRecord,
    # Stock discovery
    get_available_stocks,
    get_current_holdings,
    filter_stocks_by_criteria,
    rank_stocks_by_metric,
    # Returns computation
    compute_stock_returns_from_weights,
    compute_stock_contribution,
    # Performance
    compute_stock_performance,
    compute_stock_factor_exposure,
    compare_stocks,
    # Diagnostics
    get_stock_diagnostics,
    analyze_trades,
    compute_stock_rolling_metrics,
    # Multi-stock
    compute_stock_correlation_matrix,
    # Alerts
    generate_stock_alerts,
    # Attribution
    compute_stock_attribution_enhanced,
)
from q23.dashboard.analytics.stock_elite_analytics import (
    # Data classes
    EliteStockMetrics,
    RegimePerformance,
    # Regime analysis
    compute_stock_regime_performance,
    get_regime_exposure_summary,
    # Tail risk
    compute_stock_tail_metrics,
    compute_rolling_tail_risk,
    # Beta analysis
    compute_stock_beta_analysis,
    compute_rolling_beta,
    # Distribution
    compute_stock_distribution_stats,
    compute_qq_data,
    # Signal quality
    compute_stock_signal_quality,
    # Risk decomposition
    compute_stock_risk_decomposition,
    # Scorecard
    compute_elite_scorecard,
)

__all__ = [
    # Attribution
    "compute_factor_attribution",
    "compute_rolling_factor_regression",
    "compute_factor_vs_idio_decomposition",
    "compute_single_stock_attribution",
    # Metrics
    "summary_exposure",
    "top_positions",
    "turnover_series",
    "select_return_series",
    "period_return",
    "period_windows",
    "drawdown",
    "compute_trade_list",
    "rolling_sharpe",
    "hit_rate",
    "winner_loser_sizing",
    # Visualization
    "setup_plot_style",
    "create_cumulative_return_chart",
    "create_drawdown_chart",
    "create_rolling_metrics_chart",
    "create_factor_ic_chart",
    "create_calendar_heatmap",
    "create_exposure_time_series",
    "create_factor_attribution_chart",
    # What-If
    "WhatIfEngine",
    "WhatIfParams",
    "WhatIfResult",
    "compute_metric_deltas",
    # Advanced Analytics
    "RegimeDetector",
    "TailRiskAnalyzer",
    "ConvexityAnalyzer",
    "AlphaDecayAnalyzer",
    "CapacityEstimator",
    "TransactionCostAnalyzer",
    "MicrostructureAnalyzer",
    # Scenarios
    "ScenarioAnalyzer",
    "ABTestingEngine",
    "ParameterSurfaceExplorer",
    "RankICAnalyzer",
    "CrossSectionalMomentumTracker",
    "FactorTimingAnalyzer",
    # Performance
    "compute_comprehensive_performance",
    "format_performance_table",
    "compute_tc_comparison",
    # Risk Analytics (NEW)
    "RiskAttributionResult",
    "BrinsonResult",
    "ExAnteRiskResult",
    "ForwardRiskProjection",
    "ExpectedReturnEstimate",
    "compute_risk_attribution",
    "compute_rolling_risk_attribution",
    "brinson_attribution",
    "compute_ex_ante_risk",
    "compute_factor_covariance",
    # Forward Risk Projections
    "compute_forward_risk_cone",
    # Expected Return Estimation
    "compute_expected_return_ic",
    "compute_expected_return_drift",
    "compute_signal_freshness",
    "compute_expected_return_estimate",
    # Sector/Industry
    "compute_sector_attribution",
    "compute_sector_exposure_timeseries",
    "compute_sector_risk_contribution",
    "compute_industry_concentration",
    "compute_correlation_analysis",
    "compute_correlation_matrix",
    "compute_beta_analysis",
    "compute_up_down_capture",
    "estimate_specific_risk",
    # Stock Analytics
    "StockSummary",
    "TradeRecord",
    "get_available_stocks",
    "get_current_holdings",
    "filter_stocks_by_criteria",
    "rank_stocks_by_metric",
    "compute_stock_returns_from_weights",
    "compute_stock_contribution",
    "compute_stock_performance",
    "compute_stock_factor_exposure",
    "compare_stocks",
    "get_stock_diagnostics",
    "analyze_trades",
    "compute_stock_rolling_metrics",
    "compute_stock_correlation_matrix",
    "generate_stock_alerts",
    "compute_stock_attribution_enhanced",
    # Stock Elite Analytics
    "EliteStockMetrics",
    "RegimePerformance",
    "compute_stock_regime_performance",
    "get_regime_exposure_summary",
    "compute_stock_tail_metrics",
    "compute_rolling_tail_risk",
    "compute_stock_beta_analysis",
    "compute_rolling_beta",
    "compute_stock_distribution_stats",
    "compute_qq_data",
    "compute_stock_signal_quality",
    "compute_stock_risk_decomposition",
    "compute_elite_scorecard",
]
