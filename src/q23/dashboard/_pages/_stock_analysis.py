"""
Stock Analysis Page

Comprehensive single-stock and multi-stock analysis with performance metrics,
factor exposures, attribution, and diagnostics.

PM Features:
- Deep dive single stock analysis with 4 tabs
- Multi-stock comparison with correlation matrix
- Stock screener with filtering and ranking
- Trade-by-trade analysis
- Automated alerts for risk management
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    get_available_stocks,
    compute_stock_returns_from_weights,
    compute_stock_performance,
    compute_stock_factor_exposure,
    compare_stocks,
    get_stock_diagnostics,
    compute_stock_attribution_enhanced,
    select_return_series,
)
from q23.dashboard.analytics.stock_elite_analytics import (
    compute_stock_rating,
    compute_stock_grades,
    compute_stock_signal_quality,
    compute_elite_scorecard,
    compute_batch_stock_ratings,
    compute_batch_stock_grades,
    StockRating,
    StockGrades,
    EliteStockMetrics,
)
from q23.dashboard.analytics.stock_analytics import (
    get_current_holdings,
    filter_stocks_by_criteria,
    rank_stocks_by_metric,
    compute_stock_contribution,
    analyze_trades,
    compute_stock_rolling_metrics,
    compute_stock_correlation_matrix,
    generate_stock_alerts,
    StockSummary,
    # Portfolio comparison analytics
    compute_spearman_ic_by_horizon,
    compute_decile_spread_analysis,
    compute_stock_vs_portfolio_returns,
    compute_rolling_stock_correlation,
    compute_rolling_stock_sharpe,
    # Charts
    create_stock_portfolio_equity_chart,
    create_rolling_stock_correlation_chart,
    create_rolling_stock_sharpe_chart,
    create_ic_by_horizon_chart,
    create_decile_spread_chart,
    create_decile_spread_ts_chart,
)
from q23.dashboard.analytics.performance import format_performance_table
from q23.dashboard.components.charts import (
    PLOTLY_AVAILABLE,
    create_time_series_chart,
    get_plotly_layout,
    PM_COLORS,
)

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    import plotly.express as px


def render_stock_analysis_page(data: DashboardData) -> None:
    """
    Render the Stock Analysis page.
    
    Args:
        data: Dashboard data bundle (strategy-specific)
    """
    st.subheader("📈 Stock Analysis")
    
    # Display strategy context
    strategy_name = data.base_name if hasattr(data, 'base_name') else "Current Strategy"
    st.caption(f"Analyzing stocks for strategy: **{strategy_name}** (Tag: {data.tag})")
    st.caption("Ratings and grades are computed from this strategy's factor signals, positions, and performance.")
    
    if data.weights is None or data.weights.empty:
        st.error("No weights data available for stock analysis.")
        return
    
    # Get available stocks
    available_stocks = get_available_stocks(data.weights)
    current_holdings = get_current_holdings(data.weights)
    
    if not available_stocks:
        st.warning("No stocks found in portfolio weights.")
        return
    
    # Quick stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks Traded", len(available_stocks))
    with col2:
        st.metric("Current Holdings", len(current_holdings))
    with col3:
        if data.factor_vectors is not None:
            winners = sum(1 for s in available_stocks 
                         if s in data.factor_vectors.index 
                         and data.factor_vectors.loc[s].get("total_pnl_contrib", 0) > 0)
            st.metric("Winners", winners)
        else:
            st.metric("Winners", "—")
    with col4:
        if data.factor_vectors is not None:
            losers = sum(1 for s in available_stocks 
                        if s in data.factor_vectors.index 
                        and data.factor_vectors.loc[s].get("total_pnl_contrib", 0) < 0)
            st.metric("Losers", losers)
        else:
            st.metric("Losers", "—")
    
    st.divider()
    
    # Mode selection
    mode = st.radio(
        "Analysis Mode",
        ["Single Stock", "Compare Stocks", "Portfolio Comparison", "Stock Screener", "Stock Stack", "BUY List"],
        horizontal=True,
        help="Analyze one stock, compare multiple stocks, compare to portfolio, screen/filter stocks, view comprehensive stock stack, or see BUY-rated stocks ranked by rating"
    )
    
    st.divider()
    
    if mode == "Single Stock":
        _render_single_stock_analysis(data, available_stocks, current_holdings)
    elif mode == "Compare Stocks":
        _render_multi_stock_comparison(data, available_stocks)
    elif mode == "Portfolio Comparison":
        _render_portfolio_comparison(data, available_stocks, current_holdings)
    elif mode == "Stock Stack":
        _render_stock_stack(data, available_stocks, current_holdings)
    elif mode == "BUY List":
        _render_buy_list(data, available_stocks, current_holdings)
    else:
        _render_stock_screener(data, available_stocks)


def _render_single_stock_analysis(
    data: DashboardData,
    available_stocks: List[str],
    current_holdings: List[str],
) -> None:
    """Render single stock analysis view."""
    
    # Stock selection with quick filters
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            options=available_stocks,
            help="Choose a stock to analyze in detail"
        )
    
    with col2:
        filter_current = st.checkbox(
            "Current only",
            value=False,
            help="Show only stocks currently in portfolio"
        )
        if filter_current:
            if selected_stock not in current_holdings:
                selected_stock = current_holdings[0] if current_holdings else None
    
    if not selected_stock:
        st.info("No stock selected.")
        return
    
    # Get portfolio returns for context
    portfolio_returns = select_return_series(data.diag)
    
    # Compute stock returns (approximate)
    stock_returns = compute_stock_returns_from_weights(
        data.weights,
        portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),
        selected_stock,
    )
    
    # Get stock weight series
    stock_weights = data.weights[selected_stock].fillna(0.0)
    
    # Stock info summary
    _render_stock_summary(selected_stock, stock_weights, data.factor_vectors)
    
    # Ratings and Grades Panel
    _render_ratings_grades_panel(
        selected_stock,
        stock_returns,
        stock_weights,
        portfolio_returns,
        data.factor_vectors,
    )
    
    # Alerts panel (collapsible)
    alerts = generate_stock_alerts(
        selected_stock,
        data.weights,
        data.factor_vectors,
        portfolio_returns,
    )
    
    if alerts:
        with st.expander(f"⚠️ Alerts ({len(alerts)})", expanded=False):
            for alert in alerts:
                severity_icon = "🔴" if alert["severity"] == "warning" else "🔵"
                st.markdown(f"{severity_icon} **{alert['category']}**: {alert['message']}")
                st.caption(alert["details"])
    
    st.divider()
    
    # Data accuracy notice
    st.caption(
        "ℹ️ Stock returns are approximated from portfolio data. "
        "For precise analysis, enable stock returns artifact saving."
    )
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⭐ Ratings & Grades",
        "📊 Performance",
        "🔍 Factors",
        "📈 Attribution",
        "📋 Trades",
        "🔧 Diagnostics",
    ])
    
    with tab1:
        _render_ratings_grades_tab(
            selected_stock,
            stock_returns,
            stock_weights,
            portfolio_returns,
            data.factor_vectors,
        )
    
    with tab2:
        _render_performance_tab(
            selected_stock,
            stock_returns,
            stock_weights,
            portfolio_returns,
            data,
        )
    
    with tab3:
        _render_factors_tab(selected_stock, data.factor_vectors)
    
    with tab4:
        _render_attribution_tab(
            selected_stock,
            data.weights,
            data.factor_vectors,
            portfolio_returns,
            stock_returns,
        )
    
    with tab5:
        _render_trades_tab(
            selected_stock,
            data.weights,
            portfolio_returns,
        )
    
    with tab6:
        _render_diagnostics_tab(
            selected_stock,
            data.weights,
            portfolio_returns,
            stock_returns,
        )


def _render_stock_screener(
    data: DashboardData,
    available_stocks: List[str],
) -> None:
    """Render stock screener with filtering and ranking."""
    
    st.markdown("### Stock Screener")
    st.caption("Filter and rank stocks by various criteria")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_only = st.checkbox("Current holdings only", value=False)
        side_filter = st.selectbox(
            "Side",
            ["All", "Long", "Short"],
            help="Filter by position direction"
        )
        side = None if side_filter == "All" else side_filter.upper()
    
    with col2:
        min_days = st.number_input("Min days held", min_value=0, value=0)
        min_pnl = st.number_input("Min P&L", value=-1.0, step=0.01, format="%.4f")
    
    with col3:
        rank_metric = st.selectbox(
            "Rank by",
            ["total_pnl_contrib", "pnl_per_day_held", "days_held", "current_weight", "avg_weight"],
            help="Metric to rank stocks by"
        )
        rank_ascending = st.checkbox("Ascending", value=False)
    
    # Apply filters
    filtered_stocks = filter_stocks_by_criteria(
        data.weights,
        data.factor_vectors,
        min_days_held=min_days,
        min_pnl=min_pnl if min_pnl > -1.0 else None,
        side=side,
        current_only=current_only,
    )
    
    st.markdown(f"**{len(filtered_stocks)} stocks match criteria**")
    
    if not filtered_stocks:
        st.info("No stocks match the selected criteria.")
        return
    
    # Get rankings
    rankings = rank_stocks_by_metric(
        data.weights,
        data.factor_vectors,
        metric=rank_metric,
        ascending=rank_ascending,
        top_n=50,
    )
    
    # Filter rankings to only include filtered stocks
    rankings = [(s, v) for s, v in rankings if s in filtered_stocks]
    
    # Build display table
    rows = []
    for symbol, rank_value in rankings[:30]:  # Show top 30
        row = {"Symbol": symbol, rank_metric: rank_value}
        
        # Add summary metrics
        if data.factor_vectors is not None and symbol in data.factor_vectors.index:
            fv_row = data.factor_vectors.loc[symbol]
            row["P&L"] = fv_row.get("total_pnl_contrib", 0.0)
            row["Days Held"] = int(fv_row.get("days_held", 0))
            row["Avg Weight"] = fv_row.get("avg_weight", 0.0)
        
        # Current weight
        if symbol in data.weights.columns:
            row["Current Weight"] = float(data.weights[symbol].iloc[-1])
        
        rows.append(row)
    
    if rows:
        screen_df = pd.DataFrame(rows)
        
        # Format for display
        for col in ["P&L", "Avg Weight", "Current Weight"]:
            if col in screen_df.columns:
                screen_df[col] = screen_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        
        st.dataframe(screen_df, width='stretch', hide_index=True)
        
        # Export
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Screener Results",
            data=csv_bytes,
            file_name=f"stock_screener_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def _render_multi_stock_comparison(
    data: DashboardData,
    available_stocks: List[str],
) -> None:
    """Render multi-stock comparison view."""
    
    # Multi-select stocks (up to 5)
    selected_stocks = st.multiselect(
        "Select Stocks to Compare",
        options=available_stocks,
        default=available_stocks[:min(3, len(available_stocks))] if available_stocks else [],
        max_selections=5,
        help="Select up to 5 stocks to compare side-by-side"
    )
    
    if not selected_stocks:
        st.info("Select at least one stock to compare.")
        return
    
    if len(selected_stocks) == 1:
        st.info("Select multiple stocks for comparison, or switch to Single Stock mode.")
        return
    
    st.divider()
    
    # Get portfolio returns
    portfolio_returns = select_return_series(data.diag)
    
    # Comparison table
    st.markdown("### Comparison Table")
    comparison_df = compare_stocks(
        selected_stocks,
        data.weights,
        data.factor_vectors if data.factor_vectors is not None else pd.DataFrame(),
        portfolio_returns,
        None,  # stock_returns not available yet
    )
    
    if not comparison_df.empty:
        # Format for display
        display_df = comparison_df.copy()
        
        # Format percentages
        pct_cols = ["current_weight", "avg_weight", "annual_return", "annual_vol", "max_drawdown"]
        for col in pct_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        
        # Format other metrics
        if "sharpe" in display_df.columns:
            display_df["sharpe"] = display_df["sharpe"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        
        if "total_pnl_contrib" in display_df.columns:
            display_df["total_pnl_contrib"] = display_df["total_pnl_contrib"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
        
        st.dataframe(display_df, width='stretch')
        
        # Download button
        csv_bytes = comparison_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Comparison CSV",
            data=csv_bytes,
            file_name=f"stock_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("Could not generate comparison data.")
    
    st.divider()
    
    # Performance charts comparison
    if PLOTLY_AVAILABLE and portfolio_returns is not None:
        st.markdown("### Performance Comparison")
        
        # OPTIMIZED: Compute all stock returns at once using vectorized operations
        # This is much faster than looping through each stock individually
        valid_stocks = [s for s in selected_stocks if s in data.weights.columns]
        if valid_stocks:
            stock_weights_subset = data.weights[valid_stocks].fillna(0.0)
            stock_weights_subset, port_ret = stock_weights_subset.align(
                portfolio_returns, join='inner', axis=0
            )
            
            if not stock_weights_subset.empty:
                # Vectorized computation
                w_lag = stock_weights_subset.shift(1).fillna(0.0)
                eps = 1e-12
                
                # Compute returns for all stocks at once
                stock_returns_df = pd.DataFrame(0.0, index=stock_weights_subset.index, columns=valid_stocks)
                for stock in valid_stocks:
                    mask = w_lag[stock].abs() > eps
                    if mask.any():
                        # Use np.divide with where to avoid division warnings
                        denominator = w_lag.loc[mask, stock] + eps
                        stock_returns_df.loc[mask, stock] = np.divide(
                            port_ret[mask],
                            denominator,
                            out=np.zeros_like(port_ret[mask], dtype=float),
                            where=(denominator != 0)
                        )
                
                stock_returns_df = stock_returns_df.clip(-0.5, 0.5).replace([np.inf, -np.inf], 0.0)
                
                # Cumulative returns
                cum_returns_df = (1 + stock_returns_df).cumprod() - 1
                
                fig = create_time_series_chart(
                    cum_returns_df,
                    title="Cumulative Returns Comparison",
                )
                if fig is not None:
                    st.plotly_chart(fig)
    
    st.divider()
    
    # Correlation matrix
    if PLOTLY_AVAILABLE and portfolio_returns is not None and len(selected_stocks) >= 2:
        st.markdown("### Correlation Matrix")
        
        corr_matrix = compute_stock_correlation_matrix(
            data.weights,
            portfolio_returns,
            selected_stocks,
        )
        
        if not corr_matrix.empty:
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1,
                zmax=1,
                title="Stock Return Correlations",
            )
            fig.update_layout(
                paper_bgcolor=PM_COLORS["background"],
                plot_bgcolor=PM_COLORS["card"],
                font={"color": PM_COLORS["text"]},
                height=400,
            )
            st.plotly_chart(fig)
    
    st.divider()
    
    # Factor exposure comparison
    if data.factor_vectors is not None and not data.factor_vectors.empty:
        st.markdown("### Factor Exposure Comparison")
        
        # Get factor exposures for selected stocks
        factor_exposures = {}
        summary_cols = {
            "total_pnl_contrib", "pnl_per_day_held", "mean_score",
            "score_vol", "days_held", "avg_weight", "avg_weight_when_held",
        }
        factor_cols = [c for c in data.factor_vectors.columns if c not in summary_cols]
        
        for stock in selected_stocks:
            if stock in data.factor_vectors.index:
                exposures = data.factor_vectors.loc[stock, factor_cols].to_dict()
                factor_exposures[stock] = exposures
        
        if factor_exposures:
            # Create comparison DataFrame
            exp_df = pd.DataFrame(factor_exposures).T
            
            # Show top factors by average absolute exposure
            top_factors = exp_df.abs().mean(axis=0).sort_values(ascending=False).head(15).index.tolist()
            exp_display = exp_df[top_factors]
            
            st.dataframe(
                exp_display.style.background_gradient(cmap="RdYlGn", axis=0, vmin=-2, vmax=2),
                width='stretch',
            )


def _render_portfolio_comparison(
    data: DashboardData,
    available_stocks: List[str],
    current_holdings: List[str],
) -> None:
    """
    Render Portfolio Comparison view with IC analysis, decile spreads,
    and stock vs portfolio comparison charts.
    """
    st.markdown("### Portfolio Comparison Analytics")
    st.markdown("""
    Compare individual stocks to the portfolio and analyze signal quality.
    - **Spearman IC**: Rank correlation between position weights and forward returns
    - **Decile Spread**: Top-weighted vs bottom-weighted stock performance
    - **Stock vs Portfolio**: Direct comparison of individual stock returns to portfolio
    """)
    
    portfolio_returns = select_return_series(data.diag)
    
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns not available for comparison analysis.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📊 Signal Quality (IC)",
        "📈 Decile Analysis",
        "🔄 Stock vs Portfolio",
    ])
    
    with tab1:
        _render_ic_analysis_tab(data.weights, portfolio_returns)
    
    with tab2:
        _render_decile_analysis_tab(data.weights, portfolio_returns)
    
    with tab3:
        _render_stock_portfolio_comparison_tab(
            data, available_stocks, current_holdings, portfolio_returns
        )


def _render_ic_analysis_tab(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> None:
    """Render Spearman IC analysis tab."""
    
    st.markdown("#### Spearman IC by Horizon")
    st.markdown("""
    **Information Coefficient (IC)** measures the rank correlation between 
    portfolio weights and subsequent returns. A positive IC suggests predictive power.
    
    - **IC > 0.05**: Excellent signal
    - **IC > 0.02**: Good signal
    - **IC ~ 0**: No signal
    - **Green bars**: Statistically significant (|t-stat| ≥ 2)
    """)
    
    # Compute settings
    col1, col2 = st.columns([1, 3])
    with col1:
        horizons = st.multiselect(
            "Horizons",
            options=[1, 5, 10, 21, 42, 63],
            default=[1, 5, 10, 21],
            help="Forward horizons to test (days)"
        )
    
    if not horizons:
        st.info("Select at least one horizon.")
        return
    
    with st.spinner("Computing IC by horizon..."):
        ic_df = compute_spearman_ic_by_horizon(
            weights=weights,
            portfolio_returns=portfolio_returns,
            horizons=sorted(horizons),
            min_obs=30,
        )
    
    if ic_df.empty:
        st.warning("Insufficient data to compute IC. Need more observations.")
        return
    
    # Display chart
    if PLOTLY_AVAILABLE:
        fig = create_ic_by_horizon_chart(ic_df)
        if fig is not None:
            st.plotly_chart(fig)
    
    # Display table
    st.markdown("##### IC Statistics")
    display_ic = ic_df.copy()
    display_ic['horizon'] = display_ic['horizon'].apply(lambda x: f"T+{x}")
    display_ic['ic_mean'] = display_ic['ic_mean'].apply(lambda x: f"{x:.4f}")
    display_ic['ic_std'] = display_ic['ic_std'].apply(lambda x: f"{x:.4f}")
    display_ic['t_stat'] = display_ic['t_stat'].apply(lambda x: f"{x:.2f}")
    display_ic['p_value'] = display_ic['p_value'].apply(lambda x: f"{x:.4f}")
    display_ic['hit_rate'] = display_ic['hit_rate'].apply(lambda x: f"{x:.1%}")
    
    display_ic = display_ic.rename(columns={
        'horizon': 'Horizon',
        'ic_mean': 'Mean IC',
        'ic_std': 'Std IC',
        't_stat': 'T-Stat',
        'p_value': 'P-Value',
        'n_obs': 'N',
        'hit_rate': 'Hit Rate',
    })
    
    st.dataframe(display_ic, width='stretch', hide_index=True)
    
    # Interpretation
    st.markdown("##### Interpretation")
    avg_ic = ic_df['ic_mean'].mean()
    significant_horizons = ic_df[ic_df['t_stat'].abs() >= 2]['horizon'].tolist()
    
    if avg_ic > 0.03:
        st.success(f"✅ **Strong Signal**: Average IC of {avg_ic:.4f} indicates good predictive power.")
    elif avg_ic > 0.01:
        st.info(f"ℹ️ **Moderate Signal**: Average IC of {avg_ic:.4f} suggests some predictive ability.")
    else:
        st.warning(f"⚠️ **Weak Signal**: Average IC of {avg_ic:.4f} indicates limited predictive power.")
    
    if significant_horizons:
        st.markdown(f"Significant horizons: {', '.join([f'T+{h}' for h in significant_horizons])}")


def _render_decile_analysis_tab(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> None:
    """Render decile spread analysis tab."""
    
    st.markdown("#### Decile Spread Analysis")
    st.markdown("""
    **Decile Spread** measures whether high-conviction positions (largest weights)
    outperform low-conviction positions. A positive spread indicates good
    position sizing.
    
    - Stocks ranked by |weight| into 10 buckets (decile 1 = highest conviction)
    - Spread = Top decile return - Bottom decile return
    """)
    
    # Compute settings
    col1, col2 = st.columns([1, 3])
    with col1:
        horizons = st.multiselect(
            "Horizons",
            options=[1, 5, 10, 21, 42, 63],
            default=[1, 5, 21],
            key="decile_horizons",
            help="Forward horizons to analyze (days)"
        )
    
    if not horizons:
        st.info("Select at least one horizon.")
        return
    
    with st.spinner("Computing decile spread analysis..."):
        result = compute_decile_spread_analysis(
            weights=weights,
            portfolio_returns=portfolio_returns,
            n_deciles=10,
            horizons=sorted(horizons),
            min_assets=10,
        )
    
    if not result or result.get('spread_by_horizon', pd.DataFrame()).empty:
        st.warning("Insufficient data for decile analysis. Need more assets per period.")
        return
    
    spread_df = result['spread_by_horizon']
    decile_returns = result['decile_returns']
    spread_ts = result.get('spread_ts', pd.DataFrame())
    
    # Display decile returns chart
    st.markdown("##### Returns by Conviction Decile")
    if PLOTLY_AVAILABLE and not decile_returns.empty:
        fig = create_decile_spread_chart(decile_returns, horizons=sorted(horizons))
        if fig is not None:
            st.plotly_chart(fig)
    
    # Display spread time series
    st.markdown("##### Top-Bottom Spread Over Time")
    if PLOTLY_AVAILABLE and not spread_ts.empty:
        fig = create_decile_spread_ts_chart(spread_ts)
        if fig is not None:
            st.plotly_chart(fig)
    
    # Display spread statistics table
    st.markdown("##### Spread Statistics")
    if not spread_df.empty:
        display_spread = spread_df.copy()
        display_spread['horizon'] = display_spread['horizon'].apply(lambda x: f"T+{x}")
        display_spread['spread_mean'] = display_spread['spread_mean'].apply(lambda x: f"{x:.3%}")
        display_spread['spread_std'] = display_spread['spread_std'].apply(lambda x: f"{x:.3%}")
        display_spread['t_stat'] = display_spread['t_stat'].apply(lambda x: f"{x:.2f}")
        display_spread['p_value'] = display_spread['p_value'].apply(lambda x: f"{x:.4f}")
        display_spread['hit_rate'] = display_spread['hit_rate'].apply(lambda x: f"{x:.1%}")
        
        display_spread = display_spread.rename(columns={
            'horizon': 'Horizon',
            'spread_mean': 'Mean Spread',
            'spread_std': 'Std Spread',
            't_stat': 'T-Stat',
            'p_value': 'P-Value',
            'n_obs': 'N',
            'hit_rate': 'Hit Rate',
        })
        
        st.dataframe(display_spread, width='stretch', hide_index=True)
    
    # Interpretation
    st.markdown("##### Interpretation")
    if not spread_df.empty:
        avg_spread = spread_df['spread_mean'].mean() if 'spread_mean' in spread_df.columns else 0
        avg_hit = spread_df['hit_rate'].mean() if 'hit_rate' in spread_df.columns else 0.5
        
        # Convert back from formatted string if needed
        if isinstance(avg_spread, str):
            avg_spread = float(avg_spread.replace('%', '')) / 100
        if isinstance(avg_hit, str):
            avg_hit = float(avg_hit.replace('%', '')) / 100
        
        if avg_spread > 0.005:  # 0.5% average spread
            st.success(f"✅ **Good Position Sizing**: High-conviction positions outperform by {avg_spread:.2%} on average.")
        elif avg_spread > 0:
            st.info(f"ℹ️ **Moderate Signal**: Positive spread of {avg_spread:.2%} suggests some skill in sizing.")
        else:
            st.warning(f"⚠️ **Negative Spread**: Low-conviction positions outperform. Consider reviewing sizing logic.")


def _render_stock_portfolio_comparison_tab(
    data: DashboardData,
    available_stocks: List[str],
    current_holdings: List[str],
    portfolio_returns: pd.Series,
) -> None:
    """Render stock vs portfolio comparison tab."""
    
    st.markdown("#### Stock vs Portfolio Comparison")
    st.markdown("""
    Compare individual stock performance against the portfolio.
    See cumulative returns, rolling correlation, and rolling Sharpe ratios.
    """)
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    with col1:
        default_stocks = current_holdings[:5] if current_holdings else available_stocks[:5]
        selected_stocks = st.multiselect(
            "Select Stocks to Compare",
            options=available_stocks,
            default=default_stocks,
            max_selections=8,
            help="Select up to 8 stocks to compare with portfolio"
        )
    
    with col2:
        window = st.selectbox(
            "Rolling Window",
            options=[21, 42, 63, 126],
            index=1,
            format_func=lambda x: f"{x}d (~{x//21}mo)",
            help="Rolling window for correlation and Sharpe calculations"
        )
    
    if not selected_stocks:
        st.info("Select stocks to compare against portfolio.")
        return
    
    # Compute cumulative returns
    cum_returns = compute_stock_vs_portfolio_returns(
        symbols=selected_stocks,
        weights=data.weights,
        portfolio_returns=portfolio_returns,
    )
    
    if cum_returns.empty:
        st.warning("Could not compute returns for selected stocks.")
        return
    
    # Display cumulative returns chart
    st.markdown("##### Cumulative Returns")
    if PLOTLY_AVAILABLE:
        fig = create_stock_portfolio_equity_chart(cum_returns)
        if fig is not None:
            st.plotly_chart(fig)
    
    # Rolling analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Rolling Correlation with Portfolio")
        rolling_corr = compute_rolling_stock_correlation(
            symbols=selected_stocks,
            weights=data.weights,
            portfolio_returns=portfolio_returns,
            window=window,
        )
        
        if PLOTLY_AVAILABLE and not rolling_corr.empty:
            fig = create_rolling_stock_correlation_chart(rolling_corr)
            if fig is not None:
                st.plotly_chart(fig)
        else:
            st.info("Insufficient data for rolling correlation.")
    
    with col2:
        st.markdown("##### Rolling Sharpe Ratio")
        rolling_sharpe = compute_rolling_stock_sharpe(
            symbols=selected_stocks,
            weights=data.weights,
            portfolio_returns=portfolio_returns,
            window=window,
        )
        
        if PLOTLY_AVAILABLE and not rolling_sharpe.empty:
            fig = create_rolling_stock_sharpe_chart(rolling_sharpe)
            if fig is not None:
                st.plotly_chart(fig)
        else:
            st.info("Insufficient data for rolling Sharpe.")
    
    # Summary statistics
    st.markdown("##### Summary Statistics")
    if not cum_returns.empty:
        summary_rows = []
        for col in cum_returns.columns:
            returns = cum_returns[col].diff().dropna() if col != "Portfolio" else portfolio_returns
            if len(returns) > 20:
                total_ret = cum_returns[col].iloc[-1] if len(cum_returns) > 0 else 0
                ann_ret = float(returns.mean() * 252)
                ann_vol = float(returns.std() * np.sqrt(252))
                # Use np.divide to avoid division warnings
                sharpe = np.divide(ann_ret, ann_vol + 1e-12, out=np.array([0.0]), where=(ann_vol + 1e-12) != 0)[0]
                
                # Correlation with portfolio
                if col != "Portfolio" and portfolio_returns is not None:
                    aligned_ret, aligned_port = returns.align(portfolio_returns, join='inner')
                    corr = float(aligned_ret.corr(aligned_port)) if len(aligned_ret) > 20 else 0
                else:
                    corr = 1.0
                
                summary_rows.append({
                    "Name": col,
                    "Total Return": f"{total_ret:.2%}",
                    "Ann. Return": f"{ann_ret:.2%}",
                    "Ann. Vol": f"{ann_vol:.2%}",
                    "Sharpe": f"{sharpe:.2f}",
                    "Corr w/ Portfolio": f"{corr:.2f}",
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, width='stretch', hide_index=True)


def _render_stock_summary(
    stock_symbol: str,
    stock_weights: pd.Series,
    factor_vectors: Optional[pd.DataFrame],
) -> None:
    """Render stock summary cards."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Symbol", stock_symbol)
    
    with col2:
        current_weight = stock_weights.iloc[-1] if len(stock_weights) > 0 else 0.0
        st.metric("Current Weight", f"{current_weight:.2%}")
    
    with col3:
        held_mask = stock_weights.abs() > 1e-12
        days_held = int(held_mask.sum())
        st.metric("Days Held", days_held)
    
    with col4:
        if factor_vectors is not None and stock_symbol in factor_vectors.index:
            pnl = factor_vectors.loc[stock_symbol].get("total_pnl_contrib", 0.0)
            st.metric("Total P&L Contribution", f"{pnl:.4f}")
        else:
            st.metric("Total P&L Contribution", "—")


def _render_performance_tab(
    stock_symbol: str,
    stock_returns: Optional[pd.Series],
    stock_weights: pd.Series,
    portfolio_returns: Optional[pd.Series],
    data: DashboardData,
) -> None:
    """Render performance analysis tab."""
    
    if stock_returns is None or stock_returns.empty:
        st.warning(
            "Stock returns data not available. "
            "For accurate performance analysis, stock returns need to be saved in artifacts."
        )
        return
    
    # Performance metrics
    perf = compute_stock_performance(
        stock_returns,
        benchmark_returns=portfolio_returns,
        weights=stock_weights,
    )
    
    if perf:
        # Format and display performance table
        perf_display = format_performance_table(perf, show_tc_comparison=False)
        st.dataframe(perf_display, width='stretch', hide_index=True)
    
    st.divider()
    
    # Performance charts
    if PLOTLY_AVAILABLE:
        # Cumulative returns
        cum_returns = (1 + stock_returns).cumprod() - 1
        
        fig = create_time_series_chart(
            cum_returns.to_frame("Stock"),
            title="Cumulative Returns",
        )
        
        if fig is not None:
            # Add portfolio comparison if available
            if portfolio_returns is not None:
                port_cum = (1 + portfolio_returns).cumprod() - 1
                aligned_stock, aligned_port = cum_returns.align(port_cum, join='inner')
                
                if not aligned_port.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=aligned_port.index,
                            y=aligned_port.values,
                            name="Portfolio",
                            line=dict(color=PM_COLORS["neutral"], width=2),
                        )
                    )
            
            st.plotly_chart(fig)
        
        # Daily returns
        st.markdown("#### Daily Returns")
        daily_fig = create_time_series_chart(
            stock_returns.to_frame("Daily Return"),
            title="Daily Returns",
        )
        if daily_fig is not None:
            st.plotly_chart(daily_fig)
        
        # Weight over time
        st.markdown("#### Position Size Over Time")
        weight_fig = create_time_series_chart(
            stock_weights.to_frame("Weight"),
            title="Portfolio Weight",
        )
        if weight_fig is not None:
            st.plotly_chart(weight_fig)


def _render_factors_tab(
    stock_symbol: str,
    factor_vectors: Optional[pd.DataFrame],
) -> None:
    """Render factor analysis tab."""
    
    if factor_vectors is None or factor_vectors.empty:
        st.warning("Factor exposure data not available.")
        return
    
    if stock_symbol not in factor_vectors.index:
        st.warning(f"Factor data not found for {stock_symbol}.")
        return
    
    # Get factor exposures
    exposures = compute_stock_factor_exposure(stock_symbol, factor_vectors)
    
    if not exposures:
        st.info("No factor exposures available.")
        return
    
    # Convert to DataFrame for display
    exp_df = pd.DataFrame([exposures]).T
    exp_df.columns = ["Exposure"]
    exp_df = exp_df.sort_values("Exposure", key=abs, ascending=False)
    
    st.markdown("### Factor Exposures")
    st.dataframe(exp_df, width='stretch')
    
    # Top factors visualization
    if PLOTLY_AVAILABLE:
        top_n = st.slider("Show Top N Factors", 5, 30, 15)
        top_factors = exp_df.head(top_n)
        
        fig = go.Figure()
        
        colors = [PM_COLORS["positive"] if x >= 0 else PM_COLORS["negative"] for x in top_factors["Exposure"]]
        
        fig.add_trace(
            go.Bar(
                x=top_factors.index,
                y=top_factors["Exposure"],
                marker_color=colors,
                text=[f"{x:.3f}" for x in top_factors["Exposure"]],
                textposition="outside",
            )
        )
        
        layout = get_plotly_layout(
            title=f"Top {top_n} Factor Exposures",
            height=400,
        )
        layout["xaxis"]["title"] = "Factor"
        layout["yaxis"]["title"] = "Exposure (z-score)"
        
        fig.update_layout(**layout)
        
        st.plotly_chart(fig)
    
    # Factor summary from factor_vectors
    st.markdown("### Factor Summary")
    row = factor_vectors.loc[stock_symbol]
    summary_cols = ["mean_score", "score_vol"]
    summary_data = {col: row.get(col, "—") for col in summary_cols if col in factor_vectors.columns}
    
    if summary_data:
        summary_df = pd.DataFrame([summary_data]).T
        summary_df.columns = ["Value"]
        st.dataframe(summary_df, width='stretch')


def _render_attribution_tab(
    stock_symbol: str,
    weights: pd.DataFrame,
    factor_vectors: Optional[pd.DataFrame],
    portfolio_returns: Optional[pd.Series],
    stock_returns: Optional[pd.Series],
) -> None:
    """Render attribution analysis tab."""
    
    attribution = compute_stock_attribution_enhanced(
        stock_symbol=stock_symbol,
        weights=weights,
        factor_vectors=factor_vectors if factor_vectors is not None else pd.DataFrame(),
        portfolio_returns=portfolio_returns,
        stock_returns=stock_returns,
    )
    
    if not attribution:
        st.warning("Could not compute attribution.")
        return
    
    # P&L metrics
    st.markdown("### P&L Attribution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total P&L", f"{attribution.get('total_pnl', 0.0):.4f}")
    
    with col2:
        st.metric("P&L per Day Held", f"{attribution.get('pnl_per_day', 0.0):.6f}")
    
    with col3:
        st.metric("Days Held", attribution.get('days_held', 0))
    
    # Factor exposures
    if "factor_exposures" in attribution and attribution["factor_exposures"]:
        st.markdown("### Factor Exposures")
        factor_exp = pd.DataFrame([attribution["factor_exposures"]]).T
        factor_exp.columns = ["Exposure"]
        factor_exp = factor_exp.sort_values("Exposure", key=abs, ascending=False).head(20)
        st.dataframe(factor_exp, width='stretch')
    
    # Position statistics
    if "weight_stats" in attribution:
        st.markdown("### Position Statistics")
        weight_stats = attribution["weight_stats"]
        stats_df = pd.DataFrame([weight_stats]).T
        stats_df.columns = ["Value"]
        
        # Create display DataFrame with string types to avoid dtype incompatibility
        display_stats = stats_df.copy()
        display_stats["Value"] = display_stats["Value"].astype(str)
        
        # Format as percentages where appropriate
        for col in ["current", "max", "min", "mean", "mean_when_held"]:
            if col in display_stats.index:
                try:
                    val = stats_df.loc[col, "Value"]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        display_stats.loc[col, "Value"] = f"{val:.2%}"
                except (KeyError, TypeError):
                    pass
        
        st.dataframe(display_stats, width='stretch')


def _render_trades_tab(
    stock_symbol: str,
    weights: pd.DataFrame,
    portfolio_returns: Optional[pd.Series],
) -> None:
    """Render trade-by-trade analysis tab."""
    
    st.markdown("### Trade History")
    st.caption("Each trade represents an entry to exit cycle")
    
    trades = analyze_trades(stock_symbol, weights, portfolio_returns)
    
    if not trades:
        st.info("No completed trades found for this stock.")
        return
    
    # Trade summary metrics
    completed_trades = [t for t in trades if not t.get("is_open", False)]
    open_trades = [t for t in trades if t.get("is_open", False)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades))
    
    with col2:
        st.metric("Completed", len(completed_trades))
    
    with col3:
        if completed_trades:
            winners = sum(1 for t in completed_trades if t.get("pnl", 0) > 0)
            hit_rate = winners / len(completed_trades)
            st.metric("Hit Rate", f"{hit_rate:.1%}")
        else:
            st.metric("Hit Rate", "—")
    
    with col4:
        avg_holding = np.mean([t["holding_days"] for t in completed_trades]) if completed_trades else 0
        st.metric("Avg Holding (days)", f"{avg_holding:.0f}")
    
    st.divider()
    
    # Trade details table
    trades_df = pd.DataFrame(trades)
    
    if not trades_df.empty:
        # Create display DataFrame with proper types to avoid Arrow serialization issues
        trades_display = trades_df.copy()
        
        # Format date columns as strings
        if "entry_date" in trades_display.columns:
            trades_display["entry_date"] = pd.to_datetime(trades_display["entry_date"]).dt.strftime("%Y-%m-%d")
        if "exit_date" in trades_display.columns:
            trades_display["exit_date"] = trades_display["exit_date"].apply(
                lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "OPEN"
            )
        
        # Format numeric columns as strings for display (create new columns to avoid dtype issues)
        if "entry_weight" in trades_display.columns:
            trades_display["entry_weight"] = trades_display["entry_weight"].astype(float).apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
        if "max_weight" in trades_display.columns:
            trades_display["max_weight"] = trades_display["max_weight"].astype(float).apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
        if "pnl" in trades_display.columns:
            trades_display["pnl"] = trades_display["pnl"].astype(float).apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
        
        # Rename columns for display
        display_cols = {
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "holding_days": "Days",
            "side": "Side",
            "entry_weight": "Entry Wt",
            "max_weight": "Max Wt",
            "pnl": "P&L",
        }
        
        # Select and rename columns
        available_cols = [c for c in display_cols.keys() if c in trades_display.columns]
        trades_display = trades_display[available_cols].rename(columns=display_cols)
        
        st.dataframe(trades_display, width='stretch', hide_index=True)
    
    # Open position callout
    if open_trades:
        st.warning(f"⚠️ {len(open_trades)} open position(s) not yet closed")


def _render_diagnostics_tab(
    stock_symbol: str,
    weights: pd.DataFrame,
    portfolio_returns: Optional[pd.Series],
    stock_returns: Optional[pd.Series],
) -> None:
    """Render diagnostics tab."""
    
    diagnostics = get_stock_diagnostics(
        stock_symbol=stock_symbol,
        weights=weights,
        portfolio_returns=portfolio_returns,
        stock_returns=stock_returns,
    )
    
    if not diagnostics:
        st.warning("Could not compute diagnostics.")
        return
    
    # Key metrics
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Days Held", diagnostics.get("days_held", 0))
    
    with col2:
        st.metric("Avg Turnover", f"{diagnostics.get('avg_turnover', 0.0):.4f}")
    
    with col3:
        st.metric("Total Turnover", f"{diagnostics.get('total_turnover', 0.0):.4f}")
    
    with col4:
        corr = diagnostics.get("correlation_with_portfolio")
        if corr is not None:
            st.metric("Correlation w/ Portfolio", f"{corr:.3f}")
        else:
            st.metric("Correlation w/ Portfolio", "—")
    
    # Trade statistics
    st.markdown("### Trade Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("# Entries", diagnostics.get("n_entries", 0))
    with col2:
        st.metric("# Exits", diagnostics.get("n_exits", 0))
    with col3:
        st.metric("# Position Changes", diagnostics.get("n_position_changes", 0))
    
    st.divider()
    
    # Rolling metrics
    if stock_returns is not None and not stock_returns.empty:
        st.markdown("### Rolling Performance (21-day)")
        
        rolling = compute_stock_rolling_metrics(stock_returns, window=21)
        
        if not rolling.empty and PLOTLY_AVAILABLE:
            # Rolling Sharpe chart
            fig = create_time_series_chart(
                rolling[["rolling_sharpe"]].rename(columns={"rolling_sharpe": "Rolling Sharpe"}),
                title="Rolling Sharpe Ratio",
            )
            if fig is not None:
                st.plotly_chart(fig)
    
    st.divider()
    
    # Position changes timeline
    if "position_changes" in diagnostics and diagnostics["position_changes"]:
        st.markdown("### Position Changes Timeline")
        changes_df = pd.DataFrame(diagnostics["position_changes"])
        if not changes_df.empty:
            # Format dates
            changes_df["date"] = pd.to_datetime(changes_df["date"]).dt.strftime("%Y-%m-%d")
            changes_df = changes_df.sort_values("date", ascending=False).head(30)
            
            # Format percentages
            for col in ["weight_before", "weight_after", "weight_change"]:
                if col in changes_df.columns:
                    changes_df[col] = changes_df[col].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(changes_df, width='stretch', hide_index=True)


def _render_stock_stack(
    data: DashboardData,
    available_stocks: List[str],
    current_holdings: List[str],
) -> None:
    """Render comprehensive Stock Stack table with all stocks, ratings, and grades."""
    
    st.markdown("### 📊 Stock Stack")
    strategy_name = data.base_name if hasattr(data, 'base_name') else "Current Strategy"
    st.caption(f"Comprehensive evaluation of all stocks for **{strategy_name}** with ratings, grades, and quantitative metrics")
    st.info("💡 **Strategy-Specific Analysis**: Ratings and grades are computed from this strategy's own factor signals, position sizing, and portfolio performance. "
            "Each strategy may rate the same stock differently based on its unique factor exposures and signal quality.")
    
    # Get portfolio returns
    portfolio_returns = select_return_series(data.diag)
    
    # Filter controls
    st.markdown("#### Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_only = st.checkbox("Current holdings only", value=False)
        side_filter = st.selectbox(
            "Side",
            ["All", "Long", "Short"],
            help="Filter by position direction"
        )
        side = None if side_filter == "All" else side_filter.upper()
    
    with col2:
        rating_filter = st.selectbox(
            "Rating",
            ["All", "BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"],
            help="Filter by stock rating"
        )
        grade_filter = st.selectbox(
            "Min Overall Grade",
            ["All", "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"],
            help="Filter by minimum overall grade"
        )
    
    with col3:
        min_sharpe = st.number_input("Min Sharpe", value=-10.0, step=0.1, format="%.2f")
        min_pnl = st.number_input("Min P&L", value=-1.0, step=0.01, format="%.4f")
    
    with col4:
        min_days = st.number_input("Min Days Held", min_value=0, value=0)
        min_weight = st.number_input("Min Weight (%)", value=0.0, step=0.1, format="%.2f") / 100.0
    
    # Apply initial filters
    filtered_stocks = available_stocks.copy()
    
    if current_only:
        filtered_stocks = [s for s in filtered_stocks if s in current_holdings]
    
    if side is not None:
        filtered_stocks = [
            s for s in filtered_stocks
            if s in data.weights.columns
            and (
                (side == "LONG" and data.weights[s].iloc[-1] > 1e-12) or
                (side == "SHORT" and data.weights[s].iloc[-1] < -1e-12)
            )
        ]
    
    st.markdown(f"**{len(filtered_stocks)} stocks to evaluate**")
    
    if not filtered_stocks:
        st.info("No stocks match the selected filters.")
        return
    
    # Progress indicator
    with st.spinner(f"Computing ratings and grades for {len(filtered_stocks)} stocks..."):
        # Pre-compute stock returns for all stocks
        stock_returns_dict = {}
        signal_quality_dict = {}
        elite_metrics_dict = {}
        
        # Compute returns and signal quality in batch
        for symbol in filtered_stocks:
            if symbol in data.weights.columns:
                stock_weights = data.weights[symbol].fillna(0.0)
                
                # Compute stock returns
                stock_returns = compute_stock_returns_from_weights(
                    data.weights,
                    portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),
                    symbol,
                )
                if stock_returns is not None and len(stock_returns) >= 20:
                    stock_returns_dict[symbol] = stock_returns
                    
                    # Compute signal quality
                    signal_quality = compute_stock_signal_quality(
                        stock_weights,
                        stock_returns,
                        max_horizon=21,
                    )
                    if signal_quality:
                        signal_quality_dict[symbol] = signal_quality
                    
                    # Compute elite metrics
                    if portfolio_returns is not None:
                        elite_metrics = compute_elite_scorecard(
                            stock_returns,
                            portfolio_returns,
                            stock_weights,
                        )
                        elite_metrics_dict[symbol] = elite_metrics
        
        # Batch compute ratings and grades
        ratings = compute_batch_stock_ratings(
            symbols=filtered_stocks,
            factor_vectors=data.factor_vectors,
            weights=data.weights,
            portfolio_returns=portfolio_returns,
            stock_returns_dict=stock_returns_dict,
            signal_quality_dict=signal_quality_dict,
        )
        
        grades = compute_batch_stock_grades(
            symbols=filtered_stocks,
            factor_vectors=data.factor_vectors,
            weights=data.weights,
            portfolio_returns=portfolio_returns,
            stock_returns_dict=stock_returns_dict,
            signal_quality_dict=signal_quality_dict,
            elite_metrics_dict=elite_metrics_dict,
        )
    
    # Build comprehensive table
    rows = []
    for symbol in filtered_stocks:
        row = {"Symbol": symbol}
        
        # Rating and grades
        rating = ratings.get(symbol)
        grade = grades.get(symbol)
        
        if rating:
            row["Rating"] = rating.rating
            row["Rating Score"] = rating.score
            row["Confidence"] = rating.confidence
        else:
            row["Rating"] = "N/A"
            row["Rating Score"] = 0.0
            row["Confidence"] = 0.0
        
        if grade:
            row["Overall Grade"] = grade.overall
            row["Momentum Grade"] = grade.momentum
            row["Value/Quality Grade"] = grade.value_quality
            row["Risk Grade"] = grade.risk_management
            row["Signal Grade"] = grade.signal_quality
            row["Performance Grade"] = grade.performance
        else:
            row["Overall Grade"] = "N/A"
            row["Momentum Grade"] = "N/A"
            row["Value/Quality Grade"] = "N/A"
            row["Risk Grade"] = "N/A"
            row["Signal Grade"] = "N/A"
            row["Performance Grade"] = "N/A"
        
        # Position metrics
        if symbol in data.weights.columns:
            current_weight = float(data.weights[symbol].iloc[-1])
            row["Current Weight"] = current_weight
            stock_weights = data.weights[symbol].fillna(0.0)
            held_mask = stock_weights.abs() > 1e-12
            row["Days Held"] = int(held_mask.sum())
            row["Avg Weight"] = float(stock_weights.mean())
        else:
            row["Current Weight"] = 0.0
            row["Days Held"] = 0
            row["Avg Weight"] = 0.0
        
        # Factor vectors metrics
        if data.factor_vectors is not None and symbol in data.factor_vectors.index:
            fv_row = data.factor_vectors.loc[symbol]
            row["Total P&L"] = float(fv_row.get("total_pnl_contrib", 0.0))
            row["P&L per Day"] = float(fv_row.get("pnl_per_day_held", 0.0))
            row["Mean Score"] = float(fv_row.get("mean_score", 0.0))
            row["Score Vol"] = float(fv_row.get("score_vol", 0.0))
        else:
            row["Total P&L"] = 0.0
            row["P&L per Day"] = 0.0
            row["Mean Score"] = 0.0
            row["Score Vol"] = 0.0
        
        # Performance metrics
        elite_metrics = elite_metrics_dict.get(symbol)
        if elite_metrics:
            row["Sharpe"] = elite_metrics.sharpe
            row["Sortino"] = elite_metrics.sortino
            row["Max Drawdown"] = elite_metrics.max_drawdown
            row["VaR 95%"] = elite_metrics.var_95
        else:
            row["Sharpe"] = 0.0
            row["Sortino"] = 0.0
            row["Max Drawdown"] = 0.0
            row["VaR 95%"] = 0.0
        
        # Compute annual return and volatility from returns
        stock_returns = stock_returns_dict.get(symbol)
        if stock_returns is not None and len(stock_returns) >= 20:
            rets = stock_returns.dropna()
            row["Ann Return"] = float(rets.mean() * 252)
            row["Ann Volatility"] = float(rets.std() * np.sqrt(252))
        else:
            row["Ann Return"] = 0.0
            row["Ann Volatility"] = 0.0
        
        # Signal quality metrics
        signal_quality = signal_quality_dict.get(symbol)
        if signal_quality:
            ic_by_horizon = signal_quality.get("ic_by_horizon", {})
            if ic_by_horizon:
                avg_ic = np.mean([
                    (v.get("pearson_ic", 0.0) + v.get("rank_ic", 0.0)) / 2.0
                    for v in ic_by_horizon.values()
                ])
                row["Avg IC"] = avg_ic
            else:
                row["Avg IC"] = None
            
            row["Hit Rate"] = signal_quality.get("hit_rate")
        else:
            row["Avg IC"] = None
            row["Hit Rate"] = None
        
        rows.append(row)
    
    # Convert to DataFrame
    stack_df = pd.DataFrame(rows)
    
    # Apply additional filters
    if rating_filter != "All":
        stack_df = stack_df[stack_df["Rating"] == rating_filter]
    
    if grade_filter != "All":
        grade_order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]
        min_grade_idx = grade_order.index(grade_filter) if grade_filter in grade_order else 0
        valid_grades = grade_order[:min_grade_idx + 1]
        stack_df = stack_df[stack_df["Overall Grade"].isin(valid_grades + ["N/A"])]
    
    stack_df = stack_df[stack_df["Sharpe"] >= min_sharpe]
    stack_df = stack_df[stack_df["Total P&L"] >= min_pnl]
    stack_df = stack_df[stack_df["Days Held"] >= min_days]
    stack_df = stack_df[stack_df["Current Weight"].abs() >= min_weight]
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", len(stack_df))
    
    with col2:
        if len(stack_df) > 0:
            buy_count = len(stack_df[stack_df["Rating"] == "BUY"])
            st.metric("BUY Rated", buy_count)
        else:
            st.metric("BUY Rated", 0)
    
    with col3:
        if len(stack_df) > 0:
            avg_sharpe = stack_df["Sharpe"].mean()
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
        else:
            st.metric("Avg Sharpe", "—")
    
    with col4:
        if len(stack_df) > 0:
            avg_pnl = stack_df["Total P&L"].mean()
            st.metric("Avg P&L", f"{avg_pnl:.4f}")
        else:
            st.metric("Avg P&L", "—")
    
    with col5:
        if len(stack_df) > 0:
            a_plus_count = len(stack_df[stack_df["Overall Grade"] == "A+"])
            st.metric("A+ Grades", a_plus_count)
        else:
            st.metric("A+ Grades", 0)
    
    st.divider()
    
    # Display table
    st.markdown("#### Stock Stack Table")
    st.caption("Click column headers to sort. Scroll horizontally to see all columns.")
    
    if stack_df.empty:
        st.info("No stocks match all selected filters.")
        return
    
    # Format display DataFrame
    display_df = stack_df.copy()
    
    # Format percentages
    for col in ["Current Weight", "Avg Weight", "Max Drawdown", "VaR 95%", "Ann Return", "Ann Volatility"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    
    # Format decimals
    for col in ["Rating Score", "Confidence", "Total P&L", "P&L per Day", "Mean Score", "Score Vol", 
                "Sharpe", "Sortino", "Avg IC"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and x != 0.0 else ("—" if pd.isna(x) else "0.000"))
    
    # Format hit rate
    if "Hit Rate" in display_df.columns:
        display_df["Hit Rate"] = display_df["Hit Rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    
    # Reorder columns for better display
    priority_cols = [
        "Symbol", "Rating", "Overall Grade", "Momentum Grade", "Value/Quality Grade",
        "Risk Grade", "Signal Grade", "Performance Grade", "Current Weight", "Days Held",
        "Total P&L", "P&L per Day", "Sharpe", "Sortino", "Ann Return", "Ann Volatility",
        "Max Drawdown", "Mean Score", "Avg IC", "Hit Rate", "Rating Score", "Confidence",
        "Avg Weight", "VaR 95%", "Score Vol"
    ]
    
    # Get columns that exist
    ordered_cols = [c for c in priority_cols if c in display_df.columns]
    remaining_cols = [c for c in display_df.columns if c not in ordered_cols]
    display_df = display_df[ordered_cols + remaining_cols]
    
    # Color-code rating column using st.dataframe styling
    def color_rating(val):
        if val == "BUY":
            return "background-color: #2ecc71; color: white"
        elif val == "OVERWEIGHT":
            return "background-color: #f39c12; color: white"
        elif val == "HOLD":
            return "background-color: #95a5a6; color: white"
        elif val == "UNDERWEIGHT":
            return "background-color: #e67e22; color: white"
        elif val == "SELL":
            return "background-color: #e74c3c; color: white"
        else:
            return ""
    
    # Apply styling if Rating column exists
    if "Rating" in display_df.columns:
        try:
            styled_df = display_df.style.applymap(color_rating, subset=["Rating"])
            st.dataframe(styled_df, width='stretch', height=600, hide_index=True)
        except Exception:
            # Fallback if styling fails
            st.dataframe(display_df, width='stretch', height=600, hide_index=True)
    else:
        st.dataframe(display_df, width='stretch', height=600, hide_index=True)
    
    # Download button
    st.divider()
    col_download, col_info = st.columns([1, 3])
    
    with col_download:
        # Use raw DataFrame (not formatted) for CSV
        csv_df = stack_df.copy()
        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name=f"stock_stack_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    with col_info:
        st.caption(f"Exported {len(stack_df)} stocks with all quantitative metrics. Raw numeric values included for analysis.")


def _render_buy_list(
    data: DashboardData,
    available_stocks: List[str],
    current_holdings: List[str],
) -> None:
    """Render BUY-rated stocks ranked by rating score."""
    
    st.markdown("### 🟢 BUY List")
    strategy_name = data.base_name if hasattr(data, 'base_name') else "Current Strategy"
    st.caption(f"BUY-rated stocks for **{strategy_name}** ranked by rating score (highest first)")
    
    # Get portfolio returns
    portfolio_returns = select_return_series(data.diag)
    
    # Quick filter
    col1, col2 = st.columns(2)
    with col1:
        current_only = st.checkbox("Current holdings only", value=False)
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05, help="Minimum confidence threshold for rating")
    
    # Filter stocks
    filtered_stocks = available_stocks.copy()
    if current_only:
        filtered_stocks = [s for s in filtered_stocks if s in current_holdings]
    
    if not filtered_stocks:
        st.info("No stocks match the selected filters.")
        return
    
    # Compute ratings for all stocks
    with st.spinner(f"Computing ratings for {len(filtered_stocks)} stocks..."):
        # Pre-compute stock returns
        stock_returns_dict = {}
        signal_quality_dict = {}
        
        for symbol in filtered_stocks:
            if symbol in data.weights.columns:
                stock_weights = data.weights[symbol].fillna(0.0)
                stock_returns = compute_stock_returns_from_weights(
                    data.weights,
                    portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),
                    symbol,
                )
                if stock_returns is not None and len(stock_returns) >= 20:
                    stock_returns_dict[symbol] = stock_returns
                    signal_quality = compute_stock_signal_quality(
                        stock_weights,
                        stock_returns,
                        max_horizon=21,
                    )
                    if signal_quality:
                        signal_quality_dict[symbol] = signal_quality
        
        # Batch compute ratings
        ratings = compute_batch_stock_ratings(
            symbols=filtered_stocks,
            factor_vectors=data.factor_vectors,
            weights=data.weights,
            portfolio_returns=portfolio_returns,
            stock_returns_dict=stock_returns_dict,
            signal_quality_dict=signal_quality_dict,
        )
    
    # Filter to BUY-rated stocks only
    buy_stocks = [
        (symbol, rating)
        for symbol, rating in ratings.items()
        if rating.rating == "BUY" and rating.confidence >= min_confidence
    ]
    
    # Sort by rating score (highest first)
    buy_stocks.sort(key=lambda x: x[1].score, reverse=True)
    
    st.markdown(f"#### Found {len(buy_stocks)} BUY-rated stocks")
    
    if not buy_stocks:
        st.info("No BUY-rated stocks found matching the criteria. Try adjusting filters or check other analysis modes.")
        return
    
    # Build simple table
    rows = []
    for symbol, rating in buy_stocks:
        row = {
            "Symbol": symbol,
            "Rating Score": rating.score,
            "Confidence": rating.confidence,
        }
        
        # Add key metrics
        if symbol in data.weights.columns:
            current_weight = float(data.weights[symbol].iloc[-1])
            row["Current Weight"] = current_weight
            stock_weights = data.weights[symbol].fillna(0.0)
            held_mask = stock_weights.abs() > 1e-12
            row["Days Held"] = int(held_mask.sum())
        else:
            row["Current Weight"] = 0.0
            row["Days Held"] = 0
        
        # Factor vectors metrics
        if data.factor_vectors is not None and symbol in data.factor_vectors.index:
            fv_row = data.factor_vectors.loc[symbol]
            row["Total P&L"] = float(fv_row.get("total_pnl_contrib", 0.0))
            row["P&L per Day"] = float(fv_row.get("pnl_per_day_held", 0.0))
            row["Mean Score"] = float(fv_row.get("mean_score", 0.0))
        else:
            row["Total P&L"] = 0.0
            row["P&L per Day"] = 0.0
            row["Mean Score"] = 0.0
        
        # Performance metrics from returns
        stock_returns = stock_returns_dict.get(symbol)
        if stock_returns is not None and len(stock_returns) >= 20:
            rets = stock_returns.dropna()
            ann_ret = float(rets.mean() * 252)
            ann_vol = float(rets.std() * np.sqrt(252))
            # Use np.divide to avoid division warnings
            sharpe = np.divide(ann_ret, ann_vol + 1e-12, out=np.array([0.0]), where=(ann_vol + 1e-12) != 0)[0]
            row["Sharpe"] = sharpe
            row["Ann Return"] = ann_ret
        else:
            row["Sharpe"] = 0.0
            row["Ann Return"] = 0.0
        
        # Rationale (truncated)
        rationale = rating.rationale.split(" | ")[0] if rating.rationale else "—"
        row["Key Signal"] = rationale
        
        rows.append(row)
    
    # Convert to DataFrame
    buy_df = pd.DataFrame(rows)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BUY Stocks", len(buy_df))
    with col2:
        if len(buy_df) > 0:
            avg_score = buy_df["Rating Score"].mean()
            st.metric("Avg Rating Score", f"{avg_score:.3f}")
        else:
            st.metric("Avg Rating Score", "—")
    with col3:
        if len(buy_df) > 0:
            avg_sharpe = buy_df["Sharpe"].mean()
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
        else:
            st.metric("Avg Sharpe", "—")
    with col4:
        if len(buy_df) > 0:
            current_buys = len(buy_df[buy_df["Current Weight"].abs() > 1e-12])
            st.metric("Currently Held", current_buys)
        else:
            st.metric("Currently Held", 0)
    
    st.divider()
    
    # Display table
    st.markdown("#### BUY-Rated Stocks (Ranked by Rating Score)")
    
    # Format display
    display_df = buy_df.copy()
    display_df["Rating Score"] = display_df["Rating Score"].apply(lambda x: f"{x:.3f}")
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.0%}")
    display_df["Current Weight"] = display_df["Current Weight"].apply(lambda x: f"{x:.2%}")
    display_df["Total P&L"] = display_df["Total P&L"].apply(lambda x: f"{x:.4f}")
    display_df["P&L per Day"] = display_df["P&L per Day"].apply(lambda x: f"{x:.4f}")
    display_df["Mean Score"] = display_df["Mean Score"].apply(lambda x: f"{x:.3f}")
    display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: f"{x:.2f}")
    display_df["Ann Return"] = display_df["Ann Return"].apply(lambda x: f"{x:.2%}")
    
    # Reorder columns
    priority_cols = [
        "Symbol", "Rating Score", "Confidence", "Current Weight", "Days Held",
        "Sharpe", "Ann Return", "Total P&L", "P&L per Day", "Mean Score", "Key Signal"
    ]
    ordered_cols = [c for c in priority_cols if c in display_df.columns]
    remaining_cols = [c for c in display_df.columns if c not in ordered_cols]
    display_df = display_df[ordered_cols + remaining_cols]
    
    # Color-code rating score column
    def highlight_buy(val):
        try:
            score = float(val)
            if score >= 0.8:
                return "background-color: #2ecc71; color: white; font-weight: bold"
            elif score >= 0.6:
                return "background-color: #27ae60; color: white"
            else:
                return "background-color: #1e8449; color: white"
        except:
            return ""
    
    try:
        styled_df = display_df.style.applymap(highlight_buy, subset=["Rating Score"])
        st.dataframe(styled_df, width='stretch', height=500, hide_index=True)
    except Exception:
        st.dataframe(display_df, width='stretch', height=500, hide_index=True)
    
    # Download button
    st.divider()
    col_download, col_info = st.columns([1, 3])
    
    with col_download:
        csv_bytes = buy_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name=f"buy_list_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    with col_info:
        st.caption(f"Showing {len(buy_df)} BUY-rated stocks ranked by rating score. Higher scores indicate stronger buy signals.")


def _render_ratings_grades_panel(
    stock_symbol: str,
    stock_returns: Optional[pd.Series],
    stock_weights: pd.Series,
    portfolio_returns: Optional[pd.Series],
    factor_vectors: Optional[pd.DataFrame],
) -> None:
    """Render compact ratings and grades panel."""
    
    # Compute signal quality if we have the data
    signal_quality = None
    if stock_returns is not None and not stock_returns.empty and stock_weights is not None:
        signal_quality = compute_stock_signal_quality(
            stock_weights,
            stock_returns,
            max_horizon=21,
        )
    
    # Compute rating
    rating = compute_stock_rating(
        symbol=stock_symbol,
        factor_vectors=factor_vectors,
        stock_returns=stock_returns,
        portfolio_returns=portfolio_returns,
        weights=stock_weights,
        signal_quality=signal_quality,
    )
    
    # Compute grades
    elite_metrics = None
    if stock_returns is not None and portfolio_returns is not None:
        elite_metrics = compute_elite_scorecard(
            stock_returns,
            portfolio_returns,
            stock_weights,
        )
    
    grades = compute_stock_grades(
        symbol=stock_symbol,
        factor_vectors=factor_vectors,
        stock_returns=stock_returns,
        portfolio_returns=portfolio_returns,
        weights=stock_weights,
        signal_quality=signal_quality,
        elite_metrics=elite_metrics,
    )
    
    # Display rating
    st.markdown("### ⭐ Stock Rating & Grades")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Rating badge
        rating_colors = {
            "BUY": "🟢",
            "OVERWEIGHT": "🟡",
            "HOLD": "⚪",
            "UNDERWEIGHT": "🟠",
            "SELL": "🔴",
        }
        rating_icon = rating_colors.get(rating.rating, "⚪")
        st.markdown(f"#### {rating_icon} {rating.rating}")
        st.caption(f"Score: {rating.score:.2f}")
        st.caption(f"Confidence: {rating.confidence:.0%}")
    
    with col2:
        st.markdown("**Momentum**")
        st.markdown(f"### {grades.momentum}")
    
    with col3:
        st.markdown("**Value/Quality**")
        st.markdown(f"### {grades.value_quality}")
    
    with col4:
        st.markdown("**Risk Mgmt**")
        st.markdown(f"### {grades.risk_management}")
    
    with col5:
        st.markdown("**Overall**")
        st.markdown(f"### {grades.overall}")
    
    # Rationale
    if rating.rationale:
        st.info(f"**Rationale**: {rating.rationale}")


def _render_ratings_grades_tab(
    stock_symbol: str,
    stock_returns: Optional[pd.Series],
    stock_weights: pd.Series,
    portfolio_returns: Optional[pd.Series],
    factor_vectors: Optional[pd.DataFrame],
) -> None:
    """Render detailed ratings and grades tab."""
    
    st.markdown("## ⭐ Quantitative Stock Ratings & Grades")
    st.markdown("""
    This analysis provides institutional-grade ratings and letter grades based on 
    quantitative statistical arbitrage signals:
    - **Rating**: Overall recommendation (SELL, UNDERWEIGHT, HOLD, OVERWEIGHT, BUY)
    - **Grades**: Letter grades (A+ to F) for different aspects of quant stat arb
    """)
    
    # Compute signal quality
    signal_quality = None
    if stock_returns is not None and not stock_returns.empty and stock_weights is not None:
        with st.spinner("Computing signal quality metrics..."):
            signal_quality = compute_stock_signal_quality(
                stock_weights,
                stock_returns,
                max_horizon=21,
            )
    
    # Compute rating
    with st.spinner("Computing stock rating..."):
        rating = compute_stock_rating(
            symbol=stock_symbol,
            factor_vectors=factor_vectors,
            stock_returns=stock_returns,
            portfolio_returns=portfolio_returns,
            weights=stock_weights,
            signal_quality=signal_quality,
        )
    
    # Compute elite metrics
    elite_metrics = None
    if stock_returns is not None and portfolio_returns is not None:
        with st.spinner("Computing elite metrics..."):
            elite_metrics = compute_elite_scorecard(
                stock_returns,
                portfolio_returns,
                stock_weights,
            )
    
    # Compute grades
    with st.spinner("Computing grades..."):
        grades = compute_stock_grades(
            symbol=stock_symbol,
            factor_vectors=factor_vectors,
            stock_returns=stock_returns,
            portfolio_returns=portfolio_returns,
            weights=stock_weights,
            signal_quality=signal_quality,
            elite_metrics=elite_metrics,
        )
    
    st.divider()
    
    # Overall Rating Section
    st.markdown("### Overall Rating")
    
    rating_colors = {
        "BUY": "🟢",
        "OVERWEIGHT": "🟡",
        "HOLD": "⚪",
        "UNDERWEIGHT": "🟠",
        "SELL": "🔴",
    }
    rating_icon = rating_colors.get(rating.rating, "⚪")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"#### {rating_icon} **{rating.rating}**")
        
        # Score gauge
        score_pct = (rating.score + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        st.progress(score_pct, text=f"Score: {rating.score:.3f}")
        
        st.metric("Confidence", f"{rating.confidence:.0%}")
    
    with col2:
        st.markdown("**Rating Components:**")
        st.caption(rating.rationale)
        
        st.markdown("**Rating Scale:**")
        st.caption("""
        - **BUY** (Score ≥ 0.6): Strong positive signals across multiple dimensions
        - **OVERWEIGHT** (Score 0.2-0.6): Positive signals, good risk-adjusted returns
        - **HOLD** (Score -0.2 to 0.2): Neutral signals or mixed signals
        - **UNDERWEIGHT** (Score -0.6 to -0.2): Weak signals or negative performance
        - **SELL** (Score ≤ -0.6): Strong negative signals or poor risk management
        """)
    
    st.divider()
    
    # Detailed Grades Section
    st.markdown("### Detailed Grades")
    st.markdown("Letter grades (A+ to F) for different aspects of quantitative statistical arbitrage:")
    
    # Grade display with color coding
    grade_colors = {
        "A+": "🟢", "A": "🟢", "A-": "🟢",
        "B+": "🟡", "B": "🟡", "B-": "🟡",
        "C+": "🟠", "C": "🟠", "C-": "🟠",
        "D+": "🔴", "D": "🔴", "F": "🔴",
    }
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    grade_info = [
        ("Momentum", grades.momentum, "Momentum factor exposures and signal strength"),
        ("Value/Quality", grades.value_quality, "Value and quality factor exposures"),
        ("Risk Management", grades.risk_management, "Volatility, drawdown, and downside protection"),
        ("Signal Quality", grades.signal_quality, "IC, hit rate, and signal consistency"),
        ("Performance", grades.performance, "Sharpe ratio and statistical significance"),
        ("Overall", grades.overall, "Weighted average of all grades"),
    ]
    
    for i, (name, grade, desc) in enumerate(grade_info):
        with [col1, col2, col3, col4, col5, col6][i]:
            icon = grade_colors.get(grade, "⚪")
            st.markdown(f"**{name}**")
            st.markdown(f"### {icon} {grade}")
            with st.expander("Details"):
                st.caption(desc)
    
    st.divider()
    
    # Grade Breakdown
    st.markdown("### Grade Breakdown")
    
    grade_explanations = {
        "Momentum": {
            "grade": grades.momentum,
            "description": "Based on momentum factor exposures (residual momentum, breakouts, MTF alignment, etc.)",
            "factors": ["resid_mom", "breakout", "slope", "mtf_ic_momentum", "momentum_acceleration"],
        },
        "Value/Quality": {
            "grade": grades.value_quality,
            "description": "Based on value and quality factor exposures (value_score, quality_score, defensive factors)",
            "factors": ["value_score", "quality_score", "inv_vol", "low_beta", "beta_stability"],
        },
        "Risk Management": {
            "grade": grades.risk_management,
            "description": "Based on volatility, max drawdown, and downside protection (Sortino ratio)",
            "metrics": ["Max Drawdown", "Annual Volatility", "Sortino Ratio"],
        },
        "Signal Quality": {
            "grade": grades.signal_quality,
            "description": "Based on Information Coefficient (IC) and hit rate",
            "metrics": ["IC by Horizon", "Hit Rate", "Rolling IC"],
        },
        "Performance": {
            "grade": grades.performance,
            "description": "Based on Sharpe ratio and statistical significance",
            "metrics": ["Sharpe Ratio", "P-value", "Annual Return"],
        },
    }
    
    for aspect, info in grade_explanations.items():
        with st.expander(f"{aspect}: {info['grade']}"):
            st.caption(info['description'])
            if "factors" in info:
                st.caption(f"**Key Factors**: {', '.join(info['factors'])}")
            if "metrics" in info:
                st.caption(f"**Key Metrics**: {', '.join(info['metrics'])}")
    
    st.divider()
    
    # Signal Quality Details
    if signal_quality:
        st.markdown("### Signal Quality Details")
        
        ic_by_horizon = signal_quality.get("ic_by_horizon", {})
        if ic_by_horizon:
            st.markdown("**IC by Horizon:**")
            ic_data = []
            for horizon, ic_data_dict in ic_by_horizon.items():
                ic_data.append({
                    "Horizon": f"T+{horizon}",
                    "Pearson IC": f"{ic_data_dict.get('pearson_ic', 0.0):.4f}",
                    "Rank IC": f"{ic_data_dict.get('rank_ic', 0.0):.4f}",
                })
            if ic_data:
                st.dataframe(pd.DataFrame(ic_data), width='stretch', hide_index=True)
        
        hit_rate = signal_quality.get("hit_rate")
        if hit_rate is not None:
            st.metric("Hit Rate", f"{hit_rate:.1%}")
        
        optimal_horizon = signal_quality.get("optimal_horizon")
        if optimal_horizon:
            st.metric("Optimal Horizon", f"T+{optimal_horizon} days")
    
    # Elite Metrics Summary
    if elite_metrics:
        st.divider()
        st.markdown("### Elite Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{elite_metrics.sharpe:.2f}")
            st.metric("Sortino Ratio", f"{elite_metrics.sortino:.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{elite_metrics.max_drawdown:.1%}")
            st.metric("VaR (95%)", f"{elite_metrics.var_95:.2%}")
        
        with col3:
            st.metric("Skewness", f"{elite_metrics.skewness:.2f}")
            st.metric("Kurtosis", f"{elite_metrics.kurtosis:.2f}")
        
        with col4:
            st.metric("T-Stat", f"{elite_metrics.t_stat:.2f}")
            st.metric("P-Value", f"{elite_metrics.p_value:.4f}")
