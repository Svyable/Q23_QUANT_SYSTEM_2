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
        data: Dashboard data bundle
    """
    st.subheader("📈 Stock Analysis")
    
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
        ["Single Stock", "Compare Stocks", "Stock Screener"],
        horizontal=True,
        help="Analyze one stock, compare multiple stocks, or screen/filter stocks"
    )
    
    st.divider()
    
    if mode == "Single Stock":
        _render_single_stock_analysis(data, available_stocks, current_holdings)
    elif mode == "Compare Stocks":
        _render_multi_stock_comparison(data, available_stocks)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance",
        "🔍 Factors",
        "📈 Attribution",
        "📋 Trades",
        "🔧 Diagnostics",
    ])
    
    with tab1:
        _render_performance_tab(
            selected_stock,
            stock_returns,
            stock_weights,
            portfolio_returns,
            data,
        )
    
    with tab2:
        _render_factors_tab(selected_stock, data.factor_vectors)
    
    with tab3:
        _render_attribution_tab(
            selected_stock,
            data.weights,
            data.factor_vectors,
            portfolio_returns,
            stock_returns,
        )
    
    with tab4:
        _render_trades_tab(
            selected_stock,
            data.weights,
            portfolio_returns,
        )
    
    with tab5:
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
        
        st.dataframe(screen_df, use_container_width=True, hide_index=True)
        
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
        
        st.dataframe(display_df, use_container_width=True)
        
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
                        stock_returns_df.loc[mask, stock] = (
                            port_ret[mask] / (w_lag.loc[mask, stock] + eps)
                        )
                
                stock_returns_df = stock_returns_df.clip(-0.5, 0.5).replace([np.inf, -np.inf], 0.0)
                
                # Cumulative returns
                cum_returns_df = (1 + stock_returns_df).cumprod() - 1
                
                fig = create_time_series_chart(
                    cum_returns_df,
                    title="Cumulative Returns Comparison",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
    
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
                use_container_width=True,
            )


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
        st.dataframe(perf_display, use_container_width=True, hide_index=True)
    
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
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily returns
        st.markdown("#### Daily Returns")
        daily_fig = create_time_series_chart(
            stock_returns.to_frame("Daily Return"),
            title="Daily Returns",
        )
        if daily_fig is not None:
            st.plotly_chart(daily_fig, use_container_width=True)
        
        # Weight over time
        st.markdown("#### Position Size Over Time")
        weight_fig = create_time_series_chart(
            stock_weights.to_frame("Weight"),
            title="Portfolio Weight",
        )
        if weight_fig is not None:
            st.plotly_chart(weight_fig, use_container_width=True)


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
    st.dataframe(exp_df, use_container_width=True)
    
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
        
        fig.update_layout(
            **get_plotly_layout(
                title=f"Top {top_n} Factor Exposures",
                height=400,
            ),
            xaxis={"title": "Factor"},
            yaxis={"title": "Exposure (z-score)"},
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor summary from factor_vectors
    st.markdown("### Factor Summary")
    row = factor_vectors.loc[stock_symbol]
    summary_cols = ["mean_score", "score_vol"]
    summary_data = {col: row.get(col, "—") for col in summary_cols if col in factor_vectors.columns}
    
    if summary_data:
        summary_df = pd.DataFrame([summary_data]).T
        summary_df.columns = ["Value"]
        st.dataframe(summary_df, use_container_width=True)


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
        st.dataframe(factor_exp, use_container_width=True)
    
    # Position statistics
    if "weight_stats" in attribution:
        st.markdown("### Position Statistics")
        weight_stats = attribution["weight_stats"]
        stats_df = pd.DataFrame([weight_stats]).T
        stats_df.columns = ["Value"]
        # Format as percentages where appropriate
        for col in ["current", "max", "min", "mean", "mean_when_held"]:
            if col in stats_df.index:
                stats_df.loc[col, "Value"] = f"{stats_df.loc[col, 'Value']:.2%}"
        st.dataframe(stats_df, use_container_width=True)


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
        # Format columns
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
        trades_df["exit_date"] = trades_df["exit_date"].apply(
            lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "OPEN"
        )
        trades_df["entry_weight"] = trades_df["entry_weight"].apply(lambda x: f"{x:.2%}")
        trades_df["max_weight"] = trades_df["max_weight"].apply(lambda x: f"{x:.2%}")
        trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"{x:.4f}")
        
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
        
        trades_display = trades_df[[c for c in display_cols.keys() if c in trades_df.columns]]
        trades_display = trades_display.rename(columns=display_cols)
        
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    
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
                st.plotly_chart(fig, use_container_width=True)
    
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
            
            st.dataframe(changes_df, use_container_width=True, hide_index=True)
