"""
Performance Page

Detailed performance analytics with cumulative returns, drawdowns, and calendar heatmaps.
Includes transaction cost comparison between flat BPS and Quantiacs ATR models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    select_return_series,
    drawdown,
    create_cumulative_return_chart,
    create_drawdown_chart,
    create_calendar_heatmap,
    compute_comprehensive_performance,
    format_performance_table,
)
from q23.dashboard.analytics.performance import compute_tc_comparison
from q23.dashboard.admin_settings import get_session_tc_config
from q23.shared.config import TransactionCostConfig, TransactionCostScheme


def render_performance_page(data: DashboardData, tc_config: Optional[TransactionCostConfig] = None) -> None:
    """
    Render the Performance analytics page.
    
    Args:
        data: Dashboard data bundle
        tc_config: Transaction cost configuration (uses session default if None)
    """
    st.subheader("Performance Analytics")
    
    # Get TC config from session if not provided
    if tc_config is None:
        tc_config = get_session_tc_config()
    
    ret_series = select_return_series(data.diag)
    dd_series = drawdown(ret_series) if ret_series is not None else None
    
    # Summary metrics row
    colA, colB, colC, colD = st.columns(4)
    
    if ret_series is not None and not ret_series.empty:
        latest_ret = float(ret_series.iloc[-1])
        colA.metric("Last daily ret", f"{latest_ret:.2%}")
    else:
        colA.metric("Last daily ret", "—")
    
    if dd_series is not None and not dd_series.empty:
        current_dd = float(dd_series.iloc[-1])
        max_dd = float(dd_series.min())
        colB.metric("Current DD", f"{current_dd:.2%}")
        colC.metric("Max DD", f"{max_dd:.2%}")
    else:
        colB.metric("Current DD", "—")
        colC.metric("Max DD", "—")
    
    if data.diag is not None and not data.diag.empty and "rolling_vol" in data.diag.columns:
        colD.metric("Rolling vol", f"{float(data.diag['rolling_vol'].iloc[-1]):.2%}")
    else:
        colD.metric("Rolling vol", "—")
    
    st.divider()
    
    # TC Comparison Section
    st.markdown("### Transaction Cost Impact")
    
    # Show current TC model
    tc_scheme_label = "Quantiacs ATR (5% × ATR(14))" if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR else f"Flat {tc_config.flat_bps:.0f} BPS"
    st.caption(f"Current TC Model: **{tc_scheme_label}** (change in sidebar or Ops Console)")
    
    if data.diag is not None and not data.diag.empty:
        # Show TC metrics if available
        has_atr_tc = "tc_cost_atr" in data.diag.columns
        has_flat_tc = "tc_cost_flat" in data.diag.columns
        
        tc_col1, tc_col2, tc_col3, tc_col4 = st.columns(4)
        
        # ATR-based TC
        if has_atr_tc:
            tc_atr = data.diag["tc_cost_atr"]
            annual_tc_atr = float(tc_atr.mean() * 252)
            tc_col1.metric("TC Drag (ATR)", f"{annual_tc_atr:.2%}", help="Quantiacs ATR model")
        else:
            tc_col1.metric("TC Drag (ATR)", "—", help="Run strategy with ATR TC enabled")
        
        # Flat BPS TC
        if has_flat_tc:
            tc_flat = data.diag["tc_cost_flat"]
            annual_tc_flat = float(tc_flat.mean() * 252)
            tc_col2.metric("TC Drag (Flat)", f"{annual_tc_flat:.2%}", help="Flat basis points model")
        elif "turnover" in data.diag.columns:
            # Estimate from turnover
            turnover = data.diag["turnover"]
            annual_tc_flat = float(turnover.mean() * 252 * (tc_config.flat_bps / 10000))
            tc_col2.metric("TC Drag (Flat est)", f"{annual_tc_flat:.2%}")
        else:
            tc_col2.metric("TC Drag (Flat)", "—")
        
        # Show Sharpe comparison
        if ret_series is not None and len(ret_series) > 20:
            ann_ret = float(ret_series.mean() * 252)
            ann_vol = float(ret_series.std() * np.sqrt(252))
            gross_sharpe = ann_ret / (ann_vol + 1e-12)
            
            # Net Sharpe calculations
            if has_atr_tc:
                net_ret_atr = ret_series - tc_atr.reindex(ret_series.index).fillna(0.0)
                net_sharpe_atr = float(net_ret_atr.mean() * 252) / (float(net_ret_atr.std() * np.sqrt(252)) + 1e-12)
                tc_col3.metric("Sharpe (ATR)", f"{net_sharpe_atr:.3f}", delta=f"{net_sharpe_atr - gross_sharpe:.3f}")
            else:
                tc_col3.metric("Sharpe (ATR)", "—")
            
            tc_col4.metric("Sharpe (Gross)", f"{gross_sharpe:.3f}")
        else:
            tc_col3.metric("Sharpe (ATR)", "—")
            tc_col4.metric("Sharpe (Gross)", "—")
        
        # TC Sensitivity Analysis
        with st.expander("📊 TC Sensitivity Analysis", expanded=False):
            if ret_series is not None and data.weights is not None and len(ret_series) > 20:
                tc_comparison = compute_tc_comparison(
                    data.weights,
                    ret_series,
                    tc_bps_values=[0, 5, 10, 15, 20, 30, 50],
                )
                
                st.markdown("**Sharpe ratio at different TC assumptions:**")
                
                # Format for display
                display_df = tc_comparison.copy()
                display_df["annual_return"] = display_df["annual_return"].apply(lambda x: f"{x:.2%}")
                display_df["annual_vol"] = display_df["annual_vol"].apply(lambda x: f"{x:.2%}")
                display_df["sharpe"] = display_df["sharpe"].apply(lambda x: f"{x:.3f}")
                display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.2%}")
                display_df["annual_tc_drag"] = display_df["annual_tc_drag"].apply(lambda x: f"{x:.2%}")
                display_df.columns = ["TC (bps)", "Ann Return", "Ann Vol", "Sharpe", "Max DD", "TC Drag"]
                
                st.dataframe(display_df, width='stretch', hide_index=True)
            else:
                st.info("Not enough data for TC sensitivity analysis")
    
    st.divider()
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Cumulative Return")
        if ret_series is not None and not ret_series.empty:
            fig = create_cumulative_return_chart(ret_series, title="Cumulative Return")
            st.pyplot(fig)
        else:
            st.info("No return series available")
        
        st.markdown("### Drawdown")
        if dd_series is not None and not dd_series.empty:
            fig = create_drawdown_chart(dd_series, title="Drawdown")
            st.pyplot(fig)
        else:
            st.info("No drawdown available")
    
    with c2:
        st.markdown("### Calendar Heatmap (Recent Year)")
        if ret_series is not None and not ret_series.empty:
            recent = ret_series.tail(380)
            if len(recent) > 50:
                fig = create_calendar_heatmap(recent, title="Daily Returns Heatmap")
                st.pyplot(fig)
            else:
                st.info("Not enough data for calendar heatmap")
        else:
            st.info("No return series available")
        
        # Show performance table with TC comparison
        st.markdown("### Performance Summary")
        if data.weights is not None and not data.weights.empty:
            perf = compute_comprehensive_performance(
                data.weights,
                diag=data.diag,
                tc_config=tc_config,
            )
            perf_table = format_performance_table(perf, show_tc_comparison=True)
            st.dataframe(perf_table, width='stretch', hide_index=True, height=400)
        else:
            st.info("No weights data available")
