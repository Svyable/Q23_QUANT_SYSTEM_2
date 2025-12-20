"""
Rebalance Page

Trade list generation for rebalancing between dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import compute_trade_list


def render_rebalance_page(data: DashboardData, base_name: str) -> None:
    """
    Render the Rebalance Trade List page.
    
    Args:
        data: Dashboard data bundle
        base_name: Base name for export filenames
    """
    st.subheader("Rebalance Trade List")
    
    dates = list(data.weights.index)
    if len(dates) < 2:
        st.warning("Need at least 2 dates")
        st.stop()
    
    # Date selectors
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        t0 = st.selectbox(
            "T0 (target)",
            options=dates,
            index=len(dates) - 1,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
        )
    with c2:
        i0 = dates.index(t0)
        t1 = st.selectbox(
            "T-1 (prev)",
            options=dates,
            index=max(0, i0 - 1),
            format_func=lambda x: x.strftime("%Y-%m-%d"),
        )
    with c3:
        min_trade = st.slider("Min |Δw|", 0.0, 0.05, 0.002, step=0.0005)
    
    if t1 >= t0:
        st.info("Pick T-1 strictly earlier than T0")
        st.stop()
    
    # Compute trades
    trades = compute_trade_list(
        data.weights,
        pd.Timestamp(t0),
        pd.Timestamp(t1),
        min_abs_delta=float(min_trade)
    )
    
    turnover = float(np.abs(trades["delta_w"]).sum())
    st.metric("Turnover Σ|Δw| (filtered)", f"{turnover:.2%}")
    
    # Display trades
    st.dataframe(trades.reset_index(), width='stretch', height=560)
    
    # Download button
    csv_bytes = trades.reset_index().to_csv(index=False).encode("utf-8")
    t0_str = pd.Timestamp(t0).strftime('%Y%m%d')
    t1_str = pd.Timestamp(t1).strftime('%Y%m%d')
    st.download_button(
        "Download trades CSV",
        data=csv_bytes,
        file_name=f"{base_name}_rebalance_{t0_str}_vs_{t1_str}.csv",
        mime="text/csv",
    )
