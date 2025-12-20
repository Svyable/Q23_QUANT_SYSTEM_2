"""
Attribution Page

Factor attribution and return decomposition analysis.
"""

from __future__ import annotations

import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    select_return_series,
    compute_rolling_factor_regression,
    compute_factor_vs_idio_decomposition,
)


def render_attribution_page(data: DashboardData) -> None:
    """
    Render the Factor Attribution & Decomposition page.
    
    Args:
        data: Dashboard data bundle
    """
    st.subheader("Factor Attribution & Decomposition")
    
    ret_series = select_return_series(data.diag)
    
    if ret_series is None or ret_series.empty:
        st.warning("No return series available for attribution")
        st.stop()
    
    if data.exposure is None or data.exposure.empty:
        st.warning("No factor exposure available for attribution")
        st.stop()
    
    # Rolling Factor Regression
    st.markdown("### Rolling Factor Regression")
    window = st.slider("Regression Window (days)", 60, 500, 252, 10)
    
    if data.diag is not None and "active_ret" in data.diag.columns:
        active_ret = data.diag["active_ret"]
        regression_results = compute_rolling_factor_regression(
            active_ret, data.exposure, window=int(window)
        )
        
        if not regression_results.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Alpha & R²**")
                st.line_chart(regression_results[["alpha", "r_squared"]], height=300)
            with c2:
                st.markdown("**Factor vs Idio Returns**")
                st.line_chart(
                    regression_results[["factor_explained_ret", "idio_ret"]],
                    height=300
                )
        else:
            st.info("Not enough data for rolling regression")
    else:
        st.info("No active return column in diagnostics")
    
    st.divider()
    
    # Factor vs Idiosyncratic Decomposition
    st.markdown("### Factor vs Idiosyncratic Decomposition")
    
    if data.exposure is not None and ret_series is not None:
        # Compute factor returns proxy: change in exposures
        factor_returns = data.exposure.diff().fillna(0.0)
        
        # Validate column alignment
        if not factor_returns.columns.equals(data.exposure.columns):
            factor_returns = factor_returns.reindex(
                columns=data.exposure.columns,
                fill_value=0.0
            )
        
        try:
            decomp = compute_factor_vs_idio_decomposition(
                ret_series, data.exposure, factor_returns
            )
            
            if not decomp.empty:
                st.line_chart(
                    decomp[["factor_ret_cum", "idio_ret_cum", "total_ret_cum"]],
                    height=400
                )
            else:
                st.info("Could not compute decomposition - insufficient overlapping data")
        except Exception as e:
            st.error(f"Error computing decomposition: {str(e)}")
            st.info(
                "Please check that return series and factor exposures "
                "have compatible time indices"
            )
