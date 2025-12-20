"""
Factors/IC Page

Factor weights, IC analysis, and portfolio factor exposure.
"""

from __future__ import annotations

import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import create_factor_ic_chart


def render_factors_page(data: DashboardData) -> None:
    """
    Render the Factors/IC page.
    
    Args:
        data: Dashboard data bundle
    """
    st.subheader("Factors / IC")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Factor Weights (Time-Series)")
        if data.factor_weights is not None and not data.factor_weights.empty:
            # Select top factors by average absolute weight
            top = (
                data.factor_weights.abs()
                .mean(axis=0)
                .sort_values(ascending=False)
                .head(12)
                .index.tolist()
            )
            sel = st.multiselect(
                "Factors",
                options=list(data.factor_weights.columns),
                default=top
            )
            if sel:
                st.line_chart(data.factor_weights[sel], height=320)
        else:
            st.info("No factor_weights available")
    
    with c2:
        st.markdown("### Factor IC (Raw vs Smooth)")
        if data.ic is not None and not data.ic.empty:
            factors = sorted({c[0] for c in data.ic.columns})
            fsel = st.multiselect(
                "IC factors",
                options=factors,
                default=factors[:min(6, len(factors))]
            )
            if fsel:
                fig = create_factor_ic_chart(data.ic, fsel, title="Factor IC")
                st.pyplot(fig)
        else:
            st.info("No IC data available")
    
    st.divider()
    
    # Portfolio factor exposure
    st.markdown("### Portfolio Factor Exposure")
    if data.exposure is not None and not data.exposure.empty:
        top = (
            data.exposure.abs()
            .mean(axis=0)
            .sort_values(ascending=False)
            .head(12)
            .index.tolist()
        )
        sel = st.multiselect(
            "Exposure factors",
            options=list(data.exposure.columns),
            default=top
        )
        if sel:
            st.line_chart(data.exposure[sel], height=320)
    else:
        st.info("No exposure data available")
