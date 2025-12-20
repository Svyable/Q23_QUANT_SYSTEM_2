"""
Blotter Page

Portfolio blotter with positions, treemap visualization, and single-stock drilldown.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.components.charts import create_treemap_chart, PLOTLY_AVAILABLE


def render_blotter_page(data: DashboardData, expos: Dict) -> None:
    """
    Render the Portfolio Blotter page.
    
    Args:
        data: Dashboard data bundle
        expos: Exposure summary dict from summary_exposure()
    """
    st.subheader("Portfolio Blotter")
    
    # Summary metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric("As of", str(data.last_dt.date()))
    colB.metric("Positions", expos["n_pos"])
    colC.metric("Gross", f"{expos['gross']:.2f}")
    colD.metric("Net", f"{expos['net']:.2f}")
    
    st.divider()
    
    # ==========================================================================
    # TREEMAP VISUALIZATION
    # ==========================================================================
    st.markdown("### Position Treemap")
    
    if PLOTLY_AVAILABLE:
        w_last = data.w_last
        w_nonzero = w_last[w_last.abs() > 1e-12]
        
        if len(w_nonzero) > 0:
            # Check if we have PnL data for coloring
            has_pnl = (
                data.factor_vectors is not None 
                and not data.factor_vectors.empty
                and "total_pnl_contrib" in data.factor_vectors.columns
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if has_pnl:
                    color_mode = st.radio(
                        "Color by",
                        ["Long/Short", "PnL"],
                        horizontal=True,
                        help="Color by direction or P&L contribution"
                    )
                else:
                    color_mode = "Long/Short"
                    st.caption("PnL coloring available with factor_vectors")
            
            with col1:
                if color_mode == "PnL" and has_pnl:
                    # Use PnL contribution for coloring
                    pnl_data = data.factor_vectors["total_pnl_contrib"]
                    # Create a combined series with weights as size, PnL as color
                    # For now, show the basic treemap with direction coloring
                    fig = create_treemap_chart(
                        w_nonzero,
                        title="Position Sizes (colored by direction)",
                        color_by_value=True,
                    )
                else:
                    fig = create_treemap_chart(
                        w_nonzero,
                        title="Position Sizes (green=long, red=short)",
                        color_by_value=True,
                    )
                
                if fig is not None:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Could not create treemap")
        else:
            st.info("No positions to display")
    else:
        st.info("Install Plotly for interactive treemap: `pip install plotly`")
    
    st.divider()
    
    # ==========================================================================
    # POSITIONS TABLE
    # ==========================================================================
    # Build blotter DataFrame
    blot = pd.DataFrame({"weight": data.w_last})
    blot = blot[blot["weight"].abs() > 1e-12].copy()
    blot["side"] = np.where(blot["weight"] > 0, "LONG", "SHORT")
    blot["abs_w"] = blot["weight"].abs()
    
    # Join factor vectors if available
    if data.factor_vectors is not None and not data.factor_vectors.empty:
        join_cols = [
            c for c in [
                "total_pnl_contrib",
                "pnl_per_day_held",
                "mean_score",
                "score_vol",
                "days_held",
                "avg_weight",
                "avg_weight_when_held",
            ]
            if c in data.factor_vectors.columns
        ]
        if join_cols:
            blot = blot.join(data.factor_vectors[join_cols], how="left")
    
    blot = blot.sort_values(
        ["side", "abs_w"], ascending=[True, False]
    ).drop(columns=["abs_w"])
    
    st.markdown("### Positions (with PM summaries)")
    
    # Format and display
    st.dataframe(blot, width='stretch', height=500)
    
    # Download button
    csv_bytes = blot.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Blotter CSV",
        data=csv_bytes,
        file_name=f"blotter_{data.last_dt.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    
    st.divider()
    
    # ==========================================================================
    # SINGLE-STOCK DRILLDOWN
    # ==========================================================================
    st.markdown("### Single-Stock Drilldown")
    
    if data.factor_vectors is None or data.factor_vectors.empty:
        st.info("factor_vectors not available for drilldown")
    else:
        symbols = sorted(
            set(blot.index).intersection(set(data.factor_vectors.index))
        )
        if symbols:
            sym = st.selectbox("Symbol", options=symbols)
            row = data.factor_vectors.loc[sym].copy()
            
            summary_cols = {
                "total_pnl_contrib",
                "pnl_per_day_held",
                "mean_score",
                "score_vol",
                "days_held",
                "avg_weight",
                "avg_weight_when_held",
            }
            fac_cols = [
                c for c in data.factor_vectors.columns
                if c not in summary_cols
            ]
            
            left, right = st.columns([1, 1.2])
            with left:
                st.markdown("**Summary**")
                sview = row[[c for c in row.index if c in summary_cols]].to_frame("value")
                st.dataframe(sview, width='stretch', height=320)
            with right:
                st.markdown("**Factor Exposures (top 20)**")
                expos_vec = row[fac_cols].astype(float)
                expos_vec = expos_vec.reindex(
                    expos_vec.abs().sort_values(ascending=False).head(20).index
                )
                st.dataframe(
                    expos_vec.to_frame("exposure"),
                    width='stretch',
                    height=320
                )
        else:
            st.info("No matching symbols in factor_vectors")
