"""
Blotter Page

Portfolio blotter with positions, treemap visualization, single-stock drilldown,
and forward return expectations (T+1, T+5, T+21).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.components.charts import create_treemap_chart, PLOTLY_AVAILABLE
from q23.dashboard.analytics import select_return_series
from q23.dashboard.analytics.stock_elite_analytics import (
    compute_batch_forward_returns,
    format_forward_return_cell,
)


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
                    st.plotly_chart(fig)
                else:
                    st.info("Could not create treemap")
        else:
            st.info("No positions to display")
    else:
        st.info("Install Plotly for interactive treemap: `pip install plotly`")
    
    st.divider()
    
    # ==========================================================================
    # POSITIONS TABLE WITH FORWARD RETURNS
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
    
    # Compute forward return expectations
    portfolio_returns = select_return_series(data.diag)
    fwd_returns_df = _compute_blotter_forward_returns(
        blot.index.tolist(),
        data.factor_vectors,
        data.weights,
        portfolio_returns,
    )
    
    # Join forward returns if computed
    if fwd_returns_df is not None and not fwd_returns_df.empty:
        blot = blot.join(fwd_returns_df, how="left")
    
    blot = blot.sort_values(
        ["side", "abs_w"], ascending=[True, False]
    ).drop(columns=["abs_w"])
    
    st.markdown("### Positions (with PM summaries & Forward Returns)")
    
    # Show forward return methodology
    with st.expander("Forward Return Methodology", expanded=False):
        st.markdown("""
        **Expected Forward Returns** are computed using Factor IC-based estimation:
        
        - `E[R]` = Σ (factor_exposure × IC) × √horizon
        - Range = E[R] ± σ_stock × √horizon
        
        **Horizons:**
        - `E[T+1]`: Expected 1-day return
        - `E[T+5]`: Expected 5-day return  
        - `E[T+21]`: Expected 21-day return (approx. 1 month)
        
        *Note: These are estimates based on historical factor exposures and may not predict actual future returns.*
        """)
    
    # Format display DataFrame
    display_blot = _format_blotter_display(blot)
    
    # Display with conditional formatting
    st.dataframe(display_blot, width="stretch", height=500)
    
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
                st.dataframe(sview, width="stretch", height=320)
            with right:
                st.markdown("**Factor Exposures (top 20)**")
                expos_vec = row[fac_cols].astype(float)
                expos_vec = expos_vec.reindex(
                    expos_vec.abs().sort_values(ascending=False).head(20).index
                )
                st.dataframe(
                    expos_vec.to_frame("exposure"),
                    width="stretch",
                    height=320
                )
        else:
            st.info("No matching symbols in factor_vectors")


def _compute_blotter_forward_returns(
    symbols: list,
    factor_vectors: Optional[pd.DataFrame],
    weights: Optional[pd.DataFrame],
    portfolio_returns: Optional[pd.Series],
) -> Optional[pd.DataFrame]:
    """
    Compute forward return expectations for blotter positions.
    
    Returns DataFrame with formatted E[T+1], E[T+5], E[T+21] columns.
    """
    if not symbols:
        return None
    
    try:
        fwd_df = compute_batch_forward_returns(
            symbols=symbols,
            factor_vectors=factor_vectors if factor_vectors is not None else pd.DataFrame(),
            weights=weights if weights is not None else pd.DataFrame(),
            portfolio_returns=portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),
            horizons=[1, 5, 21],
        )
        
        if fwd_df.empty:
            return None
        
        # Create formatted columns for display
        result = pd.DataFrame(index=fwd_df.index)
        
        for h in [1, 5, 21]:
            exp_col = f"exp_t{h}"
            upper_col = f"upper_t{h}"
            lower_col = f"lower_t{h}"
            
            if exp_col in fwd_df.columns:
                # Create formatted display column
                result[f"E[T+{h}]"] = fwd_df.apply(
                    lambda row: format_forward_return_cell(
                        row.get(exp_col, 0.0),
                        row.get(lower_col, 0.0),
                        row.get(upper_col, 0.0),
                    ),
                    axis=1
                )
        
        return result
        
    except Exception:
        # Fail gracefully - return None to skip forward returns
        return None


def _format_blotter_display(blot: pd.DataFrame) -> pd.DataFrame:
    """
    Format blotter DataFrame for display with appropriate number formatting.
    """
    display = blot.copy()
    
    # Format weight as percentage
    if "weight" in display.columns:
        display["weight"] = display["weight"].apply(lambda x: f"{x:.2%}")
    
    # Format P&L columns
    pnl_cols = ["total_pnl_contrib", "pnl_per_day_held"]
    for col in pnl_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
    
    # Format score columns
    score_cols = ["mean_score", "score_vol"]
    for col in score_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )
    
    # Format weight columns
    weight_cols = ["avg_weight", "avg_weight_when_held"]
    for col in weight_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
    
    # Format days held
    if "days_held" in display.columns:
        display["days_held"] = display["days_held"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "—"
        )
    
    return display
