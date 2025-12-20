"""
Weights Explorer Page

Interactive exploration of portfolio weights over time with treemap visualization.
"""

from __future__ import annotations

import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.components.charts import create_treemap_chart, PLOTLY_AVAILABLE
from q23.dashboard.components.styles import (
    style_long_weights_gradient,
    style_short_weights_gradient,
    style_weights_diverging_gradient,
)


def render_weights_page(data: DashboardData) -> None:
    """
    Render the Weights Explorer page.
    
    Args:
        data: Dashboard data bundle
    """
    st.subheader("Weights Explorer")
    
    # Controls
    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        n_days = st.number_input(
            "Last N days",
            min_value=10,
            max_value=4000,
            value=252,
            step=10
        )
    with c2:
        topk = st.number_input(
            "Top K assets",
            min_value=10,
            max_value=300,
            value=60,
            step=5
        )
    with c3:
        st.caption("Tip: keep Top K modest for performance")
    
    # Compute filtered weights
    tail = data.weights.tail(int(n_days))
    latest = tail.iloc[-1]
    keep_cols = latest.abs().sort_values(ascending=False).head(int(topk)).index
    tail_small = tail[keep_cols]
    
    # ==========================================================================
    # TREEMAP VISUALIZATION
    # ==========================================================================
    st.markdown("### Position Treemap")
    
    if PLOTLY_AVAILABLE:
        # Get current weights for treemap
        w_last = data.w_last
        w_nonzero = w_last[w_last.abs() > 1e-12]
        
        if len(w_nonzero) > 0:
            col1, col2 = st.columns([3, 1])
            with col2:
                color_by_value = st.checkbox(
                    "Color by long/short",
                    value=True,
                    help="Color positions by direction (green=long, red=short)"
                )
            
            with col1:
                fig = create_treemap_chart(
                    w_nonzero,
                    title="Current Position Sizes",
                    color_by_value=color_by_value,
                )
                if fig is not None:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Could not create treemap")
        else:
            st.info("No positions to display in treemap")
    else:
        st.info("Install Plotly for interactive treemap visualization: `pip install plotly`")
    
    st.divider()
    
    # ==========================================================================
    # LONG/SHORT BREAKDOWN
    # ==========================================================================
    st.markdown("### Long/Short Breakdown")
    
    w_last = data.w_last
    longs = w_last[w_last > 1e-12].sort_values(ascending=False)
    shorts = w_last[w_last < -1e-12].sort_values()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Longs** ({len(longs)} positions, {longs.sum():.1%} gross)")
        if not longs.empty:
            long_df = longs.head(30).rename("weight").to_frame()
            styled_long = style_long_weights_gradient(long_df, column="weight")
            st.dataframe(
                styled_long,
                width='stretch',
                height=300
            )
        else:
            st.caption("No long positions")
    
    with col2:
        st.markdown(f"**Shorts** ({len(shorts)} positions, {abs(shorts.sum()):.1%} gross)")
        if not shorts.empty:
            short_df = shorts.head(30).rename("weight").to_frame()
            styled_short = style_short_weights_gradient(short_df, column="weight")
            st.dataframe(
                styled_short,
                width='stretch',
                height=300
            )
        else:
            st.caption("No short positions")
    
    st.divider()
    
    # ==========================================================================
    # HEATMAP VIEW
    # ==========================================================================
    st.markdown("### Weight Heatmap (tail × topK)")
    st.dataframe(
        tail_small.style.format("{:.3%}").background_gradient(cmap="RdYlGn", axis=None),
        width='stretch',
        height=420
    )
    
    # ==========================================================================
    # LATEST SNAPSHOT
    # ==========================================================================
    st.markdown("### Latest Snapshot (all assets)")
    snap = data.weights.loc[data.last_dt].sort_values()
    snap_df = snap.rename("weight").to_frame()
    styled_snap = style_weights_diverging_gradient(snap_df, column="weight")
    st.dataframe(
        styled_snap,
        width='stretch',
        height=420
    )
