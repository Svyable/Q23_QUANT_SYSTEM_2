"""
What-If Page

Interactive portfolio rebalancing and transformation without re-running strategy.
Includes transaction cost impact analysis for proposed trades.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    WhatIfEngine,
    WhatIfParams,
    compute_metric_deltas,
    summary_exposure,
)
from q23.dashboard.admin_settings import get_session_tc_config
from q23.shared.config import TransactionCostConfig, TransactionCostScheme


def _estimate_tc_for_trades(
    delta: pd.Series,
    tc_config: TransactionCostConfig,
    typical_atr_pct: float = 0.015,
) -> dict:
    """Estimate transaction costs for proposed trades.
    
    Args:
        delta: Weight changes (position changes)
        tc_config: TC configuration
        typical_atr_pct: Assumed typical ATR as % of price
        
    Returns:
        Dictionary with TC estimates
    """
    total_turnover = float(delta.abs().sum())
    
    # Flat BPS estimate
    flat_tc = total_turnover * (tc_config.flat_bps / 10000.0)
    
    # ATR-based estimate (using typical ATR)
    atr_tc = total_turnover * tc_config.atr_multiplier * typical_atr_pct
    
    return {
        "total_turnover": total_turnover,
        "tc_flat_bps": flat_tc,
        "tc_flat_bps_pct": flat_tc * 100,
        "tc_atr": atr_tc,
        "tc_atr_pct": atr_tc * 100,
        "tc_atr_bps_equiv": atr_tc * 10000 / (total_turnover + 1e-12),
    }


def render_whatif_page(
    data: DashboardData,
    tag: str,
    tc_config: Optional[TransactionCostConfig] = None,
) -> None:
    """
    Render the What-If Rebalance Engine page.
    
    Args:
        data: Dashboard data bundle
        tag: Current run tag
        tc_config: Transaction cost configuration (uses session default if None)
    """
    st.subheader("What-If Rebalance Engine")
    
    # Get TC config
    if tc_config is None:
        tc_config = get_session_tc_config()
    
    st.markdown(
        """
        Transform the current portfolio **without re-running the strategy**. 
        Apply controls (gross/net targets, seat selection, overlays, constraints) 
        and see metric deltas in real-time. **Includes TC impact preview.**
        """
    )
    
    baseline_w = data.w_last.copy()
    baseline_metrics = summary_exposure(baseline_w)
    
    st.divider()
    
    # TC Model indicator
    tc_scheme_label = "Quantiacs ATR" if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR else f"Flat {tc_config.flat_bps:.0f} BPS"
    st.caption(f"💰 TC Model: **{tc_scheme_label}** (change in sidebar)")
    
    # Target controls
    col1, col2, col3 = st.columns(3)
    with col1:
        gross_target = st.number_input("Gross Target", 0.1, 5.0, 1.0, 0.05)
    with col2:
        net_target_enabled = st.checkbox("Set Net Target", value=False)
        net_target = (
            st.number_input("Net Target", -2.0, 2.0, 0.0, 0.05)
            if net_target_enabled else None
        )
    with col3:
        tilt_alpha = st.slider("Tilt Alpha", 0.0, 2.0, 0.0, 0.05)
    
    # Seat controls
    col4, col5 = st.columns(2)
    with col4:
        topn_long_enabled = st.checkbox("Limit Long Seats", value=False)
        topn_long = (
            st.number_input("Top N Long", 1, 100, 20, 1)
            if topn_long_enabled else None
        )
    with col5:
        topn_short_enabled = st.checkbox("Limit Short Seats", value=False)
        topn_short = (
            st.number_input("Top N Short", 1, 100, 10, 1)
            if topn_short_enabled else None
        )
    
    # Position size controls
    col6, col7 = st.columns(2)
    with col6:
        max_pos_whatif = st.number_input("Max Position", 0.01, 0.50, 0.10, 0.01)
    with col7:
        min_pos_whatif = st.number_input("Min Position", 0.0, 0.05, 0.002, 0.001)
    
    # Overlay / Blacklist / Pins
    st.markdown("#### Overlay / Blacklist / Pins")
    overlay_text = st.text_area(
        "Weight Overlay (symbol=weight, one per line)",
        value="",
        height=80,
        placeholder="AAPL=0.05\nMSFT=0.03",
    )
    blacklist_text = st.text_input("Blacklist (comma-separated)", value="")
    pins_text = st.text_input("Pin List (comma-separated)", value="")
    
    # Parse overlay
    overlay_dict = {}
    if overlay_text.strip():
        for line in overlay_text.strip().split("\n"):
            if "=" in line:
                sym, w_str = line.split("=", 1)
                try:
                    overlay_dict[sym.strip()] = float(w_str.strip())
                except ValueError:
                    pass
    
    blacklist_list = [x.strip() for x in blacklist_text.split(",") if x.strip()]
    pins_list = [x.strip() for x in pins_text.split(",") if x.strip()]
    
    # Rounding
    round_to_enabled = st.checkbox("Round weights", value=False)
    round_to = (
        st.number_input("Round to", 0.001, 0.1, 0.005, 0.001)
        if round_to_enabled else None
    )
    
    # Apply button
    apply_whatif = st.button("🔄 Apply What-If Transform", type="primary")
    
    if apply_whatif:
        params = WhatIfParams(
            gross_target=gross_target,
            net_target=net_target,
            max_pos=max_pos_whatif,
            min_pos=min_pos_whatif,
            topn_long=topn_long,
            topn_short=topn_short,
            tilt_alpha=tilt_alpha,
            weight_overlay=overlay_dict,
            blacklist=blacklist_list,
            pin_list=pins_list,
            round_to=round_to,
        )
        
        engine = WhatIfEngine(baseline_w)
        result = engine.apply_transform(params)
        
        # Estimate TC for the proposed trades
        tc_estimate = _estimate_tc_for_trades(result.delta, tc_config)
        
        st.success(f"✅ Transformed. Turnover: {result.turnover:.2%}")
        
        st.divider()
        
        # TC Impact section
        st.markdown("### 💰 Transaction Cost Impact")
        
        tc_col1, tc_col2, tc_col3, tc_col4 = st.columns(4)
        
        tc_col1.metric(
            "Total Turnover",
            f"{tc_estimate['total_turnover']:.2%}",
            help="Sum of |ΔWeight| across all positions",
        )
        
        if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
            tc_col2.metric(
                "Est. TC (ATR)",
                f"{tc_estimate['tc_atr_pct']:.3f}%",
                help=f"Quantiacs model: {tc_config.atr_multiplier:.0%} × ATR(14)",
            )
            tc_col3.metric(
                "Equiv. BPS",
                f"{tc_estimate['tc_atr_bps_equiv']:.1f}",
                help="Equivalent flat basis points",
            )
        else:
            tc_col2.metric(
                "Est. TC (Flat)",
                f"{tc_estimate['tc_flat_bps_pct']:.3f}%",
                help=f"Flat {tc_config.flat_bps:.0f} bps model",
            )
            tc_col3.metric(
                "TC BPS",
                f"{tc_config.flat_bps:.1f}",
                help="Configured flat basis points",
            )
        
        # Break-even analysis
        if result.turnover > 1e-12:
            # How much return do we need to cover TC?
            primary_tc = tc_estimate['tc_atr'] if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR else tc_estimate['tc_flat_bps']
            tc_col4.metric(
                "Break-even Return",
                f"{primary_tc:.3%}",
                help="Minimum return needed to cover TC of this trade",
            )
        else:
            tc_col4.metric("Break-even Return", "—")
        
        # TC Sensitivity for this trade
        with st.expander("📊 TC Sensitivity for This Trade", expanded=False):
            st.markdown("**Cost at different TC assumptions:**")
            
            tc_levels = [5, 10, 15, 20, 30, 50]
            tc_data = []
            for bps in tc_levels:
                cost = tc_estimate['total_turnover'] * (bps / 10000.0)
                tc_data.append({
                    "TC (bps)": bps,
                    "Cost (%)": f"{cost * 100:.3f}%",
                    "Cost ($10M AUM)": f"${cost * 10_000_000:,.0f}",
                })
            
            st.dataframe(pd.DataFrame(tc_data), width='stretch', hide_index=True)
        
        st.divider()
        st.markdown("### What Changed vs Baseline")
        
        delta_df = compute_metric_deltas(result, baseline_metrics)
        st.dataframe(delta_df, height=300)
        
        st.divider()
        st.markdown("### Weight Changes (Top 30 by |delta|)")
        delta_sorted = result.delta.abs().sort_values(ascending=False).head(30)
        
        # Add TC estimate per position
        change_df = pd.DataFrame({
            "baseline": result.baseline_weights.reindex(delta_sorted.index).fillna(0.0),
            "whatif": result.transformed_weights.reindex(delta_sorted.index).fillna(0.0),
            "delta": result.delta.reindex(delta_sorted.index).fillna(0.0),
        })
        
        # Add per-position TC estimate
        if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
            typical_atr = 0.015  # 1.5%
            change_df["est_tc_bps"] = change_df["delta"].abs() * tc_config.atr_multiplier * typical_atr * 10000
        else:
            change_df["est_tc_bps"] = change_df["delta"].abs() * tc_config.flat_bps
        
        change_df["est_tc_bps"] = change_df["est_tc_bps"].round(2)
        
        st.dataframe(change_df, height=500)
        
        # Download button
        csv_bytes = change_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download What-If Trades CSV",
            data=csv_bytes,
            file_name=f"whatif_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
