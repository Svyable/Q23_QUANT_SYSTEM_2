from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.analytics import (
    WhatIfEngine,
    WhatIfParams,
    compute_metric_deltas,
    summary_exposure,
    top_positions,
    period_windows,
    drawdown,
    compute_trade_list,
    hit_rate,
    winner_loser_sizing,
    rolling_sharpe,
    create_cumulative_return_chart,
    create_drawdown_chart,
    create_exposure_time_series,
    create_factor_ic_chart,
    create_calendar_heatmap,
    compute_rolling_factor_regression,
    compute_factor_vs_idio_decomposition,
)
from q23.dashboard.core import (
    _project_root,
    _base_output_dir,
    _discover_tags,
    load_dashboard_data,
    discover_available_strategies,
    discover_strategy_tags,
    get_strategy_output_dir,
    get_strategy_display_name,
    get_strategy_configs,
    load_strategy_dashboard_data,
    get_date_preset_range,
    get_default_date_range,
)
from q23.shared.config import cfg


st.set_page_config(page_title="Q23 ELITE PM Dashboard", layout="wide")
st.title("Q23 ELITE PM Dashboard — Multi-Strategy + What-If + Advanced Viz")

proj = _project_root()
date_defaults = get_default_date_range()

with st.sidebar:
    st.header("Strategy Selection")

    available_strategies = discover_available_strategies()
    if not available_strategies:
        st.warning("No strategies registered. Falling back to legacy mode.")
        use_multi_strategy = False
        selected_strategy = None
    else:
        use_multi_strategy = True
        strategy_configs = get_strategy_configs()

        selected_strategy = st.selectbox(
            "Active Strategy",
            options=available_strategies,
            format_func=lambda x: get_strategy_display_name(x),
            index=0,
        )

        if selected_strategy and selected_strategy in strategy_configs:
            cfg_info = strategy_configs[selected_strategy]
            st.caption(f"v{cfg_info.version} | {len(cfg_info.factors)} factors | {cfg_info.min_date}+")

    st.divider()
    st.header("Date Range")

    date_preset = st.radio(
        "Quick Select",
        ["2025 YTD", "2024 Full", "2024 + 2025 YTD", "All Time"],
        index=0,
        horizontal=True,
    )

    preset_start, preset_end = get_date_preset_range(date_preset)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start",
            value=pd.to_datetime(preset_start),
            min_value=pd.to_datetime("2004-01-01"),
        )
    with col_d2:
        end_date = st.date_input(
            "End",
            value=pd.to_datetime(preset_end),
        )

    date_range = (str(start_date), str(end_date))

    st.divider()
    st.header("Run Selection")

    if use_multi_strategy and selected_strategy:
        strategy_dir = get_strategy_output_dir(selected_strategy)
        base_name = selected_strategy
        base_dir = strategy_dir

        st.caption(f"Strategy: `{selected_strategy}`")
        st.caption(f"Outputs: `{strategy_dir.name}`")

        tags = discover_strategy_tags(selected_strategy)
    else:
        base_dir = _base_output_dir()
        base_name = cfg.paths.BASE_NAME

        st.caption(f"Project: `{proj.name}`")
        st.caption(f"Outputs: `{base_dir.name}`")
        st.caption(f"BASE_NAME: `{base_name}`")

        tags = _discover_tags(base_dir, base_name)

    if not tags:
        # Provide helpful debugging info
        debug_info = []
        if base_dir.exists():
            all_csvs = list(base_dir.glob("*_wide_weights_*.csv"))
            if all_csvs:
                debug_info.append(f"Found {len(all_csvs)} wide_weights files:")
                for f in sorted(all_csvs)[:5]:  # Show first 5
                    debug_info.append(f"  - {f.name}")
                if len(all_csvs) > 5:
                    debug_info.append(f"  ... and {len(all_csvs) - 5} more")
            else:
                debug_info.append(f"No `*_wide_weights_*.csv` files found in `{base_dir}`")
        else:
            debug_info.append(f"Directory `{base_dir}` does not exist")
        
        error_msg = (
            f"No tags found. Expected files like `{base_name}_wide_weights_<tag>.csv` under:\n"
            f"`{base_dir}`\n\n"
        )
        if debug_info:
            error_msg += "\n".join(debug_info) + "\n\n"
        error_msg += "Run strategies first."
        
        st.error(error_msg)
        st.stop()

    tag = st.selectbox("Tag", options=tags, index=0)

    st.divider()
    
    page = st.radio(
        "Page",
        [
            "Overview",
            "Performance",
            "Weights",
            "Factors/IC",
            "Attribution",
            "Diagnostics",
            "What-If",
            "Live Strategy",
            "Blotter",
            "Rebalance",
            "Strategy Comparison",
        ],
        index=0,
    )
    
    st.divider()
    elite_section = st.expander("⚡ ELITE ANALYTICS", expanded=False)
    with elite_section:
        elite_page = st.radio(
            "Elite Features",
            [
                "None",
                "📋 Executive Summary",
                "Risk Attribution",
                "Brinson Attribution",
                "Ex-Ante Risk",
                "Sector Analysis",
                "Correlation Analysis",
                "Beta Analysis",
                "Regime Analysis",
                "Tail Risk & Stress",
                "Convexity & Gamma",
                "Alpha Decay",
                "Capacity Estimation",
                "A/B Testing",
                "Factor Timing",
                "Rotation Velocity",
                "Rank IC & Quintiles",
            ],
            index=0,
            key="elite_page_selector",
        )


if use_multi_strategy and selected_strategy:
    data = load_strategy_dashboard_data(selected_strategy, tag, date_range=date_range)
else:
    data = load_dashboard_data(tag, base_dir, base_name)

# Store base_dir for Live Strategy page
if "strategy_base_dir" not in st.session_state:
    st.session_state.strategy_base_dir = base_dir

if data.weights is None or data.weights.empty:
    st.error("Could not load weights for this tag.")
    st.stop()

expos = summary_exposure(data.w_last)

if page == "Overview":
    st.markdown("## 📊 STRATEGY PERFORMANCE OVERVIEW")
    st.markdown("---")

    from q23.dashboard.analytics.performance import (
        compute_comprehensive_performance,
        format_performance_table,
    )

    tc_bps = float(getattr(cfg.strategy, "TC_BASIS_POINTS", 10.0))
    perf = compute_comprehensive_performance(
        data.weights, returns=None, diag=data.diag, tc_bps=tc_bps
    )

    st.markdown("### 📈 Risk-Adjusted Returns")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Annual Return", f"{perf['annual_return']:.2%}")
    r2.metric("Sharpe Ratio", f"{perf['sharpe']:.3f}")
    r3.metric("Sortino Ratio", f"{perf['sortino']:.3f}")
    r4.metric("Calmar Ratio", f"{perf['calmar']:.3f}" if np.isfinite(perf['calmar']) else "—")
    r5.metric("Volatility", f"{perf['annual_vol']:.2%}")

    st.markdown("### 📉 Drawdown & Tail Risk")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Max Drawdown", f"{perf['max_drawdown']:.2%}")
    d2.metric("Current DD", f"{perf['current_drawdown']:.2%}")
    d3.metric("Worst Day", f"{perf['worst_day']:.2%}")
    d4.metric("Worst Month", f"{perf['worst_month']:.2%}")

    st.markdown("### 🎯 Win/Loss Profile")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Win Rate", f"{perf['win_rate']:.1%}")
    w2.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
    w3.metric("Avg Win", f"{perf['avg_win']:.3%}")
    w4.metric("Avg Loss", f"{perf['avg_loss']:.3%}")

    st.markdown("### 💼 Portfolio Snapshot (Latest)")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("As of", str(data.last_dt.date()))
    p2.metric("Positions", f"{perf['n_positions']:.0f}")
    p3.metric("Gross", f"{perf['gross_exposure']:.1%}")
    p4.metric("Net", f"{perf['net_exposure']:.1%}")
    p5.metric("Top 5 Conc.", f"{perf['top5_concentration']:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Long Exposure", f"{perf['long_exposure']:.1%}")
    with col2:
        st.metric("Short Exposure", f"{perf['short_exposure']:.1%}")

    st.markdown("### 🔄 Trading Activity")
    t1, t2, t3 = st.columns(3)
    t1.metric("Avg Daily Turnover", f"{perf['avg_turnover']:.2%}")
    t2.metric("Est. Annual TC Drag", f"{perf['est_annual_tc_drag']:.2%}")
    t3.metric("Net of TC", f"{perf['net_return_after_tc']:.2%}")

    st.markdown("---")

    perf_windows = period_windows(data.diag, data.last_dt)
    if perf_windows:
        st.markdown("### 📅 Period Returns")
        c6, c7, c8 = st.columns(3)
        wtd = perf_windows.get("wtd", float("nan"))
        mtd = perf_windows.get("mtd", float("nan"))
        ytd = perf_windows.get("ytd", float("nan"))
        if np.isfinite(wtd):
            c6.metric("WTD", f"{wtd:.2%}")
        if np.isfinite(mtd):
            c7.metric("MTD", f"{mtd:.2%}")
        if np.isfinite(ytd):
            c8.metric("YTD", f"{ytd:.2%}")

    st.markdown("---")

    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("### 📊 Exposure & Turnover Time Series")
        if data.diag is not None and not data.diag.empty:
            fig = create_exposure_time_series(data.diag, title="Exposure & Turnover")
            st.pyplot(fig)
        else:
            st.info("No portfolio diagnostics available")

    with right:
        st.markdown("### 🎯 Top Positions")
        long_df, short_df = top_positions(data.w_last, n=12)
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**Top Longs**")
            st.dataframe(long_df, height=280)
        with rc:
            st.markdown("**Top Shorts**")
            st.dataframe(short_df, height=280)

    st.markdown("---")

    with st.expander("📋 Detailed Performance Table", expanded=False):
        perf_table = format_performance_table(perf)
        st.dataframe(
            perf_table.style.apply(
                lambda row: [
                    "font-weight: bold; background-color: #262730; color: #FAFAFA"
                    if row["Metric"] == ""
                    else ""
                    for _ in row
                ],
                axis=1,
            ),
            height=600,
        )

    with st.expander("🔧 Run Metadata", expanded=False):
        if data.meta:
            meta_display = {
                "Tag": data.tag,
                "Base Name": data.base_name,
                "Timestamp": data.meta.get("timestamp", "—"),
                "Factors": ", ".join(data.meta.get("factors", [])) if data.meta.get("factors") else "—",
                "TC Basis Points": data.meta.get("tc_bps", tc_bps),
            }
            st.json(meta_display)
        else:
            st.info("No metadata available")

elif page == "Performance":
    st.subheader("Performance Analytics")

    from q23.dashboard.analytics import select_return_series

    ret_series = select_return_series(data.diag)
    dd_series = drawdown(ret_series) if ret_series is not None else None

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

elif page == "Weights":
    st.subheader("Weights Explorer")

    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        n_days = st.number_input("Last N days", min_value=10, max_value=4000, value=252, step=10)
    with c2:
        topk = st.number_input("Top K assets", min_value=10, max_value=300, value=60, step=5)
    with c3:
        st.caption("Tip: keep Top K modest for performance")

    tail = data.weights.tail(int(n_days))
    latest = tail.iloc[-1]
    keep_cols = latest.abs().sort_values(ascending=False).head(int(topk)).index
    tail_small = tail[keep_cols]

    st.markdown("### Heatmap (tail × topK)")
    st.dataframe(tail_small, width="stretch", height=420)

    st.markdown("### Latest Snapshot (all assets)")
    snap = data.weights.loc[data.last_dt].sort_values()
    st.dataframe(snap.rename("weight").to_frame(), width="stretch", height=420)

elif page == "Factors/IC":
    st.subheader("Factors / IC")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Factor Weights (Time-Series)")
        if data.factor_weights is not None and not data.factor_weights.empty:
            top = (
                data.factor_weights.abs()
                .mean(axis=0)
                .sort_values(ascending=False)
                .head(12)
                .index.tolist()
            )
            sel = st.multiselect("Factors", options=list(data.factor_weights.columns), default=top)
            if sel:
                st.line_chart(data.factor_weights[sel], height=320)
        else:
            st.info("No factor_weights available")

    with c2:
        st.markdown("### Factor IC (Raw vs Smooth)")
        if data.ic is not None and not data.ic.empty:
            factors = sorted({c[0] for c in data.ic.columns})
            fsel = st.multiselect(
                "IC factors", options=factors, default=factors[: min(6, len(factors))]
            )
            if fsel:
                fig = create_factor_ic_chart(data.ic, fsel, title="Factor IC")
                st.pyplot(fig)
        else:
            st.info("No IC data available")

    st.divider()
    st.markdown("### Portfolio Factor Exposure")
    if data.exposure is not None and not data.exposure.empty:
        top = (
            data.exposure.abs().mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
        )
        sel = st.multiselect("Exposure factors", options=list(data.exposure.columns), default=top)
        if sel:
            st.line_chart(data.exposure[sel], height=320)
    else:
        st.info("No exposure data available")

elif page == "Attribution":
    st.subheader("Factor Attribution & Decomposition")

    from q23.dashboard.analytics import select_return_series

    ret_series = select_return_series(data.diag)

    if ret_series is None or ret_series.empty:
        st.warning("No return series available for attribution")
        st.stop()

    if data.exposure is None or data.exposure.empty:
        st.warning("No factor exposure available for attribution")
        st.stop()

    st.markdown("### Rolling Factor Regression")
    window = st.slider("Regression Window (days)", 60, 500, 252, 10)

    if "active_ret" in data.diag.columns:
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
                    regression_results[["factor_explained_ret", "idio_ret"]], height=300
                )
        else:
            st.info("Not enough data for rolling regression")
    else:
        st.info("No active return column in diagnostics")

    st.divider()
    st.markdown("### Factor vs Idiosyncratic Decomposition")

    if data.exposure is not None and ret_series is not None:
        # Compute factor returns proxy: change in exposures
        # Note: This is a proxy method. True factor returns would require
        # actual factor return data or regression-based estimates
        factor_returns = data.exposure.diff().fillna(0.0)
        
        # Validate that factor returns have the same columns as exposures
        if not factor_returns.columns.equals(data.exposure.columns):
            factor_returns = factor_returns.reindex(columns=data.exposure.columns, fill_value=0.0)
        
        try:
            decomp = compute_factor_vs_idio_decomposition(ret_series, data.exposure, factor_returns)

            if not decomp.empty:
                st.line_chart(decomp[["factor_ret_cum", "idio_ret_cum", "total_ret_cum"]], height=400)
            else:
                st.info("Could not compute decomposition - insufficient overlapping data")
        except Exception as e:
            st.error(f"Error computing decomposition: {str(e)}")
            st.info("Please check that return series and factor exposures have compatible time indices")

elif page == "Diagnostics":
    st.subheader("Portfolio Diagnostics")

    if data.weights is None or data.weights.empty:
        st.warning("No weights available")
        st.stop()

    st.markdown("### Hit Rate & Sizing Analysis")

    if data.diag is not None and not data.diag.empty:
        from q23.dashboard.analytics.metrics import turnover_series

        weights_df = data.weights
        returns_df = pd.DataFrame()

        if data.diag is not None and "port_ret" in data.diag.columns:
            returns_proxy = pd.DataFrame(
                np.random.randn(len(weights_df), weights_df.shape[1]) * 0.01,
                index=weights_df.index,
                columns=weights_df.columns,
            )
            returns_df = returns_proxy

        if not returns_df.empty:
            hit_rate_series = hit_rate(weights_df, returns_df)
            wl_sizing = winner_loser_sizing(weights_df, returns_df)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Hit Rate Over Time**")
                st.line_chart(hit_rate_series.tail(500), height=300)
            with c2:
                st.markdown("**Winner vs Loser Sizing**")
                st.line_chart(wl_sizing[["avg_winner_weight", "avg_loser_weight"]].tail(500), height=300)
        else:
            st.info("Returns data not available for hit rate analysis")

    st.divider()
    st.markdown("### Rolling Sharpe Ratio")

    from q23.dashboard.analytics import select_return_series

    ret_series = select_return_series(data.diag)
    if ret_series is not None and not ret_series.empty:
        sharpe_series = rolling_sharpe(ret_series, window=252)
        st.line_chart(sharpe_series.tail(500), height=300)
    else:
        st.info("No return series for Sharpe calculation")

elif page == "What-If":
    st.subheader("What-If Rebalance Engine")

    st.markdown(
        """
        Transform the current portfolio **without re-running the strategy**. 
        Apply controls (gross/net targets, seat selection, overlays, constraints) 
        and see metric deltas in real-time.
        """
    )

    baseline_w = data.w_last.copy()
    baseline_metrics = summary_exposure(baseline_w)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        gross_target = st.number_input("Gross Target", 0.1, 5.0, 1.0, 0.05)
    with col2:
        net_target_enabled = st.checkbox("Set Net Target", value=False)
        net_target = st.number_input("Net Target", -2.0, 2.0, 0.0, 0.05) if net_target_enabled else None
    with col3:
        tilt_alpha = st.slider("Tilt Alpha", 0.0, 2.0, 0.0, 0.05)

    col4, col5 = st.columns(2)
    with col4:
        topn_long_enabled = st.checkbox("Limit Long Seats", value=False)
        topn_long = st.number_input("Top N Long", 1, 100, 20, 1) if topn_long_enabled else None
    with col5:
        topn_short_enabled = st.checkbox("Limit Short Seats", value=False)
        topn_short = st.number_input("Top N Short", 1, 100, 10, 1) if topn_short_enabled else None

    col6, col7 = st.columns(2)
    with col6:
        max_pos_whatif = st.number_input("Max Position", 0.01, 0.50, 0.10, 0.01)
    with col7:
        min_pos_whatif = st.number_input("Min Position", 0.0, 0.05, 0.002, 0.001)

    st.markdown("#### Overlay / Blacklist / Pins")
    overlay_text = st.text_area(
        "Weight Overlay (symbol=weight, one per line)",
        value="",
        height=80,
        placeholder="AAPL=0.05\nMSFT=0.03",
    )
    blacklist_text = st.text_input("Blacklist (comma-separated)", value="")
    pins_text = st.text_input("Pin List (comma-separated)", value="")

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

    round_to_enabled = st.checkbox("Round weights", value=False)
    round_to = st.number_input("Round to", 0.001, 0.1, 0.005, 0.001) if round_to_enabled else None

    apply_whatif = st.button("🔄 Apply What-If Transform")

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

        st.success(f"✅ Transformed. Turnover: {result.turnover:.2%}")

        st.divider()
        st.markdown("### What Changed vs Baseline")

        delta_df = compute_metric_deltas(result, baseline_metrics)
        st.dataframe(delta_df, height=300)

        st.divider()
        st.markdown("### Weight Changes (Top 30 by |delta|)")
        delta_sorted = result.delta.abs().sort_values(ascending=False).head(30)
        change_df = pd.DataFrame(
            {
                "baseline": result.baseline_weights.reindex(delta_sorted.index).fillna(0.0),
                "whatif": result.transformed_weights.reindex(delta_sorted.index).fillna(0.0),
                "delta": result.delta.reindex(delta_sorted.index).fillna(0.0),
            }
        )
        st.dataframe(change_df, height=500)

        csv_bytes = change_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download What-If Trades CSV",
            data=csv_bytes,
            file_name=f"whatif_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

elif page == "Live Strategy":
    from q23.dashboard.pages._live_strategy import render_live_strategy_page
    
    # Use the stored base_dir for presets
    preset_dir = st.session_state.get("strategy_base_dir", base_dir)
    render_live_strategy_page(preset_dir)

elif page == "Blotter":
    st.subheader("Portfolio Blotter")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("As of", str(data.last_dt.date()))
    colB.metric("Positions", expos["n_pos"])
    colC.metric("Gross", f"{expos['gross']:.2f}")
    colD.metric("Net", f"{expos['net']:.2f}")

    st.divider()

    blot = pd.DataFrame({"weight": data.w_last})
    blot = blot[blot["weight"].abs() > 1e-12].copy()
    blot["side"] = np.where(blot["weight"] > 0, "LONG", "SHORT")
    blot["abs_w"] = blot["weight"].abs()

    if data.factor_vectors is not None and not data.factor_vectors.empty:
        join_cols = [
            c
            for c in [
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

    blot = blot.sort_values(["side", "abs_w"], ascending=[True, False]).drop(columns=["abs_w"])
    st.markdown("### Positions (with PM summaries)")
    st.dataframe(blot, width="stretch", height=640)

    st.divider()
    st.markdown("### Single-Stock Drilldown")
    if data.factor_vectors is None or data.factor_vectors.empty:
        st.info("factor_vectors not available")
    else:
        symbols = sorted(set(blot.index).intersection(set(data.factor_vectors.index)))
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
            fac_cols = [c for c in data.factor_vectors.columns if c not in summary_cols]

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
                st.dataframe(expos_vec.to_frame("exposure"), width="stretch", height=320)
        else:
            st.info("No matching symbols in factor_vectors")

elif page == "Rebalance":
    st.subheader("Rebalance Trade List")

    dates = list(data.weights.index)
    if len(dates) < 2:
        st.warning("Need at least 2 dates")
        st.stop()

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

    trades = compute_trade_list(
        data.weights, pd.Timestamp(t0), pd.Timestamp(t1), min_abs_delta=float(min_trade)
    )
    turnover = float(np.abs(trades["delta_w"]).sum())
    st.metric("Turnover Σ|Δw| (filtered)", f"{turnover:.2%}")

    st.dataframe(trades.reset_index(), width="stretch", height=560)

    csv_bytes = trades.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trades CSV",
        data=csv_bytes,
        file_name=f"{base_name}_rebalance_{pd.Timestamp(t0).strftime('%Y%m%d')}_vs_{pd.Timestamp(t1).strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

elif page == "Strategy Comparison":
    from q23.dashboard.pages._strategy_comparison import render_strategy_comparison
    render_strategy_comparison(
        current_strategy=selected_strategy if use_multi_strategy else "",
        date_range=date_range,
    )

if elite_page != "None":
    from q23.dashboard.pages._elite_analytics import (
        render_regime_analysis_page,
        render_tail_risk_page,
        render_convexity_page,
        render_alpha_decay_page,
        render_capacity_page,
        render_ab_testing_page,
        render_factor_timing_page,
        render_rotation_page,
        render_rank_ic_page,
    )
    from q23.dashboard.pages._risk_analytics import (
        render_risk_attribution_page,
        render_brinson_attribution_page,
        render_ex_ante_risk_page,
        render_sector_analysis_page,
        render_correlation_analysis_page,
        render_beta_analysis_page,
        render_pm_executive_summary,
    )

    # Executive Summary (Top priority)
    if elite_page == "📋 Executive Summary":
        render_pm_executive_summary(data)
    # Risk Analytics Pages
    elif elite_page == "Risk Attribution":
        render_risk_attribution_page(data)
    elif elite_page == "Brinson Attribution":
        render_brinson_attribution_page(data)
    elif elite_page == "Ex-Ante Risk":
        render_ex_ante_risk_page(data)
    elif elite_page == "Sector Analysis":
        render_sector_analysis_page(data)
    elif elite_page == "Correlation Analysis":
        render_correlation_analysis_page(data)
    elif elite_page == "Beta Analysis":
        render_beta_analysis_page(data)
    # Original Elite Pages
    elif elite_page == "Regime Analysis":
        render_regime_analysis_page(data)
    elif elite_page == "Tail Risk & Stress":
        render_tail_risk_page(data)
    elif elite_page == "Convexity & Gamma":
        render_convexity_page(data)
    elif elite_page == "Alpha Decay":
        render_alpha_decay_page(data)
    elif elite_page == "Capacity Estimation":
        render_capacity_page(data)
    elif elite_page == "A/B Testing":
        render_ab_testing_page(data)
    elif elite_page == "Factor Timing":
        render_factor_timing_page(data)
    elif elite_page == "Rotation Velocity":
        render_rotation_page(data)
    elif elite_page == "Rank IC & Quintiles":
        render_rank_ic_page(data)

st.divider()
st.caption("Q23 ELITE Dashboard — Multi-Strategy + What-If + Advanced Visualizations + Elite Analytics")
