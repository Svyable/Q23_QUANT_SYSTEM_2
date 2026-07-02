"""
Rebalance Page - Enhanced for PM Decision Making

Trade list generation for rebalancing with:
- Enhanced aesthetics and color gradients
- PM analytics tooltips for real-time stat arb decisions
- Rich hover information for trade execution
- Forward return expectations (T+1, T+5, T+21)
- Optimized layout for quick decision making
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import compute_trade_list, select_return_series
from q23.dashboard.analytics.stock_elite_analytics import (
    compute_batch_forward_returns,
    format_forward_return_cell,
)
from q23.dashboard.components.charts import PLOTLY_AVAILABLE, get_plotly_config

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def _enhance_trades_with_pm_analytics(
    trades: pd.DataFrame,
    factor_vectors: Optional[pd.DataFrame],
    weights: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    portfolio_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Enhance trade list with PM analytics for decision making.
    
    Args:
        trades: Trade list DataFrame with prev_w, target_w, delta_w, action
        factor_vectors: Optional factor vectors with PM metrics
        weights: Full weights DataFrame
        t0: Target date
        t1: Previous date
        portfolio_returns: Optional portfolio returns for forward return estimation
    
    Returns:
        Enhanced DataFrame with PM analytics columns including forward returns
    """
    enhanced = trades.copy()
    
    # Add absolute delta for sorting/coloring
    enhanced['abs_delta'] = enhanced['delta_w'].abs()
    
    # Add direction indicators
    enhanced['direction'] = enhanced['delta_w'].apply(
        lambda x: 'LONG ↑' if x > 0 else 'SHORT ↓' if x < 0 else 'FLAT'
    )
    
    # Join PM analytics from factor_vectors if available
    if factor_vectors is not None and not factor_vectors.empty:
        pm_cols = [
            'total_pnl_contrib',
            'pnl_per_day_held',
            'mean_score',
            'score_vol',
            'days_held',
            'avg_weight',
            'avg_weight_when_held',
        ]
        
        available_pm_cols = [c for c in pm_cols if c in factor_vectors.columns]
        
        if available_pm_cols:
            # Join on symbol (index)
            enhanced = enhanced.join(
                factor_vectors[available_pm_cols],
                how='left'
            )
    
    # Add current weight context
    if t0 in weights.index:
        current_weights = weights.loc[t0]
        enhanced = enhanced.join(
            current_weights.rename('current_weight'),
            how='left'
        ).fillna(0.0)
    
    # Compute trade size vs current position
    if 'current_weight' in enhanced.columns:
        enhanced['trade_to_position_ratio'] = (
            enhanced['abs_delta'] / (enhanced['current_weight'].abs() + 1e-6)
        ).replace([np.inf, -np.inf], np.nan)
    
    # Add trade classification
    def classify_trade(row):
        if pd.isna(row.get('prev_w', 0)) or abs(row.get('prev_w', 0)) < 1e-6:
            return 'NEW'
        elif abs(row.get('target_w', 0)) < 1e-6:
            return 'EXIT'
        elif (row.get('prev_w', 0) > 0 and row.get('target_w', 0) < 0) or \
             (row.get('prev_w', 0) < 0 and row.get('target_w', 0) > 0):
            return 'FLIP'
        else:
            return 'SIZE'
    
    enhanced['trade_type'] = enhanced.apply(classify_trade, axis=1)
    
    # ==========================================================================
    # FORWARD RETURN EXPECTATIONS
    # ==========================================================================
    fwd_returns = _compute_forward_returns_for_trades(
        enhanced.index.tolist(),
        factor_vectors,
        weights,
        portfolio_returns,
    )
    
    if fwd_returns is not None and not fwd_returns.empty:
        enhanced = enhanced.join(fwd_returns, how='left')
    
    return enhanced


def _compute_forward_returns_for_trades(
    symbols: list,
    factor_vectors: Optional[pd.DataFrame],
    weights: Optional[pd.DataFrame],
    portfolio_returns: Optional[pd.Series],
) -> Optional[pd.DataFrame]:
    """
    Compute forward return expectations for trade list.
    
    Returns DataFrame with:
    - E[T+1], E[T+5], E[T+21]: Formatted expected returns with ranges
    - exp_t1, exp_t5, exp_t21: Raw expected values (for calculations)
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
        
        # Create result with both raw and formatted columns
        result = pd.DataFrame(index=fwd_df.index)
        
        # Keep raw values for calculations
        for h in [1, 5, 21]:
            exp_col = f"exp_t{h}"
            if exp_col in fwd_df.columns:
                result[exp_col] = fwd_df[exp_col]
        
        # Create formatted display columns
        for h in [1, 5, 21]:
            exp_col = f"exp_t{h}"
            upper_col = f"upper_t{h}"
            lower_col = f"lower_t{h}"
            
            if exp_col in fwd_df.columns:
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
        return None


def _style_trades_table(
    trades_df: pd.DataFrame,
) -> pd.Styler:
    """
    Apply professional styling to trades table with color gradients.
    
    Args:
        trades_df: Enhanced trades DataFrame (may have renamed columns)
    
    Returns:
        Styled DataFrame
    """
    # Check if columns are renamed (display version) or original
    delta_col = 'Δ Weight' if 'Δ Weight' in trades_df.columns else 'delta_w'
    type_col = 'Type' if 'Type' in trades_df.columns else 'trade_type'
    pnl_col = 'Total P&L' if 'Total P&L' in trades_df.columns else 'total_pnl_contrib'
    
    # Start with basic styling
    styled = trades_df.style
    
    # Format numeric columns
    format_dict = {}
    col_mapping = {
        'Previous Weight': ('prev_w', '{:.2%}'),
        'Target Weight': ('target_w', '{:.2%}'),
        'Δ Weight': ('delta_w', '{:.2%}'),
        'Current Weight': ('current_weight', '{:.2%}'),
        'Total P&L': ('total_pnl_contrib', '{:.4f}'),
        'P&L/Day': ('pnl_per_day_held', '{:.6f}'),
        'Mean Score': ('mean_score', '{:.3f}'),
        'Trade/Pos Ratio': ('trade_to_position_ratio', '{:.1%}'),
    }
    
    for display_col, (orig_col, fmt) in col_mapping.items():
        if display_col in trades_df.columns:
            format_dict[display_col] = fmt
        elif orig_col in trades_df.columns:
            format_dict[orig_col] = fmt
    
    if format_dict:
        styled = styled.format(format_dict)
    
    # Apply color gradients to delta_w column (buy/sell intensity)
    if delta_col in trades_df.columns:
        def apply_delta_gradient(val):
            try:
                delta = float(str(val).replace('%', '').replace(',', ''))
                if pd.isna(delta):
                    return 'background-color: #000000; color: #888888'
                
                abs_delta = abs(delta)
                # Normalize to 0-1 based on max absolute delta in column
                max_abs = trades_df[delta_col].abs().max()
                if max_abs > 0:
                    norm_val = abs_delta / max_abs
                else:
                    norm_val = 0.0
                
                if delta > 0:
                    # Long: green gradient
                    if norm_val < 0.25:
                        color = '#145a32'
                    elif norm_val < 0.5:
                        color = '#1e8449'
                    elif norm_val < 0.75:
                        color = '#27ae60'
                    else:
                        color = '#2ecc71'
                    text_color = '#ecf0f1' if norm_val < 0.5 else '#000000'
                else:
                    # Short: red gradient
                    if norm_val < 0.25:
                        color = '#7b241c'
                    elif norm_val < 0.5:
                        color = '#a93226'
                    elif norm_val < 0.75:
                        color = '#c0392b'
                    else:
                        color = '#e74c3c'
                    text_color = '#ecf0f1' if norm_val < 0.5 else '#ffffff'
                
                return f'background-color: {color}; color: {text_color}; font-weight: 600;'
            except (ValueError, TypeError):
                return ''
        
        styled = styled.map(apply_delta_gradient, subset=[delta_col])
    
    # Apply color to trade type
    if type_col in trades_df.columns:
        def color_trade_type(val):
            if not isinstance(val, str):
                return ''
            if val == 'NEW':
                return 'background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; font-weight: 600;'
            elif val == 'EXIT':
                return 'background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; font-weight: 600;'
            elif val == 'FLIP':
                return 'background-color: rgba(243, 156, 18, 0.2); color: #f39c12; font-weight: 600;'
            elif val == 'SIZE':
                return 'background-color: rgba(52, 152, 219, 0.2); color: #3498db; font-weight: 600;'
            return ''
        
        styled = styled.map(color_trade_type, subset=[type_col])
    
    # Highlight high P&L contributors if available
    if pnl_col in trades_df.columns:
        def highlight_pnl(val):
            try:
                pnl_str = str(val).replace(',', '')
                pnl = float(pnl_str)
                if pd.isna(pnl):
                    return ''
                if pnl > 0.01:  # High positive P&L
                    return 'background-color: rgba(46, 204, 113, 0.15); font-weight: 600; color: #2ecc71;'
                elif pnl < -0.01:  # High negative P&L
                    return 'background-color: rgba(231, 76, 60, 0.15); font-weight: 600; color: #e74c3c;'
            except (ValueError, TypeError):
                pass
            return ''
        
        styled = styled.map(highlight_pnl, subset=[pnl_col])
    
    return styled


def _compute_expected_rebalance_pnl(enhanced_trades: pd.DataFrame) -> float:
    """
    Compute expected P&L from rebalance at T+5 horizon.
    
    Expected P&L = Σ (delta_w × exp_t5)
    
    This represents the expected contribution from the rebalancing trades,
    weighting each trade's expected return by its position change.
    
    Args:
        enhanced_trades: Enhanced trades DataFrame with exp_t5 column
        
    Returns:
        Expected P&L contribution from rebalance
    """
    if enhanced_trades.empty:
        return 0.0
    
    if 'exp_t5' not in enhanced_trades.columns:
        return 0.0
    
    if 'delta_w' not in enhanced_trades.columns:
        return 0.0
    
    try:
        # Expected P&L = sum(delta_weight × expected_return)
        # We use absolute delta_w because the return already has direction encoded
        # Actually, we want: if we're buying (delta_w > 0) and expected return is positive,
        # that's positive P&L. If we're selling (delta_w < 0) and expected return is positive,
        # we're giving up positive P&L.
        # So P&L = delta_w × exp_return for each position
        
        valid_mask = (
            enhanced_trades['exp_t5'].notna() & 
            enhanced_trades['delta_w'].notna()
        )
        
        if not valid_mask.any():
            return 0.0
        
        valid = enhanced_trades.loc[valid_mask]
        
        # For each trade: contribution = position_change × expected_return_of_target_position
        # If buying (delta > 0), we expect to capture the expected return
        # If selling (delta < 0), we're avoiding the expected return
        expected_pnl = (valid['delta_w'] * valid['exp_t5']).sum()
        
        return float(expected_pnl)
        
    except Exception:
        return 0.0


def render_rebalance_page(data: DashboardData, base_name: str) -> None:
    """
    Render the enhanced Rebalance Trade List page for PM decision making.
    
    Args:
        data: Dashboard data bundle
        base_name: Base name for export filenames
    """
    # Inject CSS for section titles
    st.markdown("""
    <style>
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ecf0f1;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #2d2d2d;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .section-title .icon {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
        '<div class="section-title"><span class="icon">🔄</span> Rebalance Trade List</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **Real-time trade list generation for rebalancing decisions.**  
    Enhanced with PM analytics, color-coded gradients, and rich tooltips for stat arb decision making.
    """)
    
    dates = list(data.weights.index)
    if len(dates) < 2:
        st.warning("Need at least 2 dates for rebalancing")
        st.stop()
    
    # ==========================================================================
    # HEADER WITH DOWNLOAD BUTTON (moved higher)
    # ==========================================================================
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col2:
        # Placeholder - will be populated after trades computed
        download_placeholder = st.empty()
    
    # ==========================================================================
    # DATE SELECTORS
    # ==========================================================================
    st.markdown("### 📅 Date Selection")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        t0 = st.selectbox(
            "T0 (Target Date)",
            options=dates,
            index=len(dates) - 1,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            help="Target date for rebalancing - the position weights we want to achieve",
        )
    with c2:
        i0 = dates.index(t0)
        t1 = st.selectbox(
            "T-1 (Previous Date)",
            options=dates,
            index=max(0, i0 - 1),
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            help="Previous date to compare against - the current position weights",
        )
    with c3:
        min_trade = st.slider(
            "Min |Δw| Filter",
            0.0,
            0.05,
            0.002,
            step=0.0005,
            help="Minimum absolute weight change to include in trade list",
        )
    
    if t1 >= t0:
        st.warning("⚠️ Pick T-1 strictly earlier than T0")
        st.stop()
    
    # ==========================================================================
    # COMPUTE TRADES
    # ==========================================================================
    trades = compute_trade_list(
        data.weights,
        pd.Timestamp(t0),
        pd.Timestamp(t1),
        min_abs_delta=float(min_trade)
    )
    
    if trades.empty:
        st.info(f"No trades above threshold {min_trade:.4f} between {t1.strftime('%Y-%m-%d')} and {t0.strftime('%Y-%m-%d')}")
        st.stop()
    
    # Enhance with PM analytics and forward returns
    portfolio_returns = select_return_series(data.diag)
    enhanced_trades = _enhance_trades_with_pm_analytics(
        trades,
        data.factor_vectors,
        data.weights,
        pd.Timestamp(t0),
        pd.Timestamp(t1),
        portfolio_returns,
    )
    
    # Sort by absolute delta (largest trades first)
    enhanced_trades = enhanced_trades.sort_values('abs_delta', ascending=False)
    
    # ==========================================================================
    # SUMMARY METRICS
    # ==========================================================================
    turnover = float(np.abs(enhanced_trades["delta_w"]).sum())
    n_trades = len(enhanced_trades)
    buys = len(enhanced_trades[enhanced_trades['delta_w'] > 0])
    sells = len(enhanced_trades[enhanced_trades['delta_w'] < 0])
    
    # Update download button now that we have trades
    with download_placeholder:
        csv_bytes = enhanced_trades.reset_index().to_csv(index=False).encode("utf-8")
        t0_str = pd.Timestamp(t0).strftime('%Y%m%d')
        t1_str = pd.Timestamp(t1).strftime('%Y%m%d')
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name=f"{base_name}_rebalance_{t0_str}_vs_{t1_str}.csv",
            mime="text/csv",
            help="Download full trade list with PM analytics",
            width='stretch',
        )
    
    # Display summary metrics
    st.markdown("### 📊 Trade Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Turnover", f"{turnover:.2%}")
    with col2:
        st.metric("Number of Trades", n_trades)
    with col3:
        st.metric("Buys", buys, delta=f"{buys-n_trades//2}" if n_trades > 0 else None)
    with col4:
        st.metric("Sells", sells, delta=f"{sells-n_trades//2}" if n_trades > 0 else None)
    with col5:
        avg_trade_size = turnover / n_trades if n_trades > 0 else 0
        st.metric("Avg Trade Size", f"{avg_trade_size:.3%}")
    with col6:
        # Compute expected P&L from rebalance (using T+5 horizon)
        exp_pnl = _compute_expected_rebalance_pnl(enhanced_trades)
        st.metric(
            "E[P&L T+5]",
            f"{exp_pnl:+.4f}" if exp_pnl != 0 else "N/A",
            help="Expected P&L at T+5 horizon based on position changes and factor exposures"
        )
    
    st.divider()
    
    # ==========================================================================
    # TRADE TYPE BREAKDOWN
    # ==========================================================================
    if 'trade_type' in enhanced_trades.columns:
        trade_type_counts = enhanced_trades['trade_type'].value_counts()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📋 Trade Breakdown")
            
            # Create breakdown display
            breakdown_html = ""
            type_colors = {
                'NEW': '#2ecc71',
                'EXIT': '#e74c3c',
                'FLIP': '#f39c12',
                'SIZE': '#3498db',
            }
            
            breakdown_items = []
            for trade_type, count in trade_type_counts.items():
                color = type_colors.get(trade_type, '#95a5a6')
                pct = (count / n_trades * 100) if n_trades > 0 else 0
                breakdown_items.append(
                    f'<span style="color: {color}; font-weight: 600;">{trade_type}:</span> '
                    f'{count} ({pct:.1f}%)'
                )
            
            st.markdown(" | ".join(breakdown_items), unsafe_allow_html=True)
        
        with col2:
            # Trade type distribution chart if Plotly available
            if PLOTLY_AVAILABLE and go is not None:
                fig = go.Figure(data=[go.Pie(
                    labels=trade_type_counts.index.tolist(),
                    values=trade_type_counts.values.tolist(),
                    marker_colors=[type_colors.get(t, '#95a5a6') for t in trade_type_counts.index],
                    textinfo='label+percent',
                    hole=0.4,
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                )])
                
                fig.update_layout(
                    title={'text': 'Trade Types', 'font': {'size': 12, 'color': '#ecf0f1'}},
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#262730',
                    font={'color': '#ecf0f1', 'size': 10},
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                
                config = get_plotly_config()
                st.plotly_chart(fig, config=config, width='stretch')
    
    st.divider()
    
    # ==========================================================================
    # FORWARD RETURN METHODOLOGY
    # ==========================================================================
    with st.expander("📊 Forward Return Methodology", expanded=False):
        st.markdown("""
        **Expected Forward Returns** are computed using Factor IC-based estimation:
        
        - `E[R]` = Σ (factor_exposure × IC) × √horizon
        - Range = E[R] ± σ_stock × √horizon
        
        **Columns:**
        - `E[T+1]`: Expected 1-day return with ±1σ range
        - `E[T+5]`: Expected 5-day return with ±1σ range
        - `E[T+21]`: Expected 21-day return with ±1σ range
        
        **E[P&L T+5]** in summary = Σ (Δ weight × E[T+5]) across all trades
        
        *Note: These are estimates based on historical factor exposures and may not predict actual future returns.*
        """)
    
    # ==========================================================================
    # MAIN TRADE TABLE
    # ==========================================================================
    st.markdown("### 💼 Trade List (sorted by size)")
    
    # Prepare display columns - prioritize key columns
    display_cols = ['symbol']
    
    # Add key trade columns
    if 'direction' in enhanced_trades.columns:
        display_cols.append('direction')
    if 'trade_type' in enhanced_trades.columns:
        display_cols.append('trade_type')
    display_cols.extend(['prev_w', 'target_w', 'delta_w'])
    
    # Add PM analytics columns if available
    if 'total_pnl_contrib' in enhanced_trades.columns:
        display_cols.append('total_pnl_contrib')
    if 'mean_score' in enhanced_trades.columns:
        display_cols.append('mean_score')
    
    # Add forward return expectations columns
    for h in [1, 5, 21]:
        col_name = f"E[T+{h}]"
        if col_name in enhanced_trades.columns:
            display_cols.append(col_name)
    
    # Add additional PM columns
    if 'days_held' in enhanced_trades.columns:
        display_cols.append('days_held')
    if 'current_weight' in enhanced_trades.columns:
        display_cols.append('current_weight')
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in enhanced_trades.columns]
    
    display_trades = enhanced_trades[display_cols].copy()
    
    # Rename columns for better display
    column_rename = {
        'prev_w': 'Previous Weight',
        'target_w': 'Target Weight',
        'delta_w': 'Δ Weight',
        'total_pnl_contrib': 'Total P&L',
        'pnl_per_day_held': 'P&L/Day',
        'mean_score': 'Mean Score',
        'days_held': 'Days Held',
        'current_weight': 'Current Weight',
        'trade_to_position_ratio': 'Trade/Pos Ratio',
        'direction': 'Direction',
        'trade_type': 'Type',
        'symbol': 'Symbol',
    }
    
    display_trades = display_trades.rename(columns=column_rename)
    
    # Apply styling
    styled_trades = _style_trades_table(display_trades.reset_index())
    
    # Display with enhanced styling
    st.dataframe(
        styled_trades,
        height=600,
        width='stretch',
    )
    
    # ==========================================================================
    # VISUALIZATION: TRADE SIZE DISTRIBUTION
    # ==========================================================================
    if PLOTLY_AVAILABLE and go is not None:
        st.divider()
        st.markdown("### 📈 Trade Size Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of top trades
            top_20 = enhanced_trades.head(20)
            colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in top_20['delta_w']]
            
            fig = go.Figure(data=[go.Bar(
                x=top_20.index.tolist(),
                y=top_20['delta_w'].values * 100,
                marker_color=colors,
                text=[f"{d:.2f}%" for d in top_20['delta_w'].values * 100],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Δ Weight: %{y:.2f}%<extra></extra>',
            )])
            
            fig.update_layout(
                title={'text': 'Top 20 Trades by Size', 'font': {'size': 12, 'color': '#ecf0f1'}},
                xaxis_title='Symbol',
                yaxis_title='Δ Weight (%)',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#262730',
                font={'color': '#ecf0f1', 'size': 10},
                height=350,
                autosize=True,
                margin=dict(l=60, r=40, t=60, b=100),
                xaxis={'tickangle': -45, 'gridcolor': '#3A3A3A'},
                yaxis={'gridcolor': '#3A3A3A'},
            )
            
            config = get_plotly_config()
            st.plotly_chart(fig, config=config, width='stretch')
        
        with col2:
            # Distribution of trade sizes
            fig = go.Figure(data=[go.Histogram(
                x=enhanced_trades['delta_w'].abs().values * 100,
                nbinsx=30,
                marker_color='#3498db',
                hovertemplate='Trade Size: %{x:.2f}%<br>Count: %{y}<extra></extra>',
            )])
            
            fig.update_layout(
                title={'text': 'Trade Size Distribution', 'font': {'size': 12, 'color': '#ecf0f1'}},
                xaxis_title='|Δ Weight| (%)',
                yaxis_title='Count',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#262730',
                font={'color': '#ecf0f1', 'size': 10},
                height=350,
                autosize=True,
                margin=dict(l=60, r=40, t=60, b=60),
                xaxis={'gridcolor': '#3A3A3A'},
                yaxis={'gridcolor': '#3A3A3A'},
            )
            
            config = get_plotly_config()
            st.plotly_chart(fig, config=config, width='stretch')
    
    # ==========================================================================
    # PM INSIGHTS SECTION
    # ==========================================================================
    if data.factor_vectors is not None and 'total_pnl_contrib' in enhanced_trades.columns:
        st.divider()
        st.markdown("### 💡 PM Insights")
        
        # Find top winners/losers in trade list
        trades_with_pnl = enhanced_trades[
            enhanced_trades['total_pnl_contrib'].notna()
        ].copy()
        
        if not trades_with_pnl.empty:
            top_winners = trades_with_pnl.nlargest(5, 'total_pnl_contrib')
            top_losers = trades_with_pnl.nsmallest(5, 'total_pnl_contrib')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Top P&L Contributors (in trade list)**")
                # Use itertuples() instead of iterrows() for better performance
                for row in top_winners.itertuples():
                    symbol = row.Index  # Index becomes Index attribute
                    pnl = row.total_pnl_contrib
                    delta = row.delta_w
                    direction = "↑ BUY" if delta > 0 else "↓ SELL"
                    st.markdown(
                        f"**{symbol}** {direction} | "
                        f"P&L: {pnl:+.4f} | "
                        f"Δw: {delta:+.2%}"
                    )
            
            with col2:
                st.markdown("**⚠️ Top P&L Draggers (in trade list)**")
                # Use itertuples() instead of iterrows() for better performance
                for row in top_losers.itertuples():
                    symbol = row.Index  # Index becomes Index attribute
                    pnl = row.total_pnl_contrib
                    delta = row.delta_w
                    direction = "↑ BUY" if delta > 0 else "↓ SELL"
                    st.markdown(
                        f"**{symbol}** {direction} | "
                        f"P&L: {pnl:+.4f} | "
                        f"Δw: {delta:+.2%}"
                    )
