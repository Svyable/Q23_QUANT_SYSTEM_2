"""
Calendar Heatmap Page

Enhanced calendar heatmap visualization for PM use with:
- True calendar layout visualization
- Pattern detection (day-of-week, month effects)
- Statistical analysis by period
- Anomaly detection
- Year-over-year comparisons
- Alpha-generating insights
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import calendar

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import select_return_series
from q23.dashboard.components.charts import (
    PLOTLY_AVAILABLE,
    PM_COLORS,
    PM_DIVERGING,
    get_plotly_config,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None


def render_calendar_heatmap_page(data: DashboardData) -> None:
    """Render the enhanced calendar heatmap page."""
    st.subheader("📅 Calendar Heatmap & Pattern Analysis")
    
    ret_series = select_return_series(data.diag)
    
    if ret_series is None or ret_series.empty:
        st.warning("No return data available for calendar heatmap")
        return
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(ret_series.index.year.unique())
        selected_years = st.multiselect(
            "📅 Select Years",
            options=years,
            default=years[-2:] if len(years) >= 2 else years[-1:],
            help="Choose which years to include in the calendar analysis. Select multiple for year-over-year comparison."
        )

    with col2:
        view_mode = st.selectbox(
            "🎨 View Mode",
            options=["Calendar Grid", "Week Heatmap", "Month Comparison", "Year Comparison"],
            help="Calendar Grid: Traditional calendar layout. Week Heatmap: Day-of-week patterns. Month Comparison: Monthly averages. Year Comparison: Year-over-year trends."
        )

    with col3:
        color_mode = st.selectbox(
            "🎯 Color Scheme",
            options=["Return Magnitude", "Return Direction", "Volatility Adjusted"],
            help="Return Magnitude: Absolute return values. Return Direction: Green/red for gains/losses. Volatility Adjusted: Returns normalized by volatility."
        )
    
    if not selected_years:
        st.info("Select at least one year to display")
        return
    
    # Filter data for selected years
    filtered_data = ret_series[ret_series.index.year.isin(selected_years)]
    
    if len(filtered_data) < 20:
        st.warning("Not enough data for selected period")
        return
    
    # Pattern Analysis Section
    with st.expander("🔍 Pattern Analysis & Insights", expanded=True):
        _render_pattern_analysis(filtered_data)
    
    st.divider()
    
    # Main Visualization
    if view_mode == "Calendar Grid":
        _render_calendar_grid(filtered_data, selected_years, color_mode)
    elif view_mode == "Week Heatmap":
        _render_week_heatmap(filtered_data, color_mode)
    elif view_mode == "Month Comparison":
        _render_month_comparison(filtered_data, selected_years)
    elif view_mode == "Year Comparison":
        _render_year_comparison(filtered_data, selected_years)
    
    st.divider()
    
    # Statistical Breakdown
    with st.expander("📊 Statistical Breakdown by Period", expanded=False):
        _render_statistical_breakdown(filtered_data)
    
    # Anomaly Detection
    with st.expander("🚨 Anomaly Detection", expanded=False):
        _render_anomaly_detection(filtered_data)


def _render_pattern_analysis(ret_series: pd.Series) -> None:
    """Render pattern analysis and insights."""
    insights = []
    
    # Day of week analysis
    ret_series.index = pd.to_datetime(ret_series.index)
    df = pd.DataFrame({'return': ret_series})
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    
    # Day of week patterns
    dow_avg = df.groupby('weekday')['return'].agg(['mean', 'std', 'count'])
    dow_avg.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    dow_avg.columns = ['Avg Return', 'Std Dev', 'Count']
    
    best_day = dow_avg['Avg Return'].idxmax()
    worst_day = dow_avg['Avg Return'].idxmin()
    best_day_ret = dow_avg.loc[best_day, 'Avg Return']
    worst_day_ret = dow_avg.loc[worst_day, 'Avg Return']
    
    if best_day_ret > 0 and abs(best_day_ret) > abs(worst_day_ret):
        insights.append(f"✅ **Strong {best_day} Effect**: Average return of {best_day_ret:.2%}")
    
    # Month effects
    month_avg = df.groupby('month')['return'].agg(['mean', 'std'])
    month_avg.index = [calendar.month_abbr[i] for i in month_avg.index]
    month_avg.columns = ['Avg Return', 'Std Dev']
    
    best_month = month_avg['Avg Return'].idxmax()
    worst_month = month_avg['Avg Return'].idxmin()
    
    # Day of month effects (first half vs second half)
    first_half = df[df['day_of_month'] <= 15]['return']
    second_half = df[df['day_of_month'] > 15]['return']
    
    first_half_avg = first_half.mean()
    second_half_avg = second_half.mean()
    
    if abs(first_half_avg - second_half_avg) > 0.001:  # 10 bps difference
        if first_half_avg > second_half_avg:
            insights.append(f"📈 **Month-Begin Effect**: First half avg {first_half_avg:.2%} vs second half {second_half_avg:.2%}")
        else:
            insights.append(f"📉 **Month-End Effect**: Second half avg {second_half_avg:.2%} vs first half {first_half_avg:.2%}")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("No significant patterns detected in the data")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Day", best_day, f"{best_day_ret:.2%}")
        st.metric("Worst Day", worst_day, f"{worst_day_ret:.2%}")
    
    with col2:
        st.metric("Best Month", best_month, f"{month_avg.loc[best_month, 'Avg Return']:.2%}")
        st.metric("Worst Month", worst_month, f"{month_avg.loc[worst_month, 'Avg Return']:.2%}")
    
    with col3:
        st.metric("First Half Avg", f"{first_half_avg:.2%}")
        st.metric("Second Half Avg", f"{second_half_avg:.2%}")


def _render_calendar_grid(ret_series: pd.Series, years: List[int], color_mode: str) -> None:
    """Render true calendar grid layout."""
    if not PLOTLY_AVAILABLE or go is None:
        st.info("Plotly required for calendar grid visualization")
        return
    
    n_years = len(years)
    n_cols = min(3, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{year}" for year in years],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # Determine color scale based on mode - use centralized PM colors
    if color_mode == "Return Direction":
        colorscale = PM_DIVERGING
        zmid = 0
        vmin = None
        vmax = None
    elif color_mode == "Volatility Adjusted":
        # Normalize by volatility - compute z-scores for each year separately
        colorscale = PM_DIVERGING
        zmid = 0
        vmin = None  # Will be computed per year
        vmax = None
    else:  # Return Magnitude
        # Use quantiles for magnitude-based coloring
        vmin, vmax = ret_series.quantile([0.05, 0.95]).values
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        colorscale = PM_DIVERGING
        zmid = 0
    
    ret_series.index = pd.to_datetime(ret_series.index)
    
    for idx, year in enumerate(years):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        year_data = ret_series[ret_series.index.year == year]
        
        if len(year_data) == 0:
            continue
        
        # Create calendar grid: 12 months x 31 days
        cal_grid = np.full((12, 31), np.nan)
        
        # Handle volatility-adjusted mode
        if color_mode == "Volatility Adjusted":
            # Normalize by rolling volatility
            rolling_std = year_data.rolling(window=min(20, len(year_data))).std()
            rolling_std = rolling_std.replace(0, np.nan)
            z_scores = year_data / rolling_std
            z_scores = z_scores.fillna(0)
            # Use z-scores for coloring
            for date, z_val in z_scores.items():
                month_idx = date.month - 1
                day_idx = date.day - 1
                cal_grid[month_idx, day_idx] = z_val
        else:
            # Use raw returns
            for date, ret_val in year_data.items():
                month_idx = date.month - 1
                day_idx = date.day - 1
                cal_grid[month_idx, day_idx] = ret_val
        
        # Month labels
        month_names = [calendar.month_abbr[i+1] for i in range(12)]
        
        # Scale factor - use 100 for returns, 1 for z-scores
        scale_factor = 100 if color_mode != "Volatility Adjusted" else 1
        
        fig.add_trace(
            go.Heatmap(
                z=cal_grid * scale_factor,
                x=list(range(1, 32)),
                y=month_names,
                colorscale=colorscale,
                zmid=zmid,
                text=[[f"{cal_grid[i, j]*scale_factor:.2f}{'%' if color_mode != 'Volatility Adjusted' else ''}" 
                       if not np.isnan(cal_grid[i, j]) else "" 
                       for j in range(31)] for i in range(12)],
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate=f"Month: %{{y}}<br>Day: %{{x}}<br>{'Return' if color_mode != 'Volatility Adjusted' else 'Z-Score'}: %{{z:.2f}}{'%' if color_mode != 'Volatility Adjusted' else ''}<extra></extra>",
                colorbar=dict(
                    title=dict(
                        text="Return %" if color_mode != "Volatility Adjusted" else "Z-Score",
                        font=dict(size=11)
                    ),
                    x=1.02,
                    xref="paper",
                    len=0.6,
                    thickness=15,
                ) if idx == 0 else None,
            ),
            row=row,
            col=col,
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Day of Month",
            row=row,
            col=col,
            tickmode='linear',
            tick0=1,
            dtick=5,
        )
        fig.update_yaxes(
            title_text="",
            row=row,
            col=col,
            autorange='reversed',
        )
    
    fig.update_layout(
        title_text="Calendar Heatmap: Daily Returns by Year",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 11},
        showlegend=False,
        autosize=True,
        # Optimized margins to prevent scrollbars - right margin accounts for colorbar
        margin=dict(l=50, r=100, t=60, b=40),
    )
    
    # Use optimized config for responsive display
    config = get_plotly_config()
    config['toImageButtonOptions'] = {
        'format': 'png',
        'filename': 'calendar_heatmap',
        'height': None,
        'width': None,
        'scale': 2
    }
    
    st.plotly_chart(fig, config=config)


def _render_week_heatmap(ret_series: pd.Series, color_mode: str) -> None:
    """Render week heatmap (day of week vs week of year)."""
    if not PLOTLY_AVAILABLE or go is None:
        st.info("Plotly required for week heatmap")
        return
    
    ret_series.index = pd.to_datetime(ret_series.index)
    
    # Prepare data
    df = pd.DataFrame({'return': ret_series})
    df['week'] = df.index.isocalendar().week
    df['weekday'] = df.index.dayofweek
    df['year'] = df.index.year
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values='return',
        index='weekday',
        columns='week',
        aggfunc='mean'
    )
    
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    # Color scale - use centralized PM colors
    if color_mode == "Return Direction":
        colorscale = PM_DIVERGING
        zmid = 0
    elif color_mode == "Volatility Adjusted":
        colorscale = PM_DIVERGING
        zmid = 0
    else:  # Return Magnitude
        colorscale = PM_DIVERGING
        zmid = 0
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values * 100,
        x=[f"W{w}" for w in pivot_data.columns],
        y=weekday_labels,
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(
            title=dict(text="Return %", font=dict(size=11)),
            x=1.02,
            xref="paper",
            len=0.6,
            thickness=15,
        ),
        hovertemplate="Week %{x}<br>%{y}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title_text="Weekly Pattern Heatmap: Day of Week vs Week of Year",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 11},
        autosize=True,
        # Right margin accounts for colorbar
        margin=dict(l=80, r=100, t=60, b=60),
        xaxis={'title': 'Week of Year', 'gridcolor': '#3A3A3A'},
        yaxis={'title': 'Day of Week', 'gridcolor': '#3A3A3A'},
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, config=config)


def _render_month_comparison(ret_series: pd.Series, years: List[int]) -> None:
    """Render month-by-month comparison across years."""
    if not PLOTLY_AVAILABLE or go is None:
        st.info("Plotly required for month comparison")
        return
    
    ret_series.index = pd.to_datetime(ret_series.index)
    
    # Group by year and month
    df = pd.DataFrame({'return': ret_series})
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    month_avg = df.groupby(['year', 'month'])['return'].mean().unstack(level=0)
    month_avg.index = [calendar.month_abbr[i] for i in month_avg.index]
    
    # Create bar chart
    fig = go.Figure()
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c']
    
    for idx, year in enumerate(years):
        if year in month_avg.columns:
            color = colors[idx % len(colors)]
            fig.add_trace(go.Bar(
                name=str(year),
                x=month_avg.index,
                y=month_avg[year].values * 100,
                marker_color=color,
                hovertemplate=f"{year}<br>Month: %{{x}}<br>Avg Return: %{{y:.2f}}%<extra></extra>",
            ))
    
    fig.update_layout(
        title_text="Monthly Average Returns: Year Comparison",
        xaxis_title="Month",
        yaxis_title="Average Return (%)",
        barmode='group',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 11},
        autosize=True,
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'gridcolor': '#3A3A3A'},
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, config=config)


def _render_year_comparison(ret_series: pd.Series, years: List[int]) -> None:
    """Render year-over-year comparison."""
    if not PLOTLY_AVAILABLE or go is None:
        st.info("Plotly required for year comparison")
        return
    
    ret_series.index = pd.to_datetime(ret_series.index)
    
    # Cumulative returns by year
    fig = go.Figure()
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c']
    
    for idx, year in enumerate(years):
        year_data = ret_series[ret_series.index.year == year]
        if len(year_data) == 0:
            continue
        
        cumret = (1 + year_data).cumprod() - 1
        color = colors[idx % len(colors)]
        
        # Normalize dates to day of year for comparison
        day_of_year = year_data.index.dayofyear
        
        fig.add_trace(go.Scatter(
            x=day_of_year,
            y=cumret.values * 100,
            mode='lines',
            name=str(year),
            line={'color': color, 'width': 2},
            hovertemplate=f"{year}<br>Day: %{{x}}<br>Cum Return: %{{y:.2f}}%<extra></extra>",
        ))
    
    fig.update_layout(
        title_text="Year-over-Year Cumulative Returns Comparison",
        xaxis_title="Day of Year",
        yaxis_title="Cumulative Return (%)",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 11},
        autosize=True,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='x unified',
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'gridcolor': '#3A3A3A'},
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, config=config)


def _render_statistical_breakdown(ret_series: pd.Series) -> None:
    """Render statistical breakdown by various periods."""
    ret_series.index = pd.to_datetime(ret_series.index)
    df = pd.DataFrame({'return': ret_series})
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Day of week stats
    dow_stats = df.groupby('weekday')['return'].agg(['mean', 'std', 'count', 'min', 'max']).round(4)
    dow_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    dow_stats.columns = ['Mean', 'Std', 'Count', 'Min', 'Max']
    
    # Month stats
    month_stats = df.groupby('month')['return'].agg(['mean', 'std', 'count', 'min', 'max']).round(4)
    month_stats.index = [calendar.month_abbr[i] for i in month_stats.index]
    month_stats.columns = ['Mean', 'Std', 'Count', 'Min', 'Max']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### By Day of Week")
        st.dataframe(dow_stats, width="stretch")
    
    with col2:
        st.markdown("### By Month")
        st.dataframe(month_stats, width="stretch")


def _render_anomaly_detection(ret_series: pd.Series) -> None:
    """Detect and highlight anomalous days."""
    # Calculate rolling z-scores
    rolling_mean = ret_series.rolling(window=20).mean()
    rolling_std = ret_series.rolling(window=20).std()
    z_scores = (ret_series - rolling_mean) / rolling_std
    
    # Identify anomalies (|z-score| > 2.5)
    anomalies = z_scores[z_scores.abs() > 2.5].sort_values(key=abs, ascending=False)
    
    if len(anomalies) == 0:
        st.info("No significant anomalies detected")
        return
    
    st.markdown(f"**Found {len(anomalies)} anomalous days** (|z-score| > 2.5)")
    
    # Create table of top anomalies
    anomaly_df = pd.DataFrame({
        'Date': anomalies.index,
        'Return': ret_series.loc[anomalies.index],
        'Z-Score': anomalies.values,
        'Type': ['Extreme Positive' if z > 0 else 'Extreme Negative' for z in anomalies.values]
    })
    anomaly_df['Return'] = anomaly_df['Return'].apply(lambda x: f"{x:.2%}")
    anomaly_df['Z-Score'] = anomaly_df['Z-Score'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(anomaly_df.head(20), width='stretch', hide_index=True)
    
    # Visualize anomalies
    if PLOTLY_AVAILABLE and go is not None:
        fig = go.Figure()
        
        # Normal returns
        normal_dates = ret_series.index[~ret_series.index.isin(anomalies.index)]
        normal_returns = ret_series.loc[normal_dates]
        
        fig.add_trace(go.Scatter(
            x=normal_dates,
            y=normal_returns.values * 100,
            mode='markers',
            name='Normal',
            marker={'color': '#95a5a6', 'size': 4, 'opacity': 0.6},
        ))
        
        # Anomalies
        anomaly_returns = ret_series.loc[anomalies.index]
        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in anomaly_returns.values]
        
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomaly_returns.values * 100,
            mode='markers',
            name='Anomaly',
            marker={'color': colors, 'size': 8, 'opacity': 0.8},
            hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<br>Z-Score: %{customdata:.2f}<extra></extra>",
            customdata=anomalies.values,
        ))
        
        fig.update_layout(
            title_text="Return Anomalies Detection",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            paper_bgcolor='#0E1117',
            plot_bgcolor='#262730',
            font={'color': '#ecf0f1', 'size': 11},
            height=400,
            autosize=True,
            margin=dict(l=60, r=60, t=60, b=60),
            hovermode='closest',
            xaxis={'gridcolor': '#3A3A3A'},
            yaxis={'gridcolor': '#3A3A3A'},
        )
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'responsive': True,
        }
        
        st.plotly_chart(fig, config=config)
