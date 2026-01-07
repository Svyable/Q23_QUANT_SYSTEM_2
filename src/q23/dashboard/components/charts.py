"""
Enhanced Interactive Charts for PM Dashboard

Uses Plotly for interactive visualizations with:
- Hover details
- Zoom/pan
- Cross-filtering
- PM-focused color schemes and annotations
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Plotly imports with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# PM-focused color palette
PM_COLORS = {
    "positive": "#2ecc71",      # Green for gains
    "negative": "#e74c3c",      # Red for losses
    "neutral": "#3498db",       # Blue for neutral
    "accent": "#f39c12",        # Orange for highlights
    "muted": "#95a5a6",         # Gray for secondary
    "background": "#0E1117",    # Dark background
    "card": "#262730",          # Card background
    "text": "#FAFAFA",          # Light text
    "grid": "#3A3A3A",          # Grid lines
}

PM_COLORSCALE = [
    [0.0, "#e74c3c"],    # Deep red
    [0.25, "#f39c12"],   # Orange
    [0.5, "#f1c40f"],    # Yellow
    [0.75, "#2ecc71"],   # Green
    [1.0, "#27ae60"],    # Deep green
]

PM_DIVERGING = [
    [0.0, "#c0392b"],    # Deep red
    [0.5, "#f1f1f1"],    # White/neutral
    [1.0, "#27ae60"],    # Deep green
]

# Long positions: Green to black gradient (for treemaps)
LONG_TREEMAP_COLORS = [
    [0.0, "#000000"],    # Black (min weight)
    [0.25, "#145a32"],   # Very dark green
    [0.5, "#1e8449"],    # Darker green
    [0.75, "#27ae60"],   # Dark green
    [1.0, "#2ecc71"],   # Bright green (max weight)
]

# Short positions: Red to black gradient (for treemaps)
SHORT_TREEMAP_COLORS = [
    [0.0, "#000000"],    # Black (min weight)
    [0.25, "#7b241c"],   # Very dark red
    [0.5, "#a93226"],    # Darker red
    [0.75, "#c0392b"],   # Dark red
    [1.0, "#e74c3c"],   # Bright red (max absolute weight)
]


def get_plotly_config(
    responsive: bool = True,
    display_mode_bar: bool = True,
    remove_buttons: Optional[List[str]] = None,
) -> dict:
    """
    Get optimized Plotly config for responsive display without scrollbars.
    
    Optimizations:
    - Responsive sizing to fit container
    - Removes unnecessary buttons
    - Prevents scrollbars and cut-off issues
    - Optimized for dark theme
    
    Args:
        responsive: Enable responsive sizing
        display_mode_bar: Show/hide modebar
        remove_buttons: List of buttons to remove (e.g., ['lasso2d', 'select2d'])
    
    Returns:
        Plotly config dictionary
    """
    if remove_buttons is None:
        remove_buttons = ['lasso2d', 'select2d']
    
    return {
        'displayModeBar': display_mode_bar,
        'displaylogo': False,
        'modeBarButtonsToRemove': remove_buttons,
        'responsive': responsive,
        'autosizable': True,
        'fillFrame': True,
        'frameMargins': 0,
        'scrollZoom': True,
        'doubleClick': 'reset',
    }


def get_plotly_layout(
    title: str = "",
    height: Optional[int] = None,
    show_legend: bool = True,
) -> dict:
    """Get consistent Plotly layout for dark theme.
    
    Args:
        title: Chart title
        height: Optional fixed height. If None, chart will be fully responsive with autosize=True
        show_legend: Whether to show legend
    
    Returns:
        Dictionary of layout parameters for Plotly figure
    """
    layout = {
        "title": {"text": title, "font": {"size": 16, "color": PM_COLORS["text"]}},
        "paper_bgcolor": PM_COLORS["background"],
        "plot_bgcolor": PM_COLORS["card"],
        "font": {"color": PM_COLORS["text"], "size": 11},
        "autosize": True,
        "showlegend": show_legend,
        "legend": {
            "bgcolor": "rgba(0,0,0,0.5)",
            "bordercolor": PM_COLORS["grid"],
            "borderwidth": 1,
        },
        "xaxis": {
            "gridcolor": PM_COLORS["grid"],
            "zerolinecolor": PM_COLORS["grid"],
        },
        "yaxis": {
            "gridcolor": PM_COLORS["grid"],
            "zerolinecolor": PM_COLORS["grid"],
        },
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 40, "t": 60, "b": 40},
    }
    
    # Only set height if explicitly provided (for fixed-size charts)
    if height is not None:
        layout["height"] = height
    
    return layout


def create_risk_gauge(
    value: float,
    title: str = "Risk Level",
    min_val: float = 0,
    max_val: float = 0.5,
    thresholds: Optional[List[float]] = None,
    height: int = 250,
) -> "go.Figure":
    """Create a gauge chart for risk metrics."""
    if not PLOTLY_AVAILABLE:
        return None
    
    if thresholds is None:
        thresholds = [max_val * 0.33, max_val * 0.66, max_val]
    
    # Determine color based on value
    if value < thresholds[0]:
        color = PM_COLORS["positive"]
    elif value < thresholds[1]:
        color = PM_COLORS["accent"]
    else:
        color = PM_COLORS["negative"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,  # Convert to percentage
        number={"suffix": "%", "font": {"size": 24}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [min_val * 100, max_val * 100], "ticksuffix": "%"},
            "bar": {"color": color},
            "bgcolor": PM_COLORS["card"],
            "borderwidth": 2,
            "bordercolor": PM_COLORS["grid"],
            "steps": [
                {"range": [min_val * 100, thresholds[0] * 100], "color": "rgba(46, 204, 113, 0.3)"},
                {"range": [thresholds[0] * 100, thresholds[1] * 100], "color": "rgba(243, 156, 18, 0.3)"},
                {"range": [thresholds[1] * 100, max_val * 100], "color": "rgba(231, 76, 60, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    
    layout = get_plotly_layout(title="", height=height)
    layout["margin"] = {"l": 20, "r": 20, "t": 50, "b": 20}
    fig.update_layout(**layout)
    
    return fig


def create_waterfall_chart(
    values: pd.Series,
    title: str = "Attribution Waterfall",
    height: int = 400,
) -> "go.Figure":
    """Create a waterfall chart for attribution breakdown."""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Sort by absolute value for better visualization
    sorted_vals = values.reindex(values.abs().sort_values(ascending=False).index)
    
    # Create measure types
    measures = ["relative"] * len(sorted_vals) + ["total"]
    
    # Add total
    names = list(sorted_vals.index) + ["Total"]
    vals = list(sorted_vals.values) + [sorted_vals.sum()]
    
    # Colors based on sign
    colors = [PM_COLORS["positive"] if v >= 0 else PM_COLORS["negative"] for v in sorted_vals.values]
    colors.append(PM_COLORS["neutral"])
    
    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measures,
        x=names,
        textposition="outside",
        text=[f"{v:+.2%}" for v in vals],
        y=[v * 100 for v in vals],
        connector={"line": {"color": PM_COLORS["muted"]}},
        decreasing={"marker": {"color": PM_COLORS["negative"]}},
        increasing={"marker": {"color": PM_COLORS["positive"]}},
        totals={"marker": {"color": PM_COLORS["neutral"]}},
    ))
    
    layout = get_plotly_layout(title, height)
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    
    return fig


def create_treemap_chart(
    values: pd.Series,
    title: str = "Portfolio Composition",
    color_by_value: bool = True,
    stock_metrics: Optional[Dict[str, Dict]] = None,
    height: int = 450,
) -> "go.Figure":
    """
    Create a treemap for portfolio/sector visualization with enhanced tooltips.
    
    Args:
        values: Series of portfolio weights by stock
        title: Chart title
        color_by_value: Color by long/short direction
        stock_metrics: Optional dict of {symbol: metrics_dict} for enhanced tooltips
                      Each metrics_dict can contain: sharpe, sortino, var_95, max_dd,
                      skewness, days_held, ret_30d, pnl_contrib, beta, etc.
        height: Chart height
    
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Prepare data
    abs_values = values.abs()
    signs = np.sign(values)
    
    # Separate longs and shorts
    longs = values[values > 0]
    shorts = values[values < 0]
    
    # Create color array with separate gradients for longs and shorts
    if color_by_value:
        colors = []
        for val in values.values:
            abs_val = abs(val)
            if val > 0:  # Long position
                # Normalize to 0-1 for long gradient (green to black)
                max_long = longs.max() if not longs.empty else 1.0
                min_long = longs.min() if not longs.empty else 0.0
                if max_long > min_long:
                    norm_val = (abs_val - min_long) / (max_long - min_long)
                else:
                    norm_val = 1.0 if abs_val > 0 else 0.0
                # Map to long gradient colorscale
                if norm_val <= 0.0:
                    color = "#000000"
                elif norm_val <= 0.25:
                    color = "#145a32"
                elif norm_val <= 0.5:
                    color = "#1e8449"
                elif norm_val <= 0.75:
                    color = "#27ae60"
                else:
                    color = "#2ecc71"
            elif val < 0:  # Short position
                # Normalize to 0-1 for short gradient (red to black)
                max_short = abs(shorts.min()) if not shorts.empty else 1.0
                min_short = abs(shorts.max()) if not shorts.empty else 0.0
                if max_short > min_short:
                    norm_val = (abs_val - min_short) / (max_short - min_short)
                else:
                    norm_val = 1.0 if abs_val > 0 else 0.0
                # Map to short gradient colorscale
                if norm_val <= 0.0:
                    color = "#000000"
                elif norm_val <= 0.25:
                    color = "#7b241c"
                elif norm_val <= 0.5:
                    color = "#a93226"
                elif norm_val <= 0.75:
                    color = "#c0392b"
                else:
                    color = "#e74c3c"
            else:  # Zero or NaN
                color = "#000000"
            colors.append(color)
        
        # Use custom colorscale (we'll set colors directly)
        colorscale = None
    else:
        # Use absolute values with standard colorscale
        colors = abs_values.values
        colorscale = PM_COLORSCALE
    
    # Build enhanced hover text if metrics provided
    if stock_metrics:
        hover_texts = []
        for symbol in values.index:
            weight = values[symbol]
            metrics = stock_metrics.get(symbol, {})
            
            # Build rich tooltip
            lines = [
                f"<b>{symbol}</b>",
                f"<b>Weight:</b> {weight:+.2%}",
                "─" * 20,
            ]
            
            # Performance section
            if metrics.get('sharpe') is not None:
                lines.append(f"<b>Sharpe:</b> {metrics['sharpe']:.2f}")
            if metrics.get('sortino') is not None:
                lines.append(f"<b>Sortino:</b> {metrics['sortino']:.2f}")
            if metrics.get('ret_30d') is not None:
                ret_30d = metrics['ret_30d']
                color = "#2ecc71" if ret_30d >= 0 else "#e74c3c"
                lines.append(f"<b>30d Return:</b> <span style='color:{color}'>{ret_30d:+.2%}</span>")
            if metrics.get('pnl_contrib') is not None:
                pnl = metrics['pnl_contrib']
                color = "#2ecc71" if pnl >= 0 else "#e74c3c"
                lines.append(f"<b>PnL Contrib:</b> <span style='color:{color}'>{pnl:+.2%}</span>")
            
            # Risk section
            if any(k in metrics for k in ['var_95', 'max_dd', 'ann_vol']):
                lines.append("─" * 20)
                if metrics.get('var_95') is not None:
                    lines.append(f"<b>VaR 95%:</b> {metrics['var_95']:.2%}")
                if metrics.get('max_dd') is not None:
                    lines.append(f"<b>Max DD:</b> {metrics['max_dd']:.2%}")
                if metrics.get('ann_vol') is not None:
                    lines.append(f"<b>Ann Vol:</b> {metrics['ann_vol']:.1%}")
            
            # Distribution section
            if any(k in metrics for k in ['skewness', 'kurtosis']):
                lines.append("─" * 20)
                if metrics.get('skewness') is not None:
                    skew = metrics['skewness']
                    skew_label = "←Fat L" if skew < -0.5 else ("Fat R→" if skew > 0.5 else "Sym")
                    lines.append(f"<b>Skew:</b> {skew:.2f} ({skew_label})")
                if metrics.get('kurtosis') is not None:
                    kurt = metrics['kurtosis']
                    kurt_label = "Fat Tails" if kurt > 1 else ("Thin" if kurt < -1 else "Normal")
                    lines.append(f"<b>Kurt:</b> {kurt:.2f} ({kurt_label})")
            
            # Context section
            if any(k in metrics for k in ['days_held', 'beta', 'hit_rate']):
                lines.append("─" * 20)
                if metrics.get('days_held') is not None:
                    lines.append(f"<b>Days Held:</b> {metrics['days_held']}")
                if metrics.get('beta') is not None:
                    lines.append(f"<b>Beta:</b> {metrics['beta']:.2f}")
                if metrics.get('hit_rate') is not None:
                    lines.append(f"<b>Hit Rate:</b> {metrics['hit_rate']:.1%}")
                if metrics.get('n_obs') is not None:
                    lines.append(f"<b>N Obs:</b> {metrics['n_obs']}")
            
            hover_texts.append("<br>".join(lines))
        
        custom_hover = True
    else:
        hover_texts = [f"<b>{s}</b><br>Weight: {v:+.2%}" for s, v in zip(values.index, values.values)]
        custom_hover = False
    
    # Create treemap
    if color_by_value and colorscale is None:
        # Use direct color mapping
        fig = go.Figure(go.Treemap(
            labels=values.index.tolist(),
            parents=[""] * len(values),
            values=abs_values.values,
            text=[f"{v:+.2%}" for v in values.values],
            textinfo="label+text",
            marker={
                "colors": colors,
                "showscale": False,  # Don't show scale for custom colors
            },
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Treemap(
            labels=values.index.tolist(),
            parents=[""] * len(values),
            values=abs_values.values,
            text=[f"{v:+.2%}" for v in values.values],
            textinfo="label+text",
            marker={
                "colors": colors,
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": "Value"},
            },
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ))
    
    layout = get_plotly_layout(title, height)
    layout["margin"] = {"l": 10, "r": 10, "t": 50, "b": 10}
    fig.update_layout(**layout)
    
    return fig


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 500,
) -> "go.Figure":
    """Create an interactive correlation heatmap."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=PM_DIVERGING,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar={"title": "Correlation"},
    ))
    
    layout = get_plotly_layout(title, height, show_legend=False)
    layout["xaxis"]["tickangle"] = 45
    fig.update_layout(**layout)
    
    return fig


def create_risk_contribution_chart(
    contributions: pd.Series,
    title: str = "Risk Contributions",
    height: int = 400,
) -> "go.Figure":
    """Create horizontal bar chart for risk contributions."""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Sort by contribution
    sorted_contrib = contributions.sort_values(ascending=True)
    
    # Colors based on sign
    colors = [PM_COLORS["positive"] if v >= 0 else PM_COLORS["negative"] for v in sorted_contrib.values]
    
    fig = go.Figure(go.Bar(
        x=sorted_contrib.values * 100,
        y=sorted_contrib.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in sorted_contrib.values * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>",
    ))
    
    layout = get_plotly_layout(title, height)
    layout["xaxis"]["title"] = "Risk Contribution (%)"
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    
    return fig


def create_time_series_chart(
    data: pd.DataFrame,
    title: str = "Time Series",
    y_format: str = ".2%",
    height: int = 400,
    fill: bool = False,
) -> "go.Figure":
    """Create interactive time series chart."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure()
    
    colors = [PM_COLORS["positive"], PM_COLORS["neutral"], PM_COLORS["accent"], 
              PM_COLORS["negative"], PM_COLORS["muted"]]
    
    for i, col in enumerate(data.columns):
        color = colors[i % len(colors)]
        
        if fill:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col].values,
                name=col,
                mode="lines",
                line={"color": color, "width": 2},
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>Value: %{{y:{y_format}}}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col].values,
                name=col,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>Value: %{{y:{y_format}}}<extra></extra>",
            ))
    
    layout = get_plotly_layout(title, height)
    if y_format.endswith("%"):
        layout["yaxis"]["tickformat"] = y_format
    fig.update_layout(**layout)
    
    return fig


def create_scatter_with_regression(
    x: pd.Series,
    y: pd.Series,
    title: str = "Scatter Plot",
    x_label: str = "X",
    y_label: str = "Y",
    height: int = 400,
) -> "go.Figure":
    """Create scatter plot with regression line."""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Compute regression
    valid_mask = ~(x.isna() | y.isna())
    x_valid = x[valid_mask].values
    y_valid = y[valid_mask].values
    
    if len(x_valid) < 2:
        return None
    
    # Simple linear regression
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    from q23.shared.math_utils import safe_corrcoef
    corr = safe_corrcoef(x_valid, y_valid)
    r_squared = corr ** 2 if isinstance(corr, (int, float)) else 0.0
    
    # Create figure
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_valid,
        y=y_valid,
        mode="markers",
        name="Data",
        marker={"color": PM_COLORS["neutral"], "size": 6, "opacity": 0.7},
        hovertemplate=f"<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}<extra></extra>",
    ))
    
    # Regression line
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_line = slope * x_line + intercept
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"Regression (R²={r_squared:.3f})",
        line={"color": PM_COLORS["accent"], "width": 2, "dash": "dash"},
    ))
    
    layout = get_plotly_layout(title, height)
    layout["xaxis"]["title"] = x_label
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    
    return fig


def create_bullet_chart(
    actual: float,
    target: float,
    title: str = "Performance",
    ranges: Optional[List[float]] = None,
    height: int = 150,
) -> "go.Figure":
    """Create bullet chart for target vs actual."""
    if not PLOTLY_AVAILABLE:
        return None
    
    if ranges is None:
        ranges = [target * 0.5, target * 0.75, target * 1.25]
    
    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=actual,
        delta={"reference": target},
        gauge={
            "shape": "bullet",
            "axis": {"range": [None, max(ranges)]},
            "threshold": {
                "line": {"color": PM_COLORS["accent"], "width": 2},
                "thickness": 0.75,
                "value": target,
            },
            "steps": [
                {"range": [0, ranges[0]], "color": "rgba(231, 76, 60, 0.3)"},
                {"range": [ranges[0], ranges[1]], "color": "rgba(243, 156, 18, 0.3)"},
                {"range": [ranges[1], ranges[2]], "color": "rgba(46, 204, 113, 0.3)"},
            ],
            "bar": {"color": PM_COLORS["neutral"]},
        },
        title={"text": title},
    ))
    
    layout = get_plotly_layout(title="", height=height)
    layout["margin"] = {"l": 120, "r": 40, "t": 40, "b": 20}
    fig.update_layout(**layout)
    
    return fig


def create_sparkline(
    series: pd.Series,
    height: int = 60,
    width: int = 200,
) -> "go.Figure":
    """Create a minimal sparkline chart."""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Determine color based on trend
    color = PM_COLORS["positive"] if series.iloc[-1] > series.iloc[0] else PM_COLORS["negative"]
    
    fig = go.Figure(go.Scatter(
        x=list(range(len(series))),
        y=series.values,
        mode="lines",
        line={"color": color, "width": 1.5},
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        width=width,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    
    return fig


def create_multi_metric_card(
    metrics: Dict[str, Tuple[float, str, Optional[float]]],
    title: str = "Metrics",
    height: int = 120,
) -> "go.Figure":
    """
    Create a multi-metric card display.
    
    Args:
        metrics: Dict of {name: (value, format, delta)}
        title: Card title
        height: Card height
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    n_metrics = len(metrics)
    
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        specs=[[{"type": "indicator"}] * n_metrics],
    )
    
    for i, (name, (value, fmt, delta)) in enumerate(metrics.items()):
        indicator_params = {
            "mode": "number+delta" if delta is not None else "number",
            "value": value,
            "title": {"text": name, "font": {"size": 12}},
            "number": {"font": {"size": 20}},
        }
        
        if delta is not None:
            indicator_params["delta"] = {"reference": value - delta}
        
        if fmt.endswith("%"):
            indicator_params["number"]["suffix"] = "%"
            indicator_params["value"] = value * 100
        
        fig.add_trace(go.Indicator(**indicator_params), row=1, col=i+1)
    
    layout = get_plotly_layout(title=title, height=height)
    layout["title"]["font"]["size"] = 14
    layout["margin"] = {"l": 20, "r": 20, "t": 40, "b": 20}
    fig.update_layout(**layout)

    return fig


# =============================================================================
# ADVANCED VISUALIZATION COMPONENTS - STRENGTHENED UI FRAMEWORK
# =============================================================================

def create_3d_risk_surface(
    x_data: pd.Series,
    y_data: pd.Series,
    z_data: pd.Series,
    x_label: str = "X Variable",
    y_label: str = "Y Variable",
    z_label: str = "Risk Measure",
    title: str = "3D Risk Surface",
    height: int = 600,
) -> Optional["go.Figure"]:
    """
    Create interactive 3D surface plot for risk analysis.
    Perfect for visualizing risk across multiple dimensions (e.g., volatility vs drawdown).
    """
    if not PLOTLY_AVAILABLE:
        return None

    try:
        import plotly.graph_objects as go

        # Create meshgrid for surface
        x_unique = np.sort(x_data.unique())
        y_unique = np.sort(y_data.unique())

        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X)

        # Interpolate z values onto grid
        from scipy.interpolate import griddata
        points = np.column_stack([x_data.values, y_data.values])
        Z_flat = griddata(points, z_data.values, (X, Y), method='linear', fill_value=np.nan)

        # Handle NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(Z_flat)
        if nan_mask.any():
            Z_nn = griddata(points, z_data.values, (X, Y), method='nearest')
            Z_flat[nan_mask] = Z_nn[nan_mask]
            Z = Z_flat
        else:
            Z = Z_flat

        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=PM_DIVERGING,
            colorbar=dict(
                title=z_label,
                titleside="right",
                titlefont=dict(size=12, color=PM_COLORS["text"])
            ),
            hovertemplate=f"<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}<br><b>{z_label}</b>: %{{z:.4f}}<extra></extra>",
        )])

        # Update layout for 3D
        fig.update_layout(
            title={"text": title, "font": {"size": 16, "color": PM_COLORS["text"]}},
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                xaxis=dict(
                    backgroundcolor=PM_COLORS["background"],
                    gridcolor=PM_COLORS["grid"],
                    showbackground=True,
                    zerolinecolor=PM_COLORS["grid"],
                ),
                yaxis=dict(
                    backgroundcolor=PM_COLORS["background"],
                    gridcolor=PM_COLORS["grid"],
                    showbackground=True,
                    zerolinecolor=PM_COLORS["grid"],
                ),
                zaxis=dict(
                    backgroundcolor=PM_COLORS["background"],
                    gridcolor=PM_COLORS["grid"],
                    showbackground=True,
                    zerolinecolor=PM_COLORS["grid"],
                ),
                bgcolor=PM_COLORS["background"],
            ),
            paper_bgcolor=PM_COLORS["background"],
            font={"color": PM_COLORS["text"], "size": 11},
            height=height,
        )

        return fig

    except Exception as e:
        print(f"Error creating 3D risk surface: {e}")
        return None


def create_network_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    title: str = "Network Graph",
    height: int = 600,
    node_color_col: Optional[str] = None,
    node_size_col: Optional[str] = None,
    edge_weight_col: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create interactive network graph for relationship visualization.
    Perfect for factor correlations, stock relationships, or strategy connections.
    """
    if not PLOTLY_AVAILABLE:
        return None

    try:
        import plotly.graph_objects as go

        # Prepare node data
        node_x = nodes.get('x', np.random.randn(len(nodes)) * 10)
        node_y = nodes.get('y', np.random.randn(len(nodes)) * 10)
        node_text = nodes.get('label', nodes.index)

        # Node colors and sizes
        node_colors = PM_COLORS["neutral"]
        if node_color_col and node_color_col in nodes.columns:
            node_colors = nodes[node_color_col].map({
                'positive': PM_COLORS["positive"],
                'negative': PM_COLORS["negative"],
                'neutral': PM_COLORS["neutral"]
            }).fillna(PM_COLORS["neutral"])

        node_sizes = 20
        if node_size_col and node_size_col in nodes.columns:
            # Scale sizes between 10-40
            size_data = nodes[node_size_col]
            node_sizes = 10 + 30 * (size_data - size_data.min()) / (size_data.max() - size_data.min() + 1e-12)

        # Create node traces
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=2,
                line_color=PM_COLORS["card"],
            )
        )

        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_weights = []

        for _, edge in edges.iterrows():
            x0, y0 = nodes.loc[edge['source'], ['x', 'y']]
            x1, y1 = nodes.loc[edge['target'], ['x', 'y']]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            weight = edge.get(edge_weight_col, 1.0) if edge_weight_col else 1.0
            edge_weights.extend([weight, weight, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=PM_COLORS["grid"]),
            hoverinfo='none',
            mode='lines'
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout
        layout = get_plotly_layout(title, height)
        layout.update(
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.update_layout(**layout)

        return fig

    except Exception as e:
        print(f"Error creating network graph: {e}")
        return None


def create_risk_heatmap_with_clusters(
    data: pd.DataFrame,
    title: str = "Risk Heatmap with Clusters",
    height: int = 600,
    cluster_method: str = "hierarchical",
    n_clusters: int = 5,
) -> Optional["go.Figure"]:
    """
    Create clustered heatmap for risk/correlation analysis.
    Uses hierarchical clustering to group similar assets/periods.
    """
    if not PLOTLY_AVAILABLE:
        return None

    try:
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        from scipy.spatial.distance import pdist
        import plotly.figure_factory as ff

        # Prepare data matrix
        data_matrix = data.values
        row_labels = data.index.tolist()
        col_labels = data.columns.tolist()

        # Perform hierarchical clustering
        row_linkage = linkage(pdist(data_matrix), method='ward')
        col_linkage = linkage(pdist(data_matrix.T), method='ward')

        # Get cluster assignments
        row_clusters = fcluster(row_linkage, n_clusters, criterion='maxclust')
        col_clusters = fcluster(col_linkage, n_clusters, criterion='maxclust')

        # Create clustered heatmap
        fig = ff.create_dendrogram(
            data_matrix,
            orientation='bottom',
            labels=col_labels,
            linkagefun=lambda x: linkage(x, 'ward')
        )

        # Add heatmap
        heatmap = go.Heatmap(
            z=data_matrix,
            x=col_labels,
            y=row_labels,
            colorscale=PM_DIVERGING,
            showscale=True,
            colorbar=dict(
                title="Value",
                titleside="right",
                titlefont=dict(size=12, color=PM_COLORS["text"])
            ),
        )

        fig.add_trace(heatmap)

        # Update layout
        layout = get_plotly_layout(title, height)
        layout.update(
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                tickfont=dict(size=10)
            )
        )
        fig.update_layout(**layout)

        return fig

    except Exception as e:
        print(f"Error creating clustered heatmap: {e}")
        return None


def create_multi_timeframe_chart(
    data: pd.DataFrame,
    primary_metric: str,
    secondary_metrics: List[str],
    title: str = "Multi-Timeframe Analysis",
    height: int = 600,
) -> Optional["go.Figure"]:
    """
    Create multi-timeframe chart showing primary metric with secondary indicators.
    Perfect for regime analysis and multi-horizon performance.
    """
    if not PLOTLY_AVAILABLE:
        return None

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots: main chart + secondary indicators
        n_secondary = len(secondary_metrics)
        subplot_titles = [primary_metric] + secondary_metrics

        fig = make_subplots(
            rows=n_secondary + 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_width=[0.6] + [0.4/n_secondary] * n_secondary
        )

        # Primary metric (full height)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[primary_metric],
                mode='lines',
                name=primary_metric,
                line=dict(color=PM_COLORS["accent"], width=2),
                fill='tozeroy',
                fillcolor='rgba(243, 156, 18, 0.1)',
            ),
            row=1, col=1
        )

        # Secondary metrics (stacked below)
        colors = [PM_COLORS["positive"], PM_COLORS["negative"], PM_COLORS["neutral"], PM_COLORS["warning"]]

        for i, metric in enumerate(secondary_metrics):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=colors[i % len(colors)], width=1),
                ),
                row=i+2, col=1
            )

        # Update layout
        layout = get_plotly_layout(title, height)
        layout.update(
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Format y-axes appropriately
        fig.update_yaxes(title_text=primary_metric, row=1, col=1)
        for i, metric in enumerate(secondary_metrics):
            fig.update_yaxes(title_text=metric, row=i+2, col=1)

        fig.update_layout(**layout)

        return fig

    except Exception as e:
        print(f"Error creating multi-timeframe chart: {e}")
        return None


def create_factor_attribution_waterfall(
    attribution_data: pd.DataFrame,
    title: str = "Factor Attribution Waterfall",
    height: int = 500,
) -> Optional["go.Figure"]:
    """
    Create waterfall chart for factor attribution analysis.
    Shows contribution of each factor to total return.
    """
    if not PLOTLY_AVAILABLE:
        return None

    try:
        import plotly.graph_objects as go

        # Prepare waterfall data
        factors = attribution_data.index.tolist()
        contributions = attribution_data.values.flatten()

        # Calculate cumulative values for waterfall
        cumulative = np.cumsum(contributions)
        cumulative = np.concatenate([[0], cumulative[:-1]])

        # Create waterfall traces
        fig = go.Figure()

        # Positive contributions
        pos_mask = contributions >= 0
        if pos_mask.any():
            fig.add_trace(go.Bar(
                x=factors,
                y=np.where(pos_mask, contributions, 0),
                name="Positive",
                marker_color=PM_COLORS["positive"],
                offsetgroup=0,
            ))

        # Negative contributions
        neg_mask = contributions < 0
        if neg_mask.any():
            fig.add_trace(go.Bar(
                x=factors,
                y=np.where(neg_mask, contributions, 0),
                name="Negative",
                marker_color=PM_COLORS["negative"],
                offsetgroup=0,
            ))

        # Cumulative line
        fig.add_trace(go.Scatter(
            x=factors,
            y=cumulative + contributions,
            mode='lines+markers',
            name="Cumulative",
            line=dict(color=PM_COLORS["accent"], width=3),
            marker=dict(size=8, color=PM_COLORS["accent"]),
        ))

        # Update layout
        layout = get_plotly_layout(title, height)
        layout.update(
            barmode='relative',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        fig.update_xaxes(title_text="Factors", tickangle=-45)
        fig.update_yaxes(title_text="Attribution (%)")

        fig.update_layout(**layout)

        return fig

    except Exception as e:
        print(f"Error creating factor attribution waterfall: {e}")
        return None
