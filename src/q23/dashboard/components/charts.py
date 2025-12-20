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


def get_plotly_layout(
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
) -> dict:
    """Get consistent Plotly layout for dark theme."""
    return {
        "title": {"text": title, "font": {"size": 16, "color": PM_COLORS["text"]}},
        "paper_bgcolor": PM_COLORS["background"],
        "plot_bgcolor": PM_COLORS["card"],
        "font": {"color": PM_COLORS["text"], "size": 11},
        "height": height,
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


def create_risk_gauge(
    value: float,
    title: str = "Risk Level",
    min_val: float = 0,
    max_val: float = 0.5,
    thresholds: Optional[List[float]] = None,
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
    
    fig.update_layout(
        paper_bgcolor=PM_COLORS["background"],
        font={"color": PM_COLORS["text"]},
        height=250,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    
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
) -> "go.Figure":
    """Create a treemap for portfolio/sector visualization."""
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
            hovertemplate="<b>%{label}</b><br>Weight: %{text}<extra></extra>",
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
            hovertemplate="<b>%{label}</b><br>Weight: %{text}<extra></extra>",
        ))
    
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": PM_COLORS["text"]}},
        paper_bgcolor=PM_COLORS["background"],
        font={"color": PM_COLORS["text"]},
        height=400,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    
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
    r_squared = np.corrcoef(x_valid, y_valid)[0, 1] ** 2
    
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
    
    fig.update_layout(
        paper_bgcolor=PM_COLORS["background"],
        font={"color": PM_COLORS["text"]},
        height=height,
        margin={"l": 120, "r": 40, "t": 40, "b": 20},
    )
    
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
) -> "go.Figure":
    """
    Create a multi-metric card display.
    
    Args:
        metrics: Dict of {name: (value, format, delta)}
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
    
    fig.update_layout(
        paper_bgcolor=PM_COLORS["background"],
        font={"color": PM_COLORS["text"]},
        height=120,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        title={"text": title, "font": {"size": 14}},
    )
    
    return fig
