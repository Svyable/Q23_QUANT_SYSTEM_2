"""
Position Stack Meta-Analysis Page

Provides portfolio manager meta-analysis tools for analyzing position overlap
across multiple strategies. Shows aggregated positions, participation heatmaps,
and detailed position breakdowns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import (
    DashboardData,
    discover_available_strategies,
    discover_strategy_tags,
    load_strategy_dashboard_data,
    get_strategy_display_name,
    get_strategy_configs,
    artifacts_to_dashboard_data,
)
from q23.dashboard.components.styles import (
    BENCHMARK_STRATEGY_IDS,
    build_strategy_color_map,
    inject_multiselect_colors,
)
from q23.dashboard.components.charts import PLOTLY_AVAILABLE


def _to_scalar(value) -> float:
    """Convert pandas Series or scalar to float."""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    elif pd.isna(value):
        return 0.0
    else:
        return float(value)


def _get_strategy_weight_at_date(
    data: DashboardData,
    date: pd.Timestamp,
    asset: str,
) -> float:
    """
    Get strategy weight for a specific date and asset, handling date mismatches.
    
    Args:
        data: DashboardData for the strategy
        date: Target date
        asset: Asset symbol
        
    Returns:
        Weight value (0.0 if not found)
    """
    if data.weights is None or data.weights.empty:
        return 0.0
    
    if asset not in data.weights.columns:
        return 0.0
    
    # Find best matching date
    if date in data.weights.index:
        target_date = date
    else:
        # Find closest date <= selected date
        dates_before = data.weights.index[data.weights.index <= date]
        if len(dates_before) > 0:
            target_date = dates_before[-1]
        else:
            # No dates before - use earliest available
            target_date = data.weights.index[0] if len(data.weights.index) > 0 else None
    
    if target_date is None:
        return 0.0
    
    try:
        return _to_scalar(data.weights.at[target_date, asset])
    except (KeyError, IndexError, AttributeError):
        try:
            return _to_scalar(data.weights.loc[target_date, asset])
        except (KeyError, IndexError, AttributeError):
            return 0.0


def _natural_sort_key(asset: str) -> tuple:
    """
    Generate sort key for natural/numeric sorting.
    Handles cases like: 100 > 75, 11 > 9 (not alphabetical)
    """
    import re
    # Split into text and numeric parts
    parts = re.split(r'(\d+)', asset)
    # Convert numeric parts to int for proper numeric comparison
    key_parts = []
    for part in parts:
        if part.isdigit():
            key_parts.append(int(part))
        else:
            key_parts.append(part.lower())
    return tuple(key_parts)


# Conditional plotly imports
if PLOTLY_AVAILABLE:
    try:
        import plotly.graph_objects as go  # type: ignore[reportMissingImports]
    except ImportError:
        go = None
else:
    go = None


def compute_aggregated_positions(
    strategy_data: Dict[str, DashboardData],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate positions across strategies and compute participation percentages.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        
    Returns:
        Tuple of (aggregated_weights, participation_df)
        - aggregated_weights: DataFrame with summed weights (dates × assets)
        - participation_df: DataFrame with participation % (dates × assets)
    """
    if not strategy_data:
        return pd.DataFrame(), pd.DataFrame()
    
    # Collect all weights DataFrames
    weights_dict = {}
    for sid, data in strategy_data.items():
        if data.weights is not None and not data.weights.empty:
            weights_dict[sid] = data.weights.fillna(0.0)
    
    if not weights_dict:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find common date range
    all_dates = set()
    for w_df in weights_dict.values():
        all_dates.update(w_df.index)
    
    if not all_dates:
        return pd.DataFrame(), pd.DataFrame()
    
    # Sort dates
    common_dates = sorted(all_dates)
    
    # Find all unique assets
    all_assets = set()
    for w_df in weights_dict.values():
        all_assets.update(w_df.columns)
    
    if not all_assets:
        return pd.DataFrame(), pd.DataFrame()
    
    # Sort assets using natural/numeric sorting
    all_assets = sorted(all_assets, key=_natural_sort_key)
    
    # Initialize aggregated DataFrames
    aggregated_weights = pd.DataFrame(0.0, index=common_dates, columns=all_assets)
    participation_count = pd.DataFrame(0.0, index=common_dates, columns=all_assets)
    
    # Sum weights and count participation
    n_strategies = len(weights_dict)
    
    for sid, w_df in weights_dict.items():
        # Reindex to common dates and assets
        w_aligned = w_df.reindex(index=common_dates, columns=all_assets, fill_value=0.0)
        
        # Sum weights
        aggregated_weights += w_aligned
        
        # Count participation (non-zero positions)
        participation_count += (w_aligned.abs() > 1e-12).astype(float)
    
    # Compute participation percentage
    participation_pct = (participation_count / n_strategies * 100) if n_strategies > 0 else participation_count
    
    # Add cash column - vectorized computation to avoid nested loops
    # Compute gross exposure for all strategies at once using aligned DataFrames
    cash_weights_by_strategy = []
    cash_participation_by_strategy = []
    
    for sid, w_df in weights_dict.items():
        # Align to common dates
        w_aligned = w_df.reindex(index=common_dates, fill_value=0.0)
        # Compute gross exposure for all dates at once (vectorized)
        gross_exposure = w_aligned.abs().sum(axis=1)  # Series: date -> gross exposure
        cash_weights = 1.0 - gross_exposure
        cash_weights_by_strategy.append(cash_weights)
        
        # Cash participation: strategies with cash > threshold (vectorized)
        cash_participation_by_strategy.append((cash_weights > 0.01).astype(float))
    
    # Sum cash weights across all strategies (vectorized)
    if cash_weights_by_strategy:
        cash_total = sum(cash_weights_by_strategy)
        cash_weights_series = (cash_total / n_strategies if n_strategies > 0 else cash_total)
        cash_weights_series.name = 'CASH'
        
        # Sum participation counts (vectorized)
        cash_participation_total = sum(cash_participation_by_strategy)
        cash_participation_series = (cash_participation_total / n_strategies * 100 if n_strategies > 0 else cash_participation_total)
        cash_participation_series.name = 'CASH'
    else:
        cash_weights_series = pd.Series(0.0, index=common_dates, name='CASH')
        cash_participation_series = pd.Series(0.0, index=common_dates, name='CASH')
    
    # Use pd.concat to avoid DataFrame fragmentation (instead of column assignment)
    aggregated_weights = pd.concat([aggregated_weights, cash_weights_series], axis=1)
    participation_pct = pd.concat([participation_pct, cash_participation_series], axis=1)
    
    return aggregated_weights, participation_pct


def create_position_heatmap_plotly(
    aggregated_weights: pd.DataFrame,
    participation_pct: pd.DataFrame,
    strategy_data: Dict[str, DashboardData],
) -> Optional[object]:
    """
    Create interactive position heatmap showing summed weights.
    
    Colors: Green (longs), Red (shorts), White (cash/zero)
    
    Args:
        aggregated_weights: DataFrame with summed weights (dates × assets)
        participation_pct: DataFrame with participation percentages
        strategy_data: Dict of strategy_id -> DashboardData for hover details
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or go is None or aggregated_weights.empty:
        return None
    
    # Limit to reasonable number of assets/dates for performance
    # Show top assets by max absolute weight
    if len(aggregated_weights.columns) > 100:
        max_weights = aggregated_weights.abs().max()
        top_assets = max_weights.nlargest(100).index.tolist()
        if 'CASH' in aggregated_weights.columns:
            top_assets.append('CASH')
        aggregated_weights = aggregated_weights[top_assets]
        participation_pct = participation_pct[top_assets]
    
    # Limit dates if too many
    if len(aggregated_weights) > 500:
        # Show most recent 500 dates
        aggregated_weights = aggregated_weights.tail(500)
        participation_pct = participation_pct.tail(500)
    
    # Prepare data for heatmap
    z_vals = aggregated_weights.values.T  # Assets × Dates
    assets = aggregated_weights.columns.tolist()
    dates = [d.strftime('%Y-%m-%d') for d in aggregated_weights.index]
    
    # Create custom hover text with strategy breakdown
    hover_text = []
    for i, asset in enumerate(assets):
        row_text = []
        for j, date in enumerate(aggregated_weights.index):
            weight = z_vals[i, j]
            
            # Extract participation as scalar
            if asset in participation_pct.columns and date in participation_pct.index:
                participation_val = _to_scalar(participation_pct.at[date, asset])
            else:
                participation_val = 0.0
            
            # Build strategy breakdown
            strategy_weights = []
            for sid, data in strategy_data.items():
                w = _get_strategy_weight_at_date(data, date, asset)
                if abs(w) > 1e-12:
                    strategy_weights.append(f"{get_strategy_display_name(sid)}: {w:.2%}")
            
            strategy_text = "<br>".join(strategy_weights[:5])  # Limit to 5 strategies
            if len(strategy_weights) > 5:
                strategy_text += f"<br>... and {len(strategy_weights) - 5} more"
            
            row_text.append(
                f"<b>{asset}</b><br>"
                f"Date: {date.strftime('%Y-%m-%d')}<br>"
                f"Aggregated Weight: {weight:.2%}<br>"
                f"Participation: {participation_val:.1f}%<br>"
                f"{strategy_text if strategy_text else 'No strategies holding'}"
            )
        hover_text.append(row_text)
    
    # Create diverging colorscale: Red (negative) -> White (zero) -> Green (positive)
    # Find min/max for normalization
    vmin = float(aggregated_weights.min().min())
    vmax = float(aggregated_weights.max().max())
    abs_max = max(abs(vmin), abs(vmax))
    
    # Custom colorscale
    colorscale = [
        [0.0, '#e74c3c'],      # Red for most negative
        [0.4, '#c0392b'],      # Dark red
        [0.45, '#ffffff'],     # White for zero
        [0.55, '#ffffff'],     # White for zero
        [0.6, '#27ae60'],      # Dark green
        [1.0, '#2ecc71'],      # Bright green for most positive
    ]
    
    # Normalize z values to [-abs_max, abs_max] range for colorscale
    z_normalized = np.clip(z_vals / abs_max if abs_max > 0 else z_vals, -1, 1)
    # Map to [0, 1] for colorscale
    z_scaled = (z_normalized + 1) / 2
    
    fig = go.Figure(data=go.Heatmap(
        z=z_scaled,
        x=dates,
        y=assets,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Position Weight', side='right'),
            tickmode='array',
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=[f'{vmin:.2%}', f'{vmin/2:.2%}', '0%', f'{vmax/2:.2%}', f'{vmax:.2%}'],
            tickformat='.0%',
        ),
    ))
    
    fig.update_layout(
        title={
            'text': 'Aggregated Position Heatmap (Summed Across Strategies)',
            'font': {'size': 16, 'color': '#ecf0f1'},
        },
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 10},
        autosize=True,
        height=max(600, len(assets) * 15),
        margin={'l': 120, 'r': 100, 't': 60, 'b': 40},
        xaxis={
            'title': 'Date',
            'gridcolor': '#3A3A3A',
            'tickangle': 45,
        },
        yaxis={
            'title': 'Asset',
            'gridcolor': '#3A3A3A',
            'autorange': 'reversed',
        },
        hovermode='closest',
        hoverlabel={
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 11, 'color': '#ecf0f1'},
        },
    )
    
    return fig


def create_today_aggregated_treemap_plotly(
    strategy_data: Dict[str, DashboardData],
) -> Optional[object]:
    """
    Create treemap of today's aggregated positions.
    
    Each asset is a box sized by aggregated weight (summed across all strategies).
    Color indicates direction: Green (long), Red (short).
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or go is None or not strategy_data:
        return None
    
    # Get latest date positions for each strategy
    latest_positions = {}
    latest_date = None
    
    for sid, data in strategy_data.items():
        if data.weights is not None and not data.weights.empty:
            latest_pos = data.weights.iloc[-1]
            latest_positions[sid] = latest_pos
            if latest_date is None:
                latest_date = data.weights.index[-1]
            else:
                latest_date = max(latest_date, data.weights.index[-1])
    
    if not latest_positions:
        return None
    
    # Aggregate weights across all strategies
    all_assets = set()
    for pos in latest_positions.values():
        all_assets.update(pos.index)
    
    if not all_assets:
        return None
    
    # Remove CASH for this view
    all_assets = sorted([a for a in all_assets if a != 'CASH'])
    
    # Sum weights across strategies for each asset
    aggregated_weights = pd.Series(0.0, index=all_assets)
    strategy_breakdown = {}  # Store individual strategy weights for hover
    
    for asset in all_assets:
        strategy_weights = []
        total_weight = 0.0
        
        for sid, pos in latest_positions.items():
            if asset in pos.index:
                w = _to_scalar(pos.at[asset])
                total_weight += w
                if abs(w) > 1e-12:
                    strategy_weights.append((get_strategy_display_name(sid), w))
        
        aggregated_weights[asset] = total_weight
        strategy_breakdown[asset] = strategy_weights
    
    # Filter to non-zero positions
    nonzero_mask = aggregated_weights.abs() > 1e-12
    aggregated_weights = aggregated_weights[nonzero_mask]
    
    if aggregated_weights.empty:
        return None
    
    # Sort by absolute weight (largest first)
    aggregated_weights = aggregated_weights.reindex(
        aggregated_weights.abs().sort_values(ascending=False).index
    )
    
    # Limit to top assets for performance
    if len(aggregated_weights) > 200:
        aggregated_weights = aggregated_weights.head(200)
    
    # Prepare colors: Green for longs, Red for shorts
    colors = []
    hover_texts = []
    
    for asset in aggregated_weights.index:
        weight = aggregated_weights[asset]
        
        # Color based on direction
        if weight > 0:
            # Long: Green gradient based on magnitude
            abs_weight = abs(weight)
            max_weight = aggregated_weights[aggregated_weights > 0].max() if (aggregated_weights > 0).any() else 1.0
            intensity = min(abs_weight / max_weight, 1.0) if max_weight > 0 else 0.0
            
            # Green gradient: darker for larger positions
            if intensity > 0.75:
                color = '#2ecc71'  # Bright green
            elif intensity > 0.5:
                color = '#27ae60'  # Dark green
            elif intensity > 0.25:
                color = '#1e8449'  # Darker green
            else:
                color = '#145a32'  # Very dark green
        elif weight < 0:
            # Short: Red gradient based on magnitude
            abs_weight = abs(weight)
            max_weight = abs(aggregated_weights[aggregated_weights < 0].min()) if (aggregated_weights < 0).any() else 1.0
            intensity = min(abs_weight / max_weight, 1.0) if max_weight > 0 else 0.0
            
            # Red gradient: darker for larger positions
            if intensity > 0.75:
                color = '#e74c3c'  # Bright red
            elif intensity > 0.5:
                color = '#c0392b'  # Dark red
            elif intensity > 0.25:
                color = '#a93226'  # Darker red
            else:
                color = '#7b241c'  # Very dark red
        else:
            color = '#ffffff'  # White for zero
        
        colors.append(color)
        
        # Build hover text with strategy breakdown
        hover_lines = [
            f"<b>{asset}</b>",
            f"Aggregated Weight: {weight:+.2%}",
            f"Direction: {'Long' if weight > 0 else 'Short' if weight < 0 else 'Neutral'}",
        ]
        
        if asset in strategy_breakdown:
            hover_lines.append("<br>Strategy Breakdown:")
            for strategy_name, strategy_weight in strategy_breakdown[asset][:10]:  # Limit to 10 strategies
                hover_lines.append(f"  {strategy_name}: {strategy_weight:+.2%}")
            if len(strategy_breakdown[asset]) > 10:
                hover_lines.append(f"  ... and {len(strategy_breakdown[asset]) - 10} more")
        
        hover_texts.append("<br>".join(hover_lines))
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=aggregated_weights.index.tolist(),
        parents=[""] * len(aggregated_weights),
        values=aggregated_weights.abs().values,  # Size by absolute weight
        text=[f"{w:+.1%}" for w in aggregated_weights.values],
        textinfo="label+text",
        marker={
            "colors": colors,
            "showscale": False,
            "line": {"width": 2, "color": "#1e1e1e"},
        },
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    
    fig.update_layout(
        title={
            'text': f'Today\'s Aggregated Positions ({latest_date.strftime("%Y-%m-%d")}) - Box Size = Weight Sum',
            'font': {'size': 16, 'color': '#ecf0f1'},
        },
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 11},
        autosize=True,
        height=700,
        margin={'l': 10, 'r': 10, 't': 60, 'b': 10},
        hoverlabel={
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 11, 'color': '#ecf0f1'},
        },
    )
    
    return fig


def create_participation_heatmap_plotly(
    participation_pct: pd.DataFrame,
) -> Optional[object]:
    """
    Create interactive participation heatmap showing strategy consensus.
    
    Colors: White (0%) -> Light Green (50%) -> Dark Green (100%)
    
    Args:
        participation_pct: DataFrame with participation percentages (dates × assets)
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or go is None or participation_pct.empty:
        return None
    
    # Limit to reasonable size
    if len(participation_pct.columns) > 100:
        # Show top assets by average participation
        avg_participation = participation_pct.mean()
        top_assets = avg_participation.nlargest(100).index.tolist()
        if 'CASH' in participation_pct.columns:
            top_assets.append('CASH')
        participation_pct = participation_pct[top_assets]
    
    if len(participation_pct) > 500:
        participation_pct = participation_pct.tail(500)
    
    # Prepare data
    z_vals = participation_pct.values.T  # Assets × Dates
    assets = participation_pct.columns.tolist()
    dates = [d.strftime('%Y-%m-%d') for d in participation_pct.index]
    
    # Create hover text
    hover_text = []
    for i, asset in enumerate(assets):
        row_text = []
        for j, date in enumerate(participation_pct.index):
            pct = z_vals[i, j]
            row_text.append(
                f"<b>{asset}</b><br>"
                f"Date: {date.strftime('%Y-%m-%d')}<br>"
                f"Participation: {pct:.1f}%"
            )
        hover_text.append(row_text)
    
    # Colorscale: White (0%) -> Light Green (50%) -> Dark Green (100%)
    colorscale = [
        [0.0, '#ffffff'],      # White
        [0.25, '#d5f4e6'],     # Very light green
        [0.5, '#85c1a0'],      # Light green
        [0.75, '#52b788'],     # Medium green
        [1.0, '#2ecc71'],      # Dark green
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=dates,
        y=assets,
        colorscale=colorscale,
        zmin=0,
        zmax=100,
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Participation %', side='right'),
            tickformat='.0f',
        ),
    ))
    
    fig.update_layout(
        title={
            'text': 'Strategy Participation Heatmap (Consensus Positions)',
            'font': {'size': 16, 'color': '#ecf0f1'},
        },
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1', 'size': 10},
        autosize=True,
        height=max(600, len(assets) * 15),
        margin={'l': 120, 'r': 100, 't': 60, 'b': 40},
        xaxis={
            'title': 'Date',
            'gridcolor': '#3A3A3A',
            'tickangle': 45,
        },
        yaxis={
            'title': 'Asset',
            'gridcolor': '#3A3A3A',
            'autorange': 'reversed',
        },
        hovermode='closest',
        hoverlabel={
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 11, 'color': '#ecf0f1'},
        },
    )
    
    return fig


def create_position_detail_table(
    aggregated_weights: pd.DataFrame,
    participation_pct: pd.DataFrame,
    strategy_data: Dict[str, DashboardData],
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Create detailed position table for a specific date.
    
    Args:
        aggregated_weights: DataFrame with summed weights
        participation_pct: DataFrame with participation percentages
        strategy_data: Dict of strategy_id -> DashboardData
        date: Date to show positions for
        
    Returns:
        DataFrame with columns: Asset, Aggregated Weight, Participation %, Strategy weights...
    """
    if aggregated_weights.empty or date not in aggregated_weights.index:
        return pd.DataFrame()
    
    # Get positions for this date
    date_weights = aggregated_weights.loc[date]
    date_participation = participation_pct.loc[date] if date in participation_pct.index else pd.Series()
    
    # Filter to non-zero positions
    significant_mask = date_weights.abs() > 1e-6
    if not significant_mask.any():
        return pd.DataFrame()
    
    # Build table data using pd.concat to avoid fragmentation
    # Collect all data first, then build DataFrame at once
    asset_list = []
    agg_weight_list = []
    long_gross_list = []  # Sum of all long positions
    short_gross_list = []  # Sum of all short positions (as positive)
    theo_weight_list = []  # Theo Weight = normalized aggregated weight (divided by n_strategies)
    part_pct_list = []
    strategy_weights_dict = {}  # strategy_name -> list of weights
    
    n_strategies = len(strategy_data)
    
    # Get strategy order (consistent ordering)
    strategy_order = list(strategy_data.keys())
    strategy_names = [get_strategy_display_name(sid) for sid in strategy_order]
    
    # Initialize strategy columns in consistent order
    for strategy_name in strategy_names:
        strategy_weights_dict[strategy_name] = []
    
    # Collect data for all assets
    for asset in date_weights[significant_mask].index:
        # Extract scalar values
        agg_weight = _to_scalar(date_weights.at[asset])
        part_pct = _to_scalar(date_participation.get(asset, 0.0)) if not date_participation.empty else 0.0
        
        # Calculate gross long and short separately
        long_total = 0.0
        short_total = 0.0
        
        # Add individual strategy weights and track long/short separately
        # Use consistent order
        for sid in strategy_order:
            data = strategy_data[sid]
            strategy_name = get_strategy_display_name(sid)
            
            # Use helper function for consistent date/asset lookup
            w = _get_strategy_weight_at_date(data, date, asset)
            
            # Track long/short totals
            if abs(w) > 1e-12:
                if w > 0:
                    long_total += w
                else:
                    short_total += abs(w)  # Store as positive for display
            
            # Store weight (keep sign for display)
            strategy_weights_dict[strategy_name].append(w)
        
        asset_list.append(asset)
        agg_weight_list.append(agg_weight)
        long_gross_list.append(long_total)
        short_gross_list.append(short_total)
        # Theo Weight = normalized aggregated weight (as if one unified portfolio)
        theo_weight = agg_weight / n_strategies if n_strategies > 0 else 0.0
        theo_weight_list.append(theo_weight)
        part_pct_list.append(part_pct)
    
    # Build DataFrame using pd.concat to avoid fragmentation
    data_dict = {
        'Asset': asset_list,
        'Net Aggregated': agg_weight_list,  # Net (long - short)
        'Long Gross': long_gross_list,  # Sum of all long positions
        'Short Gross': short_gross_list,  # Sum of all short positions (positive)
        'Meta Weight': theo_weight_list,  # Normalized weight (as if one portfolio)
        'Participation %': part_pct_list,
    }
    data_dict.update(strategy_weights_dict)
    
    df = pd.DataFrame(data_dict)
    
    # Sort by absolute net aggregated weight (descending)
    if not df.empty:
        df = df.reindex(df['Net Aggregated'].abs().sort_values(ascending=False).index)
    
    return df


def compute_overlap_metrics(
    strategy_data: Dict[str, DashboardData],
) -> Dict[str, float]:
    """
    Compute overlap metrics between strategies.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        
    Returns:
        Dict with overlap metrics
    """
    if len(strategy_data) < 2:
        return {}
    
    metrics = {}
    
    # Get latest positions for each strategy
    latest_positions = {}
    for sid, data in strategy_data.items():
        if data.weights is not None and not data.weights.empty:
            latest_positions[sid] = data.weights.iloc[-1]
    
    if len(latest_positions) < 2:
        return {}
    
    # Compute pairwise Jaccard similarity (overlap of non-zero positions)
    strategy_ids = list(latest_positions.keys())
    similarities = []
    
    for i, sid1 in enumerate(strategy_ids):
        for sid2 in strategy_ids[i+1:]:
            pos1 = latest_positions[sid1]
            pos2 = latest_positions[sid2]
            
            # Find common assets
            common_assets = pos1.index.intersection(pos2.index)
            if len(common_assets) == 0:
                continue
            
            # Non-zero positions
            non_zero1 = set(common_assets[pos1[common_assets].abs() > 1e-12])
            non_zero2 = set(common_assets[pos2[common_assets].abs() > 1e-12])
            
            # Jaccard similarity
            intersection = len(non_zero1 & non_zero2)
            union = len(non_zero1 | non_zero2)
            jaccard = intersection / union if union > 0 else 0.0
            
            similarities.append({
                'strategy1': get_strategy_display_name(sid1),
                'strategy2': get_strategy_display_name(sid2),
                'jaccard': jaccard,
            })
    
    if similarities:
        metrics['avg_jaccard_similarity'] = np.mean([s['jaccard'] for s in similarities])
        metrics['max_jaccard_similarity'] = max([s['jaccard'] for s in similarities])
        metrics['min_jaccard_similarity'] = min([s['jaccard'] for s in similarities])
    
    # Compute Herfindahl index of aggregated positions
    if latest_positions:
        # Aggregate latest positions
        all_assets = set()
        for pos in latest_positions.values():
            all_assets.update(pos.index)
        
        aggregated = pd.Series(0.0, index=sorted(all_assets, key=_natural_sort_key))
        for pos in latest_positions.values():
            aggregated += pos.reindex(aggregated.index, fill_value=0.0)
        
        # Normalize
        aggregated = aggregated / len(latest_positions)
        
        # Herfindahl index
        herfindahl = (aggregated.abs() ** 2).sum()
        metrics['herfindahl_index'] = herfindahl
    
    return metrics


def render_position_stack(
    current_strategy: str,
    date_range: Tuple[str, str],
) -> None:
    """Render the Position Stack meta-analysis page."""
    st.subheader("Position Stack")
    
    available_strategies = discover_available_strategies()
    
    # Use centralized benchmark IDs
    regular_strategies = [s for s in available_strategies if s not in BENCHMARK_STRATEGY_IDS]
    
    # Sync with Strategy Comparison selections if available
    # Check session state for strategy comparison selections
    comparison_key = "comparison_strategies"
    if comparison_key in st.session_state and st.session_state[comparison_key]:
        default_strategies = st.session_state[comparison_key]
        # Filter to only valid regular strategies (exclude benchmarks from default)
        default_strategies = [s for s in default_strategies if s in regular_strategies]
        if not default_strategies:
            default_strategies = regular_strategies[:min(2, len(regular_strategies))]
    else:
        default_strategies = regular_strategies[:min(2, len(regular_strategies))]
    
    # Add benchmark toggle
    if "q23_include_benchmarks" not in st.session_state:
        st.session_state.q23_include_benchmarks = bool(
            st.session_state.get("admin_default_include_benchmarks", False)
        )
    include_benchmarks = st.checkbox(
        "Include Market Benchmarks",
        value=bool(st.session_state.q23_include_benchmarks),
        key="q23_include_benchmarks_pos_stack",
        help="Include NYSE and NASDAQ equal-weighted and market-cap weighted benchmarks",
    )
    
    if len(regular_strategies) < 2 and not include_benchmarks:
        st.info("At least 2 strategies required for position stack analysis. Only 1 registered.")
        return
    
    st.markdown(
        """
        Meta-analysis of position overlap across strategies. View aggregated positions,
        strategy participation, and detailed position breakdowns.
        """
    )
    
    # Build color map
    all_available = regular_strategies + (BENCHMARK_STRATEGY_IDS if include_benchmarks else [])
    color_map = build_strategy_color_map(all_available)
    
    # Inject CSS for multiselect tag colors
    inject_multiselect_colors(color_map, get_strategy_display_name)
    
    comparison_strategies = st.multiselect(
        "Strategies to Analyze",
        options=regular_strategies,
        default=default_strategies,
        format_func=lambda x: get_strategy_display_name(x),
        key="position_stack_strategies",
    )
    
    # Add benchmarks if enabled
    if include_benchmarks:
        benchmark_selection = st.multiselect(
            "Benchmarks to Include",
            options=BENCHMARK_STRATEGY_IDS,
            default=BENCHMARK_STRATEGY_IDS,
            format_func=lambda x: get_strategy_display_name(x),
            key="position_stack_benchmarks",
        )
        comparison_strategies.extend(benchmark_selection)
    
    if len(comparison_strategies) < 2:
        st.warning("Select at least 2 strategies/benchmarks to analyze.")
        return
    
    # Load strategy data (reuse logic from strategy comparison)
    strategy_data: Dict[str, DashboardData] = {}
    strategy_configs = get_strategy_configs()
    
    for sid in comparison_strategies:
        if sid in BENCHMARK_STRATEGY_IDS:
            try:
                from q23.strategies.registry import StrategyRegistry
                benchmark_strategy = StrategyRegistry.get_instance(sid)
                
                start_date, end_date = date_range
                artifacts = benchmark_strategy.run(
                    min_date=start_date,
                    max_date=end_date,
                    write_outputs=False,
                )
                
                data = artifacts_to_dashboard_data(artifacts, sid)
                
                if data.weights is not None and not data.weights.empty:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    mask = (data.weights.index >= start_dt) & (data.weights.index <= end_dt)
                    data.weights = data.weights.loc[mask].copy() if mask.any() else data.weights
                    
                    if data.diag is not None and not data.diag.empty:
                        diag_mask = (data.diag.index >= start_dt) & (data.diag.index <= end_dt)
                        data.diag = data.diag.loc[diag_mask].copy() if diag_mask.any() else data.diag
                
                if data.weights is None or data.weights.empty:
                    st.warning(f"⚠️ Could not compute benchmark '{sid}' - no data available")
                    continue

                if data.weights.abs().sum().sum() < 1e-6:
                    st.warning(f"⚠️ Benchmark '{sid}' has zero weights - market cap data may be unavailable")
                    continue
                
                strategy_data[sid] = data
            except Exception as e:
                st.error(f"❌ Failed to compute benchmark '{sid}'")
                with st.expander("Error Details", expanded=False):
                    st.code(str(e))
                continue
        else:
            tags = discover_strategy_tags(sid)
            if not tags:
                st.warning(f"No output tags found for strategy '{sid}'")
                continue
            
            tag = tags[0]  # Use latest run
            data = load_strategy_dashboard_data(sid, tag, date_range=date_range)
            
            if data.weights is None or data.weights.empty:
                st.warning(f"No data loaded for strategy '{sid}'")
                continue
            
            strategy_data[sid] = data
    
    if len(strategy_data) < 2:
        st.error("Could not load data for at least 2 strategies.")
        return
    
    # Compute aggregated positions
    aggregated_weights, participation_pct = compute_aggregated_positions(strategy_data)
    
    if aggregated_weights.empty:
        st.error("No position data available.")
        return
    
    st.markdown("---")
    st.markdown("### Today's Aggregated Position Treemap")
    st.markdown("*Box size = Aggregated weight (summed across all strategies) | Green = Long, Red = Short*")
    
    # Today's aggregated treemap (front and center)
    today_treemap_fig = create_today_aggregated_treemap_plotly(strategy_data)
    if today_treemap_fig is not None:
        st.plotly_chart(today_treemap_fig, width='stretch')
    else:
        st.info("Plotly not available for interactive treemap")
    
    st.markdown("---")
    st.markdown("### Position Detail Table")
    
    # Date selector and download button on same row
    available_dates = sorted(aggregated_weights.index)
    if available_dates:
        col_date, col_download = st.columns([3, 1])
        
        with col_date:
            selected_date_idx = st.selectbox(
                "Select Date",
                options=range(len(available_dates)),
                index=len(available_dates) - 1,  # Default to latest
                format_func=lambda i: available_dates[i].strftime('%Y-%m-%d'),
            )
            selected_date = available_dates[selected_date_idx]
        
        # Create detail table
        detail_df = create_position_detail_table(
            aggregated_weights, participation_pct, strategy_data, selected_date
        )
        
        if not detail_df.empty:
            # Add Position Type column based on net aggregated weight and long/short gross
            def get_position_type(row):
                net = row['Net Aggregated']
                long_gross = row['Long Gross']
                short_gross = row['Short Gross']
                if long_gross > 1e-12 and short_gross > 1e-12:
                    return 'Long/Short'
                elif net > 1e-12:
                    return 'Long'
                elif net < -1e-12:
                    return 'Short'
                else:
                    return 'Cash'
            
            detail_df['Position Type'] = detail_df.apply(get_position_type, axis=1)
            
            # Prepare CSV data for download button (after adding Position Type)
            csv_df = detail_df.copy()
            # Round numeric columns to 2 decimals
            for col in csv_df.columns:
                if col != 'Asset' and col != 'Position Type':
                    csv_df[col] = csv_df[col].round(2)
            csv_data = csv_df.to_csv(index=False)
        else:
            csv_data = ""
        
        with col_download:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"position_details_{selected_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                disabled=detail_df.empty,
            )
        
        if not detail_df.empty:
            
            # Format columns for display
            display_df = detail_df.copy()
            display_df['Net Aggregated'] = display_df['Net Aggregated'].apply(lambda x: f"{x:+.2%}")
            display_df['Long Gross'] = display_df['Long Gross'].apply(lambda x: f"{x:.2%}")
            display_df['Short Gross'] = display_df['Short Gross'].apply(lambda x: f"{x:.2%}")
            display_df['Meta Weight'] = display_df['Meta Weight'].apply(lambda x: f"{x:.2%}")
            display_df['Participation %'] = display_df['Participation %'].apply(lambda x: f"{x:.1f}%")
            
            # Format strategy weight columns
            strategy_cols = [col for col in display_df.columns if col not in ['Asset', 'Net Aggregated', 'Long Gross', 'Short Gross', 'Meta Weight', 'Participation %', 'Position Type']]
            for col in strategy_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:+.2%}" if abs(x) > 1e-12 else "—")
            
            # Reorder columns: Asset, Position Type, Net Aggregated, Long Gross, Short Gross, Meta Weight, Participation %, then strategy columns
            col_order = ['Asset', 'Position Type', 'Net Aggregated', 'Long Gross', 'Short Gross', 'Meta Weight', 'Participation %'] + strategy_cols
            display_df = display_df[col_order]
            
            st.dataframe(display_df, height=400, width='stretch')
        else:
            st.info("No positions found for selected date")
    
    st.markdown("---")
    st.markdown("### Aggregated Position Heatmap")
    
    # Main heatmap
    heatmap_fig = create_position_heatmap_plotly(
        aggregated_weights, participation_pct, strategy_data
    )
    if heatmap_fig is not None:
        st.plotly_chart(heatmap_fig, width='stretch')
    else:
        st.info("Plotly not available for interactive heatmap")
    
    st.markdown("---")
    st.markdown("### Strategy Participation Heatmap")
    
    # Participation heatmap
    participation_fig = create_participation_heatmap_plotly(participation_pct)
    if participation_fig is not None:
        st.plotly_chart(participation_fig, width='stretch')
    else:
        st.info("Plotly not available for interactive heatmap")
    
    st.markdown("---")
    st.markdown("### Cash Analysis")
    
    # Cash metrics
    if 'CASH' in aggregated_weights.columns:
        cash_series = aggregated_weights['CASH']
        cash_participation = participation_pct['CASH'] if 'CASH' in participation_pct.columns else pd.Series()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Cash Weight", f"{cash_series.mean():.2%}")
        with col2:
            st.metric("Current Cash", f"{cash_series.iloc[-1]:.2%}" if len(cash_series) > 0 else "N/A")
        with col3:
            st.metric("Min Cash", f"{cash_series.min():.2%}")
        with col4:
            if not cash_participation.empty:
                st.metric("Avg Cash Participation", f"{cash_participation.mean():.1f}%")
        
        # Cash over time chart
        if PLOTLY_AVAILABLE and go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cash_series.index,
                y=cash_series.values * 100,
                mode='lines',
                name='Aggregated Cash',
                line={'color': '#ecf0f1', 'width': 2},
            ))
            fig.update_layout(
                title={'text': 'Cash Position Over Time', 'font': {'size': 14, 'color': '#ecf0f1'}},
                paper_bgcolor='#0E1117',
                plot_bgcolor='#262730',
                font={'color': '#ecf0f1'},
                xaxis={'title': 'Date', 'gridcolor': '#3A3A3A'},
                yaxis={'title': 'Cash Weight (%)', 'gridcolor': '#3A3A3A', 'tickformat': '.0f'},
                height=300,
            )
            st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Advanced Analytics
    with st.expander("🔍 Advanced Analytics", expanded=False):
        overlap_metrics = compute_overlap_metrics(strategy_data)
        
        if overlap_metrics:
            st.markdown("#### Overlap Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Jaccard Similarity", f"{overlap_metrics.get('avg_jaccard_similarity', 0):.3f}")
            with col2:
                st.metric("Max Jaccard Similarity", f"{overlap_metrics.get('max_jaccard_similarity', 0):.3f}")
            with col3:
                st.metric("Herfindahl Index", f"{overlap_metrics.get('herfindahl_index', 0):.3f}")
        
        # Top positions by aggregated weight
        if not aggregated_weights.empty:
            latest_weights = aggregated_weights.iloc[-1]
            top_positions = latest_weights.abs().nlargest(20)
            
            st.markdown("#### Top 20 Positions (Latest Date)")
            top_df = pd.DataFrame({
                'Asset': top_positions.index,
                'Aggregated Weight': latest_weights[top_positions.index],
                'Participation %': participation_pct.iloc[-1][top_positions.index] if not participation_pct.empty else 0.0,
            })
            top_df['Position Type'] = top_df['Aggregated Weight'].apply(
                lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Cash'
            )
            st.dataframe(top_df, width='stretch')

