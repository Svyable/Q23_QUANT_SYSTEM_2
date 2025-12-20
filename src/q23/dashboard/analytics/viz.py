from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import calendar


def setup_plot_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.facecolor': '#0E1117',
        'axes.facecolor': '#262730',
        'axes.edgecolor': '#4A4A4A',
        'axes.labelcolor': '#FAFAFA',
        'text.color': '#FAFAFA',
        'xtick.color': '#FAFAFA',
        'ytick.color': '#FAFAFA',
        'grid.color': '#3A3A3A',
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'lines.linewidth': 2,
    })


def create_cumulative_return_chart(
    ret_series: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Cumulative Return",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    perf = (1.0 + ret_series).cumprod() - 1.0
    ax.plot(perf.index, perf.values, label='Strategy', linewidth=2.5, color='#2ecc71')

    if benchmark is not None:
        bench_perf = (1.0 + benchmark).cumprod() - 1.0
        ax.plot(bench_perf.index, bench_perf.values, label='Benchmark', 
                linewidth=2, color='#e74c3c', alpha=0.7)

    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def create_drawdown_chart(ret_series: pd.Series, title: str = "Drawdown", 
                          figsize: Tuple[int, int] = (12, 5)) -> Figure:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    perf = (1.0 + ret_series).cumprod()
    peak = perf.cummax()
    dd = perf / peak - 1.0

    ax.fill_between(dd.index, dd.values, 0, alpha=0.7, color='#e74c3c', label='Drawdown')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def create_rolling_metrics_chart(
    metrics_df: pd.DataFrame,
    metrics: List[str],
    title: str = "Rolling Metrics",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            ax.plot(metrics_df.index, metrics_df[metric].values, 
                   label=metric.replace('_', ' ').title(),
                   linewidth=2, color=colors[i % len(colors)])

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def create_factor_ic_chart(
    ic_df: pd.DataFrame,
    factors: List[str],
    title: str = "Factor IC (Raw vs Smooth)",
    figsize: Tuple[int, int] = (14, 7)
) -> Figure:
    setup_plot_style()
    n_factors = len(factors)
    n_cols = 2
    n_rows = (n_factors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, factor in enumerate(factors):
        ax = axes[i]
        if (factor, 'ic_raw') in ic_df.columns:
            ax.plot(ic_df.index, ic_df[(factor, 'ic_raw')].values, 
                   label='Raw', alpha=0.5, linewidth=1, color='#95a5a6')
        if (factor, 'ic_smooth') in ic_df.columns:
            ax.plot(ic_df.index, ic_df[(factor, 'ic_smooth')].values, 
                   label='Smooth', linewidth=2, color='#3498db')

        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(factor, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def create_calendar_heatmap(
    ret_series: pd.Series,
    title: str = "Daily Returns Calendar",
    figsize: Tuple[int, int] = (16, 10)
) -> Figure:
    setup_plot_style()

    ret_df = ret_series.to_frame('ret')
    ret_df['year'] = ret_df.index.year
    ret_df['month'] = ret_df.index.month
    ret_df['day'] = ret_df.index.day

    years = sorted(ret_df['year'].unique())
    n_years = len(years)
    n_cols = 3
    n_rows = (n_years + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    vmin, vmax = ret_series.quantile([0.05, 0.95]).values
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    for i, year in enumerate(years):
        ax = axes[i]
        year_data = ret_df[ret_df['year'] == year]

        cal_data = np.full((12, 31), np.nan)
        for _, row in year_data.iterrows():
            m, d = int(row['month']) - 1, int(row['day']) - 1
            cal_data[m, d] = row['ret']

        im = ax.imshow(cal_data, aspect='auto', cmap='RdYlGn', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_yticks(range(12))
        ax.set_yticklabels([calendar.month_abbr[m+1] for m in range(12)], fontsize=8)
        ax.set_xticks(range(0, 31, 5))
        ax.set_xticklabels(range(1, 32, 5), fontsize=8)
        ax.set_title(f'{year}', fontsize=10, fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.colorbar(im, ax=axes, orientation='horizontal', 
                fraction=0.02, pad=0.04, label='Daily Return')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])

    return fig


def create_exposure_time_series(
    diag_df: pd.DataFrame,
    title: str = "Portfolio Exposure Over Time",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if 'gross_exposure' in diag_df.columns:
        ax1.plot(diag_df.index, diag_df['gross_exposure'].values, 
                label='Gross', linewidth=2, color='#3498db')
    if 'net_exposure' in diag_df.columns:
        ax1.plot(diag_df.index, diag_df['net_exposure'].values, 
                label='Net', linewidth=2, color='#2ecc71')

    ax1.set_title('Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Exposure', fontsize=10)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    if 'turnover' in diag_df.columns:
        ax2.bar(diag_df.index, diag_df['turnover'].values, 
               color='#e74c3c', alpha=0.7, width=1)

    ax2.set_title('Turnover', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Turnover', fontsize=10)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def create_factor_attribution_chart(
    attribution_df: pd.DataFrame,
    title: str = "Factor Attribution",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    cum_attr = attribution_df.cumsum()
    colors = plt.cm.tab20(np.linspace(0, 1, len(cum_attr.columns)))

    for i, col in enumerate(cum_attr.columns):
        ax.plot(cum_attr.index, cum_attr[col].values, 
               label=col, linewidth=2, color=colors[i])

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Attribution', fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
