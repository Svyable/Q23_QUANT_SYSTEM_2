from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from q23.shared.config import TransactionCostConfig, TransactionCostScheme


def compute_comprehensive_performance(
    weights: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    diag: Optional[pd.DataFrame] = None,
    tc_bps: float = 10.0,
    tc_config: Optional[TransactionCostConfig] = None,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics.
    
    Args:
        weights: Portfolio weights (time x asset)
        returns: Asset returns (optional, used if diag is None)
        diag: Portfolio diagnostics DataFrame (from engine)
        tc_bps: Legacy flat basis points (used if tc_config is None)
        tc_config: Transaction cost configuration
        
    Returns:
        Dictionary of performance metrics
    """
    # Get daily returns from diag or compute from weights/returns
    if diag is not None and not diag.empty:
        if "port_ret" in diag.columns:
            daily = diag["port_ret"].fillna(0.0)
        elif "active_ret" in diag.columns:
            daily = diag["active_ret"].fillna(0.0)
        else:
            daily = pd.Series(0.0, index=diag.index)
        
        # Get ATR-based TC if available
        if "tc_cost_atr" in diag.columns:
            tc_cost_atr = diag["tc_cost_atr"].fillna(0.0)
        else:
            tc_cost_atr = None
        
        if "tc_cost_flat" in diag.columns:
            tc_cost_flat = diag["tc_cost_flat"].fillna(0.0)
        else:
            tc_cost_flat = None
    else:
        daily = pd.Series(0.0, index=weights.index)
        tc_cost_atr = None
        tc_cost_flat = None

    eps = 1e-12

    # Basic return metrics
    ann_ret = float(daily.mean() * 252.0)
    ann_vol = float(daily.std(ddof=0) * np.sqrt(252.0))
    sharpe = ann_ret / (ann_vol + eps)

    neg = daily.copy()
    neg[neg > 0] = 0.0
    downside_vol = float(np.sqrt((neg**2).mean() * 252.0))
    sortino = ann_ret / (downside_vol + eps)

    eq = (1.0 + daily).cumprod()
    dd_series = eq / eq.cummax() - 1.0
    max_dd = float(dd_series.min()) if len(dd_series) else np.nan

    calmar = ann_ret / (abs(max_dd) + eps) if np.isfinite(max_dd) and max_dd != 0 else np.nan

    win_rate = float((daily > 0).mean()) if len(daily) else np.nan

    gains = float(daily[daily > 0].sum())
    losses = float(abs(daily[daily < 0].sum()))
    profit_factor = gains / (losses + eps)

    avg_win = float(daily[daily > 0].mean()) if (daily > 0).any() else 0.0
    avg_loss = float(daily[daily < 0].mean()) if (daily < 0).any() else 0.0

    best_day = float(daily.max()) if len(daily) else np.nan
    worst_day = float(daily.min()) if len(daily) else np.nan

    monthly = daily.resample("ME").sum() if len(daily) > 20 else daily
    best_month = float(monthly.max()) if len(monthly) else np.nan
    worst_month = float(monthly.min()) if len(monthly) else np.nan

    w_last = weights.iloc[-1]
    n_positions = int((np.abs(w_last.values) > eps).sum())

    long_exposure = float(w_last[w_last > 0].sum())
    short_exposure = float(abs(w_last[w_last < 0].sum()))
    gross_exposure = long_exposure + short_exposure
    net_exposure = long_exposure - short_exposure

    sorted_w = np.sort(np.abs(w_last.values))[::-1]
    top5_concentration = (
        float(sorted_w[:5].sum() / (gross_exposure + eps)) if gross_exposure > eps else 0.0
    )
    top10_concentration = (
        float(sorted_w[:10].sum() / (gross_exposure + eps)) if gross_exposure > eps else 0.0
    )

    turnover_series = weights.fillna(0.0).diff().abs().sum(axis=1)
    avg_turnover = float(turnover_series.mean())

    # Transaction cost metrics
    # Legacy flat BPS estimate
    est_annual_tc_drag_flat = avg_turnover * 252.0 * (tc_bps / 10000.0)
    net_of_tc_flat = ann_ret - est_annual_tc_drag_flat
    sharpe_net_flat = net_of_tc_flat / (ann_vol + eps)
    
    # ATR-based TC (from diag if available)
    if tc_cost_atr is not None:
        est_annual_tc_drag_atr = float(tc_cost_atr.mean() * 252.0)
        net_of_tc_atr = ann_ret - est_annual_tc_drag_atr
        sharpe_net_atr = net_of_tc_atr / (ann_vol + eps)
    else:
        # Estimate using typical ATR (1.5% of price) * multiplier (5%)
        if tc_config is not None and tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
            typical_atr_pct = 0.015  # 1.5%
            est_tc_per_trade = tc_config.atr_multiplier * typical_atr_pct
            est_annual_tc_drag_atr = avg_turnover * 252.0 * est_tc_per_trade
        else:
            est_annual_tc_drag_atr = est_annual_tc_drag_flat
        net_of_tc_atr = ann_ret - est_annual_tc_drag_atr
        sharpe_net_atr = net_of_tc_atr / (ann_vol + eps)
    
    # Use ATR as primary if config specifies it
    if tc_config is not None and tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
        est_annual_tc_drag = est_annual_tc_drag_atr
        net_of_tc = net_of_tc_atr
        sharpe_net = sharpe_net_atr
        tc_scheme = "quantiacs_atr"
    else:
        est_annual_tc_drag = est_annual_tc_drag_flat
        net_of_tc = net_of_tc_flat
        sharpe_net = sharpe_net_flat
        tc_scheme = "flat_bps"

    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sharpe_net_tc": sharpe_net,
        "sharpe_net_flat": sharpe_net_flat,
        "sharpe_net_atr": sharpe_net_atr,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "current_drawdown": float(dd_series.iloc[-1]) if len(dd_series) else 0.0,
        "volatility": ann_vol,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_day": best_day,
        "worst_day": worst_day,
        "best_month": best_month,
        "worst_month": worst_month,
        "n_positions": n_positions,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "top5_concentration": top5_concentration,
        "top10_concentration": top10_concentration,
        "avg_turnover": avg_turnover,
        "est_annual_tc_drag": est_annual_tc_drag,
        "est_annual_tc_drag_flat": est_annual_tc_drag_flat,
        "est_annual_tc_drag_atr": est_annual_tc_drag_atr,
        "net_return_after_tc": net_of_tc,
        "net_return_after_tc_flat": net_of_tc_flat,
        "net_return_after_tc_atr": net_of_tc_atr,
        "tc_scheme": tc_scheme,
    }


def format_performance_table(perf: Dict[str, Any], show_tc_comparison: bool = True) -> pd.DataFrame:
    """Format performance metrics as a display table.
    
    Args:
        perf: Performance metrics dictionary
        show_tc_comparison: Whether to show both flat and ATR TC metrics
        
    Returns:
        DataFrame formatted for display
    """
    # Get TC scheme
    tc_scheme = perf.get("tc_scheme", "flat_bps")
    tc_label = "Quantiacs ATR" if tc_scheme == "quantiacs_atr" else "Flat BPS"
    
    # Build sections
    sections = {
        "Risk-Adjusted Returns": [
            ("Annual Return", perf["annual_return"], ".2%"),
            (f"Net of TC ({tc_label})", perf["net_return_after_tc"], ".2%"),
            ("Annual Volatility", perf["annual_vol"], ".2%"),
            ("Sharpe Ratio (gross)", perf["sharpe"], ".3f"),
            (f"Sharpe (net {tc_label})", perf["sharpe_net_tc"], ".3f"),
            ("Sortino Ratio", perf["sortino"], ".3f"),
            ("Calmar Ratio", perf["calmar"], ".3f"),
        ],
        "Drawdown Analysis": [
            ("Maximum Drawdown", perf["max_drawdown"], ".2%"),
            ("Current Drawdown", perf["current_drawdown"], ".2%"),
            ("Worst Day", perf["worst_day"], ".2%"),
            ("Worst Month", perf["worst_month"], ".2%"),
        ],
        "Win/Loss Statistics": [
            ("Win Rate", perf["win_rate"], ".1%"),
            ("Profit Factor", perf["profit_factor"], ".2f"),
            ("Average Win", perf["avg_win"], ".3%"),
            ("Average Loss", perf["avg_loss"], ".3%"),
            ("Best Day", perf["best_day"], ".2%"),
            ("Best Month", perf["best_month"], ".2%"),
        ],
        "Portfolio Snapshot": [
            ("Positions", perf["n_positions"], ".0f"),
            ("Long Exposure", perf["long_exposure"], ".1%"),
            ("Short Exposure", perf["short_exposure"], ".1%"),
            ("Gross Exposure", perf["gross_exposure"], ".1%"),
            ("Net Exposure", perf["net_exposure"], ".1%"),
            ("Top 5 Concentration", perf["top5_concentration"], ".1%"),
            ("Top 10 Concentration", perf["top10_concentration"], ".1%"),
        ],
        "Trading Activity": [
            ("Avg Daily Turnover", perf["avg_turnover"], ".2%"),
            (f"Est. Annual TC ({tc_label})", perf["est_annual_tc_drag"], ".2%"),
        ],
    }
    
    # Add TC comparison section if requested
    if show_tc_comparison and "sharpe_net_flat" in perf and "sharpe_net_atr" in perf:
        sections["Transaction Cost Comparison"] = [
            ("TC Model", tc_scheme, "s"),
            ("Sharpe (Flat BPS)", perf["sharpe_net_flat"], ".3f"),
            ("Sharpe (Quantiacs ATR)", perf["sharpe_net_atr"], ".3f"),
            ("TC Drag (Flat)", perf.get("est_annual_tc_drag_flat", 0.0), ".2%"),
            ("TC Drag (ATR)", perf.get("est_annual_tc_drag_atr", 0.0), ".2%"),
        ]

    rows = []
    for section_name, metrics in sections.items():
        rows.append({"Section": section_name, "Metric": "", "Value": ""})
        for metric_name, value, fmt in metrics:
            if isinstance(value, str):
                formatted = value
            elif np.isfinite(value) if isinstance(value, (int, float)) else False:
                formatted = f"{value:{fmt}}"
            else:
                formatted = "—"
            rows.append({"Section": "", "Metric": metric_name, "Value": formatted})

    return pd.DataFrame(rows)


def compute_tc_comparison(
    weights: pd.DataFrame,
    daily_returns: pd.Series,
    tc_bps_values: Optional[list] = None,
) -> pd.DataFrame:
    """Compute performance comparison across different TC assumptions.
    
    Args:
        weights: Portfolio weights
        daily_returns: Portfolio daily returns
        tc_bps_values: List of TC bps values to compare
        
    Returns:
        DataFrame comparing metrics at each TC level
    """
    if tc_bps_values is None:
        tc_bps_values = [0, 5, 10, 15, 20, 30]
    
    eps = 1e-12
    turnover = weights.fillna(0.0).diff().abs().sum(axis=1)
    
    results = []
    for bps in tc_bps_values:
        tc_cost = turnover * (bps / 10000.0)
        net_ret = daily_returns - tc_cost.reindex(daily_returns.index).fillna(0.0)
        
        ann_ret = float(net_ret.mean() * 252)
        ann_vol = float(net_ret.std() * np.sqrt(252))
        sharpe = ann_ret / (ann_vol + eps)
        
        eq = (1 + net_ret).cumprod()
        max_dd = float((eq / eq.cummax() - 1).min())
        
        results.append({
            "tc_bps": bps,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "annual_tc_drag": float(tc_cost.mean() * 252),
        })
    
    return pd.DataFrame(results)
