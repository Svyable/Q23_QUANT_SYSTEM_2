"""
Stock-Level Analytics

Functions for analyzing individual stocks and comparing multiple stocks.
Provides performance metrics, factor exposures, attribution, and diagnostics.

PM Best Practices:
- All metrics use consistent time periods for fair comparison
- Attribution tracks both factor-driven and idiosyncratic components
- Diagnostics highlight concentration and turnover risks
- Performance metrics align with industry standards (annualized, risk-adjusted)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from q23.dashboard.analytics.performance import compute_comprehensive_performance
from q23.dashboard.analytics.attribution import compute_single_stock_attribution
from q23.dashboard.analytics.metrics import select_return_series


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class StockSummary:
    """Summary metrics for a single stock."""
    symbol: str
    current_weight: float
    days_held: int
    total_pnl: float
    pnl_per_day: float
    avg_weight: float
    avg_weight_when_held: float
    side: str  # "LONG", "SHORT", or "FLAT"
    
    @classmethod
    def from_weights_and_factors(
        cls,
        symbol: str,
        weights: pd.Series,
        factor_vectors: Optional[pd.DataFrame],
    ) -> "StockSummary":
        """Create summary from weights and factor vectors."""
        held_mask = weights.abs() > 1e-12
        current = float(weights.iloc[-1]) if len(weights) > 0 else 0.0
        
        # Get P&L from factor_vectors if available
        total_pnl = 0.0
        pnl_per_day = 0.0
        if factor_vectors is not None and symbol in factor_vectors.index:
            row = factor_vectors.loc[symbol]
            total_pnl = float(row.get("total_pnl_contrib", 0.0))
            pnl_per_day = float(row.get("pnl_per_day_held", 0.0))
        
        # Determine side
        if current > 1e-12:
            side = "LONG"
        elif current < -1e-12:
            side = "SHORT"
        else:
            side = "FLAT"
        
        return cls(
            symbol=symbol,
            current_weight=current,
            days_held=int(held_mask.sum()),
            total_pnl=total_pnl,
            pnl_per_day=pnl_per_day,
            avg_weight=float(weights.mean()),
            avg_weight_when_held=float(weights[held_mask].mean()) if held_mask.any() else 0.0,
            side=side,
        )


@dataclass
class TradeRecord:
    """Record of a single trade (entry or exit)."""
    date: pd.Timestamp
    action: str  # "ENTRY", "EXIT", "INCREASE", "DECREASE"
    weight_before: float
    weight_after: float
    weight_change: float


# =============================================================================
# Stock Discovery and Filtering
# =============================================================================

def get_available_stocks(weights: pd.DataFrame) -> List[str]:
    """Get list of stocks that have been held in the portfolio.
    
    Args:
        weights: Portfolio weights (time x asset)
        
    Returns:
        List of stock symbols that have non-zero weights at some point
    """
    if weights is None or weights.empty:
        return []
    
    # Find stocks with any non-zero weight
    has_position = (weights.abs() > 1e-12).any(axis=0)
    stocks = weights.columns[has_position].tolist()
    return sorted(stocks)


def get_current_holdings(weights: pd.DataFrame) -> List[str]:
    """Get stocks currently held in the portfolio.
    
    Args:
        weights: Portfolio weights (time x asset)
        
    Returns:
        List of stock symbols with non-zero current weight
    """
    if weights is None or weights.empty:
        return []
    
    w_last = weights.iloc[-1]
    current = w_last[w_last.abs() > 1e-12].index.tolist()
    return sorted(current)


def filter_stocks_by_criteria(
    weights: pd.DataFrame,
    factor_vectors: Optional[pd.DataFrame] = None,
    min_days_held: int = 0,
    min_pnl: Optional[float] = None,
    max_pnl: Optional[float] = None,
    side: Optional[str] = None,  # "LONG", "SHORT", or None for all
    current_only: bool = False,
) -> List[str]:
    """Filter stocks by various criteria (stock screener).
    
    Args:
        weights: Portfolio weights (time x asset)
        factor_vectors: Factor vectors with summary metrics
        min_days_held: Minimum days the stock was held
        min_pnl: Minimum total P&L contribution
        max_pnl: Maximum total P&L contribution
        side: Filter by current side ("LONG", "SHORT", or None)
        current_only: Only include stocks currently held
        
    Returns:
        List of stock symbols matching criteria
    """
    if weights is None or weights.empty:
        return []
    
    candidates = get_available_stocks(weights)
    if current_only:
        current = set(get_current_holdings(weights))
        candidates = [s for s in candidates if s in current]
    
    filtered = []
    
    for symbol in candidates:
        w_series = weights[symbol].fillna(0.0)
        held_mask = w_series.abs() > 1e-12
        days_held = int(held_mask.sum())
        
        # Days held filter
        if days_held < min_days_held:
            continue
        
        # Side filter
        current_weight = float(w_series.iloc[-1])
        if side == "LONG" and current_weight <= 1e-12:
            continue
        if side == "SHORT" and current_weight >= -1e-12:
            continue
        
        # P&L filters
        if factor_vectors is not None and symbol in factor_vectors.index:
            pnl = float(factor_vectors.loc[symbol].get("total_pnl_contrib", 0.0))
            if min_pnl is not None and pnl < min_pnl:
                continue
            if max_pnl is not None and pnl > max_pnl:
                continue
        
        filtered.append(symbol)
    
    return filtered


def rank_stocks_by_metric(
    weights: pd.DataFrame,
    factor_vectors: Optional[pd.DataFrame],
    metric: str = "total_pnl_contrib",
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Rank stocks by a given metric.
    
    Args:
        weights: Portfolio weights (time x asset)
        factor_vectors: Factor vectors with summary metrics
        metric: Metric to rank by (from factor_vectors or computed)
        ascending: Sort ascending (True) or descending (False)
        top_n: Return only top N stocks
        
    Returns:
        List of (symbol, metric_value) tuples sorted by metric
    """
    stocks = get_available_stocks(weights)
    
    if not stocks:
        return []
    
    rankings = []
    
    for symbol in stocks:
        value = 0.0
        
        if metric in ["current_weight", "days_held", "avg_weight"]:
            # Computed from weights
            w_series = weights[symbol].fillna(0.0)
            if metric == "current_weight":
                value = float(w_series.iloc[-1])
            elif metric == "days_held":
                value = float((w_series.abs() > 1e-12).sum())
            elif metric == "avg_weight":
                value = float(w_series.mean())
        elif factor_vectors is not None and symbol in factor_vectors.index:
            value = float(factor_vectors.loc[symbol].get(metric, 0.0))
        
        rankings.append((symbol, value))
    
    # Sort
    rankings.sort(key=lambda x: x[1], reverse=not ascending)
    
    if top_n is not None:
        rankings = rankings[:top_n]
    
    return rankings


# =============================================================================
# Stock Returns Computation
# =============================================================================

def compute_stock_returns_from_weights(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    stock_symbol: str,
) -> Optional[pd.Series]:
    """
    Infer stock returns from weight changes and portfolio contribution.
    
    WARNING: This is an approximation. For accurate returns, stock prices/returns
    should be saved in artifacts. This method provides estimates only.
    
    The method uses: contribution[t] = w[t-1] * r_stock[t]
    So: r_stock[t] ≈ contribution[t] / w[t-1] (when w[t-1] > 0)
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns (time)
        stock_symbol: Stock symbol to analyze
        
    Returns:
        Series of inferred stock returns, or None if stock not found
    """
    if stock_symbol not in weights.columns:
        return None
    
    w_series = weights[stock_symbol].fillna(0.0)
    
    # Align weights and portfolio returns
    w_series, port_ret = w_series.align(portfolio_returns, join='inner')
    
    if w_series.empty:
        return None
    
    # Lag weights (position at start of day) - VECTORIZED
    w_lag = w_series.shift(1).fillna(0.0)
    
    # Vectorized computation: r_stock ≈ port_ret / w_lag (where w_lag is significant)
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    
    # Create mask for significant positions
    significant_mask = w_lag.abs() > eps
    
    # Initialize with zeros
    stock_returns = pd.Series(0.0, index=w_series.index)
    
    # Vectorized division only where we have positions
    # Note: This is a crude approximation - it assumes the stock's contribution
    # is proportional to its weight, which is only accurate if the portfolio
    # consists solely of this stock
    stock_returns[significant_mask] = port_ret[significant_mask] / (w_lag[significant_mask] + eps)
    
    # Clip extreme values (more than 50% daily move is likely noise)
    stock_returns = stock_returns.clip(-0.5, 0.5)
    
    return stock_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def compute_stock_contribution(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    stock_symbol: str,
) -> pd.Series:
    """
    Compute the stock's contribution to portfolio returns.
    
    contribution[t] = w[t-1] * r_portfolio[t] * (w[t-1] / sum(abs(w[t-1])))
    
    This is more accurate than inferring stock returns directly.
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns (time)
        stock_symbol: Stock symbol
        
    Returns:
        Series of contribution values
    """
    if stock_symbol not in weights.columns:
        return pd.Series(dtype=float)
    
    w_series = weights[stock_symbol].fillna(0.0)
    w_series, port_ret = w_series.align(portfolio_returns, join='inner')
    
    if w_series.empty:
        return pd.Series(dtype=float)
    
    # Lag weights
    w_lag = w_series.shift(1).fillna(0.0)
    
    # Compute total absolute weight (gross exposure)
    gross = weights.abs().sum(axis=1).shift(1).fillna(1.0)
    gross, _ = gross.align(w_series, join='inner')
    
    # Contribution = (w_lag / gross) * port_ret
    # This represents the portion of portfolio return attributable to this stock
    weight_ratio = w_lag / (gross + 1e-12)
    contribution = weight_ratio * port_ret
    
    return contribution.fillna(0.0)


def compute_stock_performance(
    stock_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics for a stock.
    
    Args:
        stock_returns: Stock daily returns (time)
        benchmark_returns: Optional benchmark returns for comparison
        weights: Optional portfolio weights for this stock (for context)
        
    Returns:
        Dictionary of performance metrics
    """
    if stock_returns.empty:
        return {}
    
    # Use existing performance function but adapt for single stock
    # Create a dummy weights DataFrame with single stock
    dummy_weights = stock_returns.to_frame("stock").abs()  # Use abs returns as proxy weights
    dummy_weights = dummy_weights / dummy_weights.sum(axis=1).replace(0, 1)  # Normalize
    
    # Create diagnostics with stock returns
    diag = pd.DataFrame({
        "port_ret": stock_returns,
        "active_ret": stock_returns if benchmark_returns is None else stock_returns - benchmark_returns,
    }, index=stock_returns.index)
    
    # Compute performance using existing function
    perf = compute_comprehensive_performance(
        weights=dummy_weights,
        diag=diag,
    )
    
    # Add stock-specific metrics
    if weights is not None:
        held_mask = weights.abs() > 1e-12
        perf["days_held"] = float(held_mask.sum())
        perf["avg_weight"] = float(weights.mean())
        perf["avg_weight_when_held"] = float(weights[held_mask].mean()) if held_mask.any() else 0.0
    else:
        perf["days_held"] = float(len(stock_returns))
        perf["avg_weight"] = 0.0
        perf["avg_weight_when_held"] = 0.0
    
    # Add benchmark comparison if available
    if benchmark_returns is not None:
        aligned_stock, aligned_bench = stock_returns.align(benchmark_returns, join='inner')
        if not aligned_stock.empty:
            excess_returns = aligned_stock - aligned_bench
            perf["excess_return"] = float(excess_returns.mean() * 252)
            perf["tracking_error"] = float(excess_returns.std() * np.sqrt(252))
            perf["information_ratio"] = perf["excess_return"] / (perf["tracking_error"] + 1e-12)
            
            # Beta
            if aligned_bench.std() > 1e-12:
                cov = np.cov(aligned_stock, aligned_bench)[0, 1]
                perf["beta"] = float(cov / (aligned_bench.std() ** 2))
            else:
                perf["beta"] = 0.0
    
    return perf


def compute_stock_factor_exposure(
    stock_symbol: str,
    factor_vectors: pd.DataFrame,
) -> Dict[str, float]:
    """
    Get factor exposures for a single stock.
    
    Args:
        stock_symbol: Stock symbol
        factor_vectors: DataFrame with factor exposures (asset x factors)
        
    Returns:
        Dictionary of factor exposures
    """
    if factor_vectors is None or factor_vectors.empty:
        return {}
    
    if stock_symbol not in factor_vectors.index:
        return {}
    
    # Get factor columns (exclude summary columns)
    summary_cols = {
        "total_pnl_contrib",
        "pnl_per_day_held",
        "mean_score",
        "score_vol",
        "days_held",
        "avg_weight",
        "avg_weight_when_held",
    }
    
    factor_cols = [c for c in factor_vectors.columns if c not in summary_cols]
    
    row = factor_vectors.loc[stock_symbol]
    exposures = {}
    
    for col in factor_cols:
        if pd.api.types.is_numeric_dtype(row[col]):
            exposures[col] = float(row[col])
    
    return exposures


def compare_stocks(
    stock_symbols: List[str],
    weights: pd.DataFrame,
    factor_vectors: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
    stock_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare multiple stocks across key metrics.
    
    Args:
        stock_symbols: List of stock symbols to compare
        weights: Portfolio weights (time x asset)
        factor_vectors: Factor vectors (asset x factors + summary)
        portfolio_returns: Optional portfolio returns for context
        stock_returns: Optional stock returns DataFrame (time x asset)
        
    Returns:
        DataFrame with stocks as rows, metrics as columns
    """
    if not stock_symbols:
        return pd.DataFrame()
    
    results = []
    
    for symbol in stock_symbols:
        if symbol not in weights.columns:
            continue
        
        # Get weight series
        w_series = weights[symbol].fillna(0.0)
        
        # Get stock returns if available
        if stock_returns is not None and symbol in stock_returns.columns:
            stock_ret = stock_returns[symbol]
        elif portfolio_returns is not None:
            stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        else:
            stock_ret = None
        
        # Compute performance
        perf = {}
        if stock_ret is not None and not stock_ret.empty:
            perf = compute_stock_performance(stock_ret, weights=w_series)
        
        # Get attribution if available
        if factor_vectors is not None and symbol in factor_vectors.index:
            row = factor_vectors.loc[symbol]
            perf["total_pnl_contrib"] = float(row.get("total_pnl_contrib", 0.0))
            perf["pnl_per_day_held"] = float(row.get("pnl_per_day_held", 0.0))
            perf["mean_score"] = float(row.get("mean_score", 0.0))
            perf["days_held"] = int(row.get("days_held", 0))
            perf["avg_weight"] = float(row.get("avg_weight", 0.0))
        
        # Current weight
        perf["current_weight"] = float(w_series.iloc[-1]) if len(w_series) > 0 else 0.0
        
        # Add symbol
        perf["symbol"] = symbol
        
        results.append(perf)
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Set symbol as index
    if "symbol" in df.columns:
        df = df.set_index("symbol")
    
    # Reorder columns for better display
    priority_cols = [
        "current_weight",
        "days_held",
        "avg_weight",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "total_pnl_contrib",
        "pnl_per_day_held",
        "mean_score",
    ]
    
    # Get available priority columns
    available_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    
    df = df[available_priority + other_cols]
    
    return df


# =============================================================================
# Stock Diagnostics and Trade Analysis
# =============================================================================

def get_stock_diagnostics(
    stock_symbol: str,
    weights: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
    stock_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive diagnostics for a single stock.
    
    Args:
        stock_symbol: Stock symbol
        weights: Portfolio weights (time x asset)
        portfolio_returns: Optional portfolio returns
        stock_returns: Optional stock returns series
        
    Returns:
        Dictionary with diagnostic information
    """
    if stock_symbol not in weights.columns:
        return {}
    
    w_series = weights[stock_symbol].fillna(0.0)
    
    # Entry/exit dates
    w_diff = w_series.diff()
    entries = w_diff[w_diff.abs() > 1e-12].index.tolist()
    
    # Position changes with action classification
    position_changes = []
    for date in entries:
        change = w_diff.loc[date]
        prev_weight = float(w_series.shift(1).loc[date]) if date in w_series.index else 0.0
        new_weight = float(w_series.loc[date])
        
        # Classify action
        if abs(prev_weight) < 1e-12 and abs(new_weight) > 1e-12:
            action = "ENTRY"
        elif abs(prev_weight) > 1e-12 and abs(new_weight) < 1e-12:
            action = "EXIT"
        elif abs(new_weight) > abs(prev_weight):
            action = "INCREASE"
        else:
            action = "DECREASE"
        
        position_changes.append({
            "date": date,
            "action": action,
            "weight_before": prev_weight,
            "weight_after": new_weight,
            "weight_change": float(change),
        })
    
    # Days held
    held_mask = w_series.abs() > 1e-12
    days_held = int(held_mask.sum())
    
    # Weight statistics
    weight_stats = {
        "current": float(w_series.iloc[-1]),
        "max": float(w_series.abs().max()),
        "min_nonzero": float(w_series[held_mask].abs().min()) if held_mask.any() else 0.0,
        "mean": float(w_series.mean()),
        "mean_when_held": float(w_series[held_mask].mean()) if held_mask.any() else 0.0,
        "std_when_held": float(w_series[held_mask].std()) if held_mask.any() else 0.0,
    }
    
    # Correlation with portfolio
    correlation = None
    if stock_returns is not None and portfolio_returns is not None:
        aligned_stock, aligned_port = stock_returns.align(portfolio_returns, join='inner')
        if len(aligned_stock) > 10:  # Need sufficient data
            correlation = float(aligned_stock.corr(aligned_port))
    
    # Turnover contribution
    turnover_series = w_series.diff().abs()
    avg_turnover = float(turnover_series.mean())
    total_turnover = float(turnover_series.sum())
    
    # Trade statistics
    n_entries = sum(1 for p in position_changes if p["action"] == "ENTRY")
    n_exits = sum(1 for p in position_changes if p["action"] == "EXIT")
    n_changes = len(position_changes)
    
    diagnostics = {
        "symbol": stock_symbol,
        "days_held": days_held,
        "position_changes": position_changes,
        "weight_stats": weight_stats,
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "correlation_with_portfolio": correlation,
        "n_entries": n_entries,
        "n_exits": n_exits,
        "n_position_changes": n_changes,
    }
    
    return diagnostics


def analyze_trades(
    stock_symbol: str,
    weights: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze individual trades (entry to exit) for a stock.
    
    Each trade record includes:
    - Entry date, exit date
    - Holding period
    - P&L (if returns available)
    - Max weight during trade
    
    Args:
        stock_symbol: Stock symbol
        weights: Portfolio weights (time x asset)
        portfolio_returns: Optional portfolio returns for P&L
        
    Returns:
        List of trade dictionaries
    """
    if stock_symbol not in weights.columns:
        return []
    
    w_series = weights[stock_symbol].fillna(0.0)
    held_mask = w_series.abs() > 1e-12
    
    # Find trade boundaries
    trades = []
    in_trade = False
    entry_date = None
    entry_weight = 0.0
    max_weight = 0.0
    
    for date in w_series.index:
        current_weight = w_series.loc[date]
        
        if not in_trade and abs(current_weight) > 1e-12:
            # New trade entry
            in_trade = True
            entry_date = date
            entry_weight = current_weight
            max_weight = abs(current_weight)
        elif in_trade:
            max_weight = max(max_weight, abs(current_weight))
            
            if abs(current_weight) < 1e-12:
                # Trade exit
                exit_date = date
                holding_days = (exit_date - entry_date).days
                
                # Compute P&L if we have returns
                trade_pnl = 0.0
                if portfolio_returns is not None:
                    contribution = compute_stock_contribution(weights, portfolio_returns, stock_symbol)
                    mask = (contribution.index >= entry_date) & (contribution.index <= exit_date)
                    trade_pnl = float(contribution[mask].sum())
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "holding_days": holding_days,
                    "entry_weight": float(entry_weight),
                    "max_weight": float(max_weight),
                    "side": "LONG" if entry_weight > 0 else "SHORT",
                    "pnl": trade_pnl,
                })
                
                in_trade = False
                entry_date = None
                entry_weight = 0.0
                max_weight = 0.0
    
    # Handle open trade
    if in_trade and entry_date is not None:
        exit_date = w_series.index[-1]
        holding_days = (exit_date - entry_date).days
        
        trade_pnl = 0.0
        if portfolio_returns is not None:
            contribution = compute_stock_contribution(weights, portfolio_returns, stock_symbol)
            mask = (contribution.index >= entry_date) & (contribution.index <= exit_date)
            trade_pnl = float(contribution[mask].sum())
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": None,  # Still open
            "holding_days": holding_days,
            "entry_weight": float(entry_weight),
            "max_weight": float(max_weight),
            "side": "LONG" if entry_weight > 0 else "SHORT",
            "pnl": trade_pnl,
            "is_open": True,
        })
    
    return trades


def compute_stock_rolling_metrics(
    stock_returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling performance metrics for a stock.
    
    Args:
        stock_returns: Stock daily returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling metrics (time x metric)
    """
    if stock_returns is None or stock_returns.empty:
        return pd.DataFrame()
    
    metrics = pd.DataFrame(index=stock_returns.index)
    
    # Rolling return (annualized)
    metrics["rolling_return"] = stock_returns.rolling(window).mean() * 252
    
    # Rolling volatility (annualized)
    metrics["rolling_vol"] = stock_returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe (using 0 as risk-free rate)
    metrics["rolling_sharpe"] = metrics["rolling_return"] / (metrics["rolling_vol"] + 1e-12)
    
    # Rolling win rate
    metrics["rolling_win_rate"] = stock_returns.rolling(window).apply(lambda x: (x > 0).mean())
    
    # Rolling max drawdown
    def rolling_max_dd(returns):
        cum = (1 + returns).cumprod()
        dd = cum / cum.cummax() - 1
        return dd.min()
    
    metrics["rolling_max_dd"] = stock_returns.rolling(window).apply(rolling_max_dd)
    
    return metrics.dropna()


# =============================================================================
# Multi-Stock Analysis
# =============================================================================

def compute_stock_correlation_matrix(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    stock_symbols: List[str],
) -> pd.DataFrame:
    """
    Compute correlation matrix between multiple stocks.
    
    OPTIMIZED: Uses vectorized operations to compute all stock returns at once
    instead of looping, which is much faster for large stock lists.
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns
        stock_symbols: List of stock symbols
        
    Returns:
        Correlation matrix DataFrame (stock x stock)
    """
    if not stock_symbols or len(stock_symbols) < 2:
        return pd.DataFrame()
    
    # Filter to only stocks that exist in weights
    valid_symbols = [s for s in stock_symbols if s in weights.columns]
    if len(valid_symbols) < 2:
        return pd.DataFrame()
    
    # Vectorized computation: get all stock weights at once
    stock_weights = weights[valid_symbols].fillna(0.0)
    
    # Align with portfolio returns
    stock_weights, port_ret = stock_weights.align(portfolio_returns, join='inner', axis=0)
    
    if stock_weights.empty:
        return pd.DataFrame()
    
    # Lag weights (vectorized for all stocks at once)
    w_lag = stock_weights.shift(1).fillna(0.0)
    
    # Vectorized stock returns computation for all stocks at once
    eps = 1e-12
    significant_mask = w_lag.abs() > eps
    
    # Vectorized division: compute returns for all symbols simultaneously
    # Use numpy broadcasting: port_ret[:, None] broadcasts to (n_dates, n_symbols)
    port_ret_2d = port_ret.values[:, np.newaxis]  # Shape: (n_dates, 1)
    w_lag_values = w_lag.values  # Shape: (n_dates, n_symbols)
    
    # Vectorized division with where parameter to handle division by zero
    stock_returns_values = np.divide(
        port_ret_2d,
        w_lag_values + eps,
        out=np.zeros_like(w_lag_values, dtype=float),
        where=significant_mask.values
    )
    
    # Create DataFrame from computed values
    stock_returns = pd.DataFrame(
        stock_returns_values,
        index=stock_weights.index,
        columns=valid_symbols
    )
    
    # Clip extreme values
    stock_returns = stock_returns.clip(-0.5, 0.5)
    stock_returns = stock_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # Remove stocks with insufficient data
    valid_cols = [col for col in stock_returns.columns if stock_returns[col].abs().sum() > eps]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    
    stock_returns = stock_returns[valid_cols]
    
    # Compute correlation
    return stock_returns.corr()


def generate_stock_alerts(
    stock_symbol: str,
    weights: pd.DataFrame,
    factor_vectors: Optional[pd.DataFrame],
    portfolio_returns: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """
    Generate PM alerts for a stock.
    
    Alerts include:
    - High concentration warnings
    - Unusual turnover
    - Large P&L impact
    - Factor exposure changes
    
    Args:
        stock_symbol: Stock symbol
        weights: Portfolio weights (time x asset)
        factor_vectors: Factor vectors with summary metrics
        portfolio_returns: Optional portfolio returns
        
    Returns:
        List of alert dictionaries with severity, message, and details
    """
    alerts = []
    
    if stock_symbol not in weights.columns:
        return alerts
    
    w_series = weights[stock_symbol].fillna(0.0)
    current_weight = float(w_series.iloc[-1])
    
    # Concentration alert
    if abs(current_weight) > 0.10:  # More than 10%
        alerts.append({
            "severity": "warning",
            "category": "Concentration",
            "message": f"High concentration: {abs(current_weight):.1%} of portfolio",
            "details": "Consider position sizing limits",
        })
    
    # Recent turnover alert
    recent_turnover = float(w_series.diff().abs().tail(5).mean())
    avg_turnover = float(w_series.diff().abs().mean())
    if recent_turnover > avg_turnover * 2 and avg_turnover > 0:
        alerts.append({
            "severity": "info",
            "category": "Turnover",
            "message": f"Elevated recent turnover: {recent_turnover:.2%} (avg: {avg_turnover:.2%})",
            "details": "Review recent position changes",
        })
    
    # P&L impact alert
    if factor_vectors is not None and stock_symbol in factor_vectors.index:
        total_pnl = float(factor_vectors.loc[stock_symbol].get("total_pnl_contrib", 0.0))
        
        if total_pnl < -0.01:  # Significant loss
            alerts.append({
                "severity": "warning",
                "category": "P&L",
                "message": f"Significant loss: {total_pnl:.4f} total P&L contribution",
                "details": "Review position thesis",
            })
        elif total_pnl > 0.02:  # Strong winner
            alerts.append({
                "severity": "info",
                "category": "P&L",
                "message": f"Strong performer: {total_pnl:.4f} total P&L contribution",
                "details": "Consider taking profits",
            })
    
    return alerts


def compute_stock_attribution_enhanced(
    stock_symbol: str,
    weights: pd.DataFrame,
    factor_vectors: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
    stock_returns: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """
    Enhanced stock attribution combining multiple data sources.
    
    Args:
        stock_symbol: Stock symbol
        weights: Portfolio weights (time x asset)
        factor_vectors: Factor vectors (asset x factors + summary)
        portfolio_returns: Optional portfolio returns
        stock_returns: Optional stock returns
        
    Returns:
        Dictionary with comprehensive attribution
    """
    # Use existing function
    if stock_returns is not None:
        returns_df = stock_returns.to_frame(stock_symbol)
    else:
        returns_df = None
    
    attribution = compute_single_stock_attribution(
        symbol=stock_symbol,
        factor_vectors=factor_vectors if factor_vectors is not None else pd.DataFrame(),
        weights=weights,
        returns=returns_df if returns_df is not None else pd.DataFrame(),
    )
    
    # Add diagnostics
    diagnostics = get_stock_diagnostics(
        stock_symbol=stock_symbol,
        weights=weights,
        portfolio_returns=portfolio_returns,
        stock_returns=stock_returns,
    )
    
    attribution.update(diagnostics)
    
    return attribution


# =============================================================================
# Batch Stock Metrics for Treemap
# =============================================================================

def compute_batch_stock_metrics(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    factor_vectors: Optional[pd.DataFrame] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Efficiently compute quick metrics for multiple stocks (for treemap tooltips).
    
    This function is optimized for speed when computing metrics for many stocks,
    using vectorized operations where possible.
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns
        factor_vectors: Optional factor vectors with summary metrics
        symbols: Optional list of symbols to compute (defaults to current holdings)
        
    Returns:
        Dict of {symbol: metrics_dict} where metrics_dict contains:
        - sharpe, sortino: Risk-adjusted returns
        - var_95, max_dd: Risk metrics
        - skewness, kurtosis: Distribution metrics
        - ret_30d, pnl_contrib: Performance metrics
        - days_held, ann_vol, hit_rate: Context metrics
        - beta: Market sensitivity (if portfolio returns available)
        - n_obs: Number of observations
    """
    from scipy import stats
    
    if weights is None or weights.empty:
        return {}
    
    # Default to current holdings
    if symbols is None:
        w_last = weights.iloc[-1]
        symbols = w_last[w_last.abs() > 1e-12].index.tolist()
    
    if not symbols:
        return {}
    
    results = {}
    
    for symbol in symbols:
        if symbol not in weights.columns:
            continue
        
        metrics = {}
        w_series = weights[symbol].fillna(0.0)
        
        # Basic weight metrics
        held_mask = w_series.abs() > 1e-12
        days_held = int(held_mask.sum())
        metrics['days_held'] = days_held
        
        # Get stock returns (approximation)
        stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        
        if stock_ret is not None and len(stock_ret) >= 20:
            rets = stock_ret.dropna()
            rets = rets[held_mask.reindex(rets.index, fill_value=False)]  # Only when held
            
            if len(rets) >= 20:
                n_obs = len(rets)
                metrics['n_obs'] = n_obs
                
                # Returns
                ann_ret = float(rets.mean() * 252)
                ann_vol = float(rets.std() * np.sqrt(252))
                metrics['ann_vol'] = ann_vol
                
                # Sharpe
                sharpe = ann_ret / (ann_vol + 1e-12)
                metrics['sharpe'] = sharpe
                
                # Sortino
                downside = rets[rets < 0]
                downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
                sortino = ann_ret / (downside_std + 1e-12)
                metrics['sortino'] = sortino
                
                # VaR 95%
                var_95 = float(np.percentile(rets, 5))
                metrics['var_95'] = var_95
                
                # Max Drawdown
                cum = (1 + rets).cumprod()
                drawdown = cum / cum.cummax() - 1
                max_dd = float(drawdown.min())
                metrics['max_dd'] = max_dd
                
                # Distribution moments
                if len(rets) >= 30:
                    metrics['skewness'] = float(stats.skew(rets))
                    metrics['kurtosis'] = float(stats.kurtosis(rets))
                
                # Hit rate (% positive days)
                hit_rate = float((rets > 0).sum() / len(rets))
                metrics['hit_rate'] = hit_rate
                
                # 30d return
                if len(stock_ret) >= 30:
                    ret_30d = float((1 + stock_ret.tail(30)).prod() - 1)
                    metrics['ret_30d'] = ret_30d
                
                # Beta to portfolio
                if portfolio_returns is not None and len(portfolio_returns) >= 20:
                    aligned_port, aligned_stock = portfolio_returns.align(rets, join='inner')
                    if len(aligned_port) >= 20:
                        try:
                            cov = np.cov(aligned_stock, aligned_port)[0, 1]
                            var_port = np.var(aligned_port)
                            if var_port > 1e-12:
                                beta = cov / var_port
                                metrics['beta'] = float(beta)
                        except Exception:
                            pass
        
        # P&L contribution from factor_vectors if available
        if factor_vectors is not None and symbol in factor_vectors.index:
            row = factor_vectors.loc[symbol]
            pnl = row.get('total_pnl_contrib', 0.0)
            if pnl is not None:
                metrics['pnl_contrib'] = float(pnl)
        
        results[symbol] = metrics
    
    return results


# =============================================================================
# Spearman IC Analysis by Horizon
# =============================================================================

def compute_spearman_ic_by_horizon(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    horizons: List[int] = [1, 5, 10, 21, 63],
    min_obs: int = 50,
) -> pd.DataFrame:
    """
    Compute Spearman rank correlation (IC) of weights vs forward returns.
    
    This measures how well the portfolio weights predict forward returns
    at different horizons. A positive IC indicates predictive power.
    
    IC = Spearman ρ(weight_t, return_{t+h})
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns (for computing stock returns)
        horizons: List of forward horizons in days (e.g., [1, 5, 10, 21, 63])
        min_obs: Minimum observations required
        
    Returns:
        DataFrame with columns: horizon, ic_mean, ic_std, t_stat, p_value, n_obs, hit_rate
    """
    from scipy import stats
    
    if weights is None or weights.empty:
        return pd.DataFrame()
    
    if portfolio_returns is None or portfolio_returns.empty:
        return pd.DataFrame()
    
    results = []
    
    for horizon in horizons:
        ic_values = []
        
        # For each date, compute cross-sectional IC
        for date_idx in range(len(weights.index) - horizon - 1):
            date = weights.index[date_idx]
            future_date = weights.index[min(date_idx + horizon, len(weights.index) - 1)]
            
            # Get weights at date
            w = weights.iloc[date_idx]
            active_assets = w[w.abs() > 1e-12].index.tolist()
            
            if len(active_assets) < 5:  # Need enough assets for correlation
                continue
            
            # Compute forward returns for active assets
            fwd_returns = {}
            for asset in active_assets:
                if asset not in weights.columns:
                    continue
                
                # Approximate forward return from weight changes and portfolio return
                stock_ret = compute_stock_returns_from_weights(
                    weights.iloc[date_idx:date_idx+horizon+2],
                    portfolio_returns.iloc[date_idx:date_idx+horizon+2],
                    asset
                )
                if stock_ret is not None and len(stock_ret) > 0:
                    # Cumulative return over horizon
                    fwd_returns[asset] = float((1 + stock_ret).prod() - 1)
            
            if len(fwd_returns) < 5:
                continue
            
            # Compute Spearman correlation
            common = list(fwd_returns.keys())
            w_vals = w[common].values
            r_vals = np.array([fwd_returns[a] for a in common])
            
            try:
                ic, _ = stats.spearmanr(w_vals, r_vals)
                if not np.isnan(ic):
                    ic_values.append(ic)
            except Exception:
                pass
        
        if len(ic_values) >= min_obs:
            ic_array = np.array(ic_values)
            ic_mean = float(np.mean(ic_array))
            ic_std = float(np.std(ic_array))
            
            # T-stat for mean IC being different from zero
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_array)) + 1e-12)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ic_array) - 1))
            
            # Hit rate: % of times IC > 0
            hit_rate = float((ic_array > 0).sum() / len(ic_array))
            
            results.append({
                "horizon": horizon,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "n_obs": len(ic_array),
                "hit_rate": hit_rate,
            })
    
    return pd.DataFrame(results)


def compute_decile_spread_analysis(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    n_deciles: int = 10,
    horizons: List[int] = [1, 5, 21],
    min_assets: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Compute decile spread analysis: top decile minus bottom decile returns.
    
    For each date, ranks stocks by weight and computes returns of
    top decile vs bottom decile portfolios. The spread measures
    whether high-weight stocks outperform low-weight stocks.
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns
        n_deciles: Number of buckets (default 10 for deciles)
        horizons: List of forward horizons to analyze
        min_assets: Minimum assets required to form deciles
        
    Returns:
        Dict with:
        - 'spread_by_horizon': DataFrame with horizon, spread_mean, spread_std, t_stat
        - 'decile_returns': DataFrame with average return by decile for each horizon
        - 'spread_ts': DataFrame with time series of spreads
    """
    from scipy import stats
    
    if weights is None or weights.empty:
        return {}
    
    if portfolio_returns is None or portfolio_returns.empty:
        return {}
    
    # Compute decile returns for each horizon
    decile_returns_by_horizon = {}
    spread_ts_by_horizon = {}
    
    for horizon in horizons:
        decile_returns = {d: [] for d in range(1, n_deciles + 1)}
        spread_values = []
        spread_dates = []
        
        for date_idx in range(len(weights.index) - horizon - 1):
            date = weights.index[date_idx]
            
            # Get weights at date
            w = weights.iloc[date_idx]
            active = w[w.abs() > 1e-12]
            
            if len(active) < min_assets:
                continue
            
            # Compute forward returns for active assets
            fwd_returns = {}
            for asset in active.index:
                stock_ret = compute_stock_returns_from_weights(
                    weights.iloc[date_idx:date_idx+horizon+2],
                    portfolio_returns.iloc[date_idx:date_idx+horizon+2],
                    asset
                )
                if stock_ret is not None and len(stock_ret) > 0:
                    fwd_returns[asset] = float((1 + stock_ret).prod() - 1)
            
            if len(fwd_returns) < min_assets:
                continue
            
            # Rank by absolute weight (higher = more conviction)
            assets_sorted = sorted(fwd_returns.keys(), key=lambda x: abs(w[x]), reverse=True)
            n_per_decile = max(1, len(assets_sorted) // n_deciles)
            
            # Assign to deciles
            for i, asset in enumerate(assets_sorted):
                decile = min(n_deciles, i // n_per_decile + 1)
                decile_returns[decile].append(fwd_returns[asset])
            
            # Top vs bottom spread
            top_decile_assets = assets_sorted[:n_per_decile]
            bottom_decile_assets = assets_sorted[-n_per_decile:]
            
            top_ret = np.mean([fwd_returns[a] for a in top_decile_assets])
            bottom_ret = np.mean([fwd_returns[a] for a in bottom_decile_assets])
            spread = top_ret - bottom_ret
            
            spread_values.append(spread)
            spread_dates.append(date)
        
        # Store decile returns
        decile_returns_by_horizon[horizon] = {
            d: np.mean(rets) if rets else 0.0 
            for d, rets in decile_returns.items()
        }
        
        # Store spread time series
        if spread_values:
            spread_ts_by_horizon[horizon] = pd.Series(spread_values, index=spread_dates)
    
    # Build summary DataFrame
    spread_summary = []
    for horizon in horizons:
        if horizon in spread_ts_by_horizon:
            ts = spread_ts_by_horizon[horizon]
            if len(ts) >= 20:
                mean_spread = float(ts.mean())
                std_spread = float(ts.std())
                t_stat = mean_spread / (std_spread / np.sqrt(len(ts)) + 1e-12)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ts) - 1))
                
                spread_summary.append({
                    "horizon": horizon,
                    "spread_mean": mean_spread,
                    "spread_std": std_spread,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "n_obs": len(ts),
                    "hit_rate": float((ts > 0).sum() / len(ts)),
                })
    
    # Build decile returns DataFrame
    decile_df_rows = []
    for decile in range(1, n_deciles + 1):
        row = {"decile": decile}
        for horizon in horizons:
            row[f"ret_t{horizon}"] = decile_returns_by_horizon.get(horizon, {}).get(decile, 0.0)
        decile_df_rows.append(row)
    
    # Build spread time series DataFrame
    spread_ts_df = pd.DataFrame(spread_ts_by_horizon)
    if not spread_ts_df.empty:
        spread_ts_df.columns = [f"spread_t{h}" for h in spread_ts_df.columns]
    
    return {
        "spread_by_horizon": pd.DataFrame(spread_summary),
        "decile_returns": pd.DataFrame(decile_df_rows),
        "spread_ts": spread_ts_df,
    }


# =============================================================================
# Stock vs Portfolio Comparison
# =============================================================================

def compute_stock_vs_portfolio_returns(
    symbols: List[str],
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Compute cumulative returns for stocks and portfolio for comparison.
    
    Args:
        symbols: List of stock symbols to compare
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio daily returns
        
    Returns:
        DataFrame with cumulative returns indexed by date,
        columns = [symbol1, symbol2, ..., 'Portfolio']
    """
    if not symbols:
        return pd.DataFrame()
    
    returns_dict = {}
    
    # Add portfolio returns
    if portfolio_returns is not None and not portfolio_returns.empty:
        returns_dict["Portfolio"] = portfolio_returns
    
    # Add stock returns
    for symbol in symbols:
        stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        if stock_ret is not None and len(stock_ret) > 0:
            returns_dict[symbol] = stock_ret
    
    if not returns_dict:
        return pd.DataFrame()
    
    # Align all returns
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    # Compute cumulative returns
    cum_returns = (1 + returns_df).cumprod() - 1
    
    return cum_returns


def compute_rolling_stock_correlation(
    symbols: List[str],
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling correlation between stocks and portfolio.
    
    Args:
        symbols: List of stock symbols
        weights: Portfolio weights
        portfolio_returns: Portfolio daily returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling correlations (stock vs portfolio)
    """
    if not symbols:
        return pd.DataFrame()
    
    if portfolio_returns is None or portfolio_returns.empty:
        return pd.DataFrame()
    
    rolling_corrs = {}
    
    for symbol in symbols:
        stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        if stock_ret is not None and len(stock_ret) >= window:
            aligned_stock, aligned_port = stock_ret.align(portfolio_returns, join='inner')
            if len(aligned_stock) >= window:
                rolling_corrs[f"{symbol} vs Portfolio"] = aligned_stock.rolling(window).corr(aligned_port)
    
    return pd.DataFrame(rolling_corrs).dropna(how='all')


def compute_rolling_stock_sharpe(
    symbols: List[str],
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratios for stocks and portfolio.
    
    Args:
        symbols: List of stock symbols
        weights: Portfolio weights
        portfolio_returns: Portfolio daily returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling Sharpe ratios
    """
    if not symbols:
        return pd.DataFrame()
    
    rolling_sharpe = {}
    
    # Portfolio Sharpe
    if portfolio_returns is not None and len(portfolio_returns) >= window:
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        rolling_sharpe["Portfolio"] = (rolling_mean * 252) / (rolling_std * np.sqrt(252) + 1e-8)
    
    # Stock Sharpe ratios
    for symbol in symbols:
        stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        if stock_ret is not None and len(stock_ret) >= window:
            rolling_mean = stock_ret.rolling(window).mean()
            rolling_std = stock_ret.rolling(window).std()
            rolling_sharpe[symbol] = (rolling_mean * 252) / (rolling_std * np.sqrt(252) + 1e-8)
    
    return pd.DataFrame(rolling_sharpe).dropna(how='all')


def compute_stock_portfolio_beta(
    symbols: List[str],
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling beta of stocks to portfolio.
    
    Args:
        symbols: List of stock symbols
        weights: Portfolio weights
        portfolio_returns: Portfolio daily returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling beta values
    """
    if not symbols:
        return pd.DataFrame()
    
    if portfolio_returns is None or portfolio_returns.empty:
        return pd.DataFrame()
    
    rolling_betas = {}
    
    for symbol in symbols:
        stock_ret = compute_stock_returns_from_weights(weights, portfolio_returns, symbol)
        if stock_ret is not None and len(stock_ret) >= window:
            aligned_stock, aligned_port = stock_ret.align(portfolio_returns, join='inner')
            
            if len(aligned_stock) >= window:
                # Rolling beta = rolling_cov / rolling_var
                rolling_cov = aligned_stock.rolling(window).cov(aligned_port)
                rolling_var = aligned_port.rolling(window).var()
                rolling_betas[symbol] = rolling_cov / (rolling_var + 1e-12)
    
    return pd.DataFrame(rolling_betas).dropna(how='all')


# =============================================================================
# Plotly Charts for Stock vs Portfolio Comparison
# =============================================================================

# Check for Plotly availability
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_stock_portfolio_equity_chart(
    cum_returns: pd.DataFrame,
    height: int = 400,
) -> Optional[Any]:
    """
    Create multi-equity cumulative return chart comparing stocks to portfolio.
    
    Args:
        cum_returns: DataFrame with cumulative returns (columns are symbol names)
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or cum_returns.empty:
        return None
    
    fig = go.Figure()
    
    # Color palette - portfolio gets special color
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63']
    portfolio_color = '#f1c40f'  # Gold for portfolio
    
    for i, col in enumerate(cum_returns.columns):
        if col == "Portfolio":
            color = portfolio_color
            line_width = 3
            dash = None
        else:
            color = colors[i % len(colors)]
            line_width = 2
            dash = None
        
        final_ret = cum_returns[col].iloc[-1] if len(cum_returns) > 0 else 0
        
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            mode='lines',
            name=col,
            line={'color': color, 'width': line_width, 'dash': dash},
            hovertemplate=f'{col}<br>%{{y:.2%}}<extra></extra>',
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    fig.update_layout(
        title={'text': 'Cumulative Returns: Stocks vs Portfolio', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 60, 'r': 40, 't': 60, 'b': 40},
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'gridcolor': '#3A3A3A', 'tickformat': '.0%'},
        hovermode='x unified',
        hoverlabel={'bgcolor': '#1e1e1e', 'bordercolor': '#444', 'font': {'size': 11, 'color': '#ecf0f1'}},
        legend={'orientation': 'h', 'y': -0.15, 'x': 0.5, 'xanchor': 'center'},
    )
    
    return fig


def create_rolling_stock_correlation_chart(
    rolling_corr: pd.DataFrame,
    height: int = 350,
) -> Optional[Any]:
    """
    Create rolling correlation chart between stocks and portfolio.
    
    Args:
        rolling_corr: DataFrame with rolling correlation series
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or rolling_corr.empty:
        return None
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    fig = go.Figure()
    
    for i, col in enumerate(rolling_corr.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr[col],
            mode='lines',
            name=col,
            line={'color': color, 'width': 1.5},
            hovertemplate='%{y:.3f}<extra></extra>',
        ))
    
    # Reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(46, 204, 113, 0.4)")
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    fig.add_hline(y=-1.0, line_dash="dot", line_color="rgba(231, 76, 60, 0.4)")
    
    fig.update_layout(
        title={'text': 'Rolling Correlation with Portfolio', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 50, 'r': 30, 't': 50, 'b': 40},
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'gridcolor': '#3A3A3A', 'range': [-1.1, 1.1], 'tickformat': '.2f'},
        hovermode='x unified',
        hoverlabel={'bgcolor': '#1e1e1e', 'bordercolor': '#444', 'font': {'size': 11, 'color': '#ecf0f1'}},
        legend={'orientation': 'h', 'y': -0.15, 'x': 0.5, 'xanchor': 'center'},
    )
    
    return fig


def create_rolling_stock_sharpe_chart(
    rolling_sharpe: pd.DataFrame,
    height: int = 350,
) -> Optional[Any]:
    """
    Create rolling Sharpe ratio chart for stocks and portfolio.
    
    Args:
        rolling_sharpe: DataFrame with rolling Sharpe series
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or rolling_sharpe.empty:
        return None
    
    fig = go.Figure()
    
    colors = ['#f1c40f', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']  # Gold for portfolio first
    
    for i, col in enumerate(rolling_sharpe.columns):
        if col == "Portfolio":
            color = '#f1c40f'  # Gold
            line_width = 3
        else:
            color = colors[(i + 1) % len(colors)]
            line_width = 2
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe[col],
            mode='lines',
            name=col,
            line={'color': color, 'width': line_width},
            hovertemplate='%{y:.2f}<extra></extra>',
        ))
    
    # Reference lines for Sharpe quality
    fig.add_hline(y=2.0, line_dash="dot", line_color="rgba(46, 204, 113, 0.3)",
                  annotation_text="Excellent", annotation_position="right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(52, 152, 219, 0.3)",
                  annotation_text="Good", annotation_position="right")
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    fig.update_layout(
        title={'text': 'Rolling Sharpe Ratio', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 50, 'r': 60, 't': 50, 'b': 40},
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'gridcolor': '#3A3A3A', 'tickformat': '.1f', 'title': 'Sharpe (Ann.)'},
        hovermode='x unified',
        hoverlabel={'bgcolor': '#1e1e1e', 'bordercolor': '#444', 'font': {'size': 11, 'color': '#ecf0f1'}},
        legend={'orientation': 'h', 'y': -0.15, 'x': 0.5, 'xanchor': 'center'},
    )
    
    return fig


def create_ic_by_horizon_chart(
    ic_df: pd.DataFrame,
    height: int = 350,
) -> Optional[Any]:
    """
    Create bar chart showing Spearman IC by horizon.
    
    Args:
        ic_df: DataFrame with columns: horizon, ic_mean, ic_std, t_stat
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or ic_df.empty:
        return None
    
    # Color bars based on t-stat significance - vectorized operation
    t_stats = ic_df.get('t_stat', pd.Series(0, index=ic_df.index))
    ic_means = ic_df.get('ic_mean', pd.Series(0, index=ic_df.index))
    significant = t_stats.abs() >= 2.0
    colors = [
        '#2ecc71' if (sig and ic_mean > 0) else '#e74c3c' if sig else '#95a5a6'
        for sig, ic_mean in zip(significant, ic_means)
    ]
    
    # Fallback for any remaining rows (shouldn't happen, but safe)
    if len(colors) < len(ic_df):
        for _ in range(len(ic_df) - len(colors)):
            colors.append('#95a5a6')
            colors.append('#7f8c8d')  # Gray for non-significant
    
    fig = go.Figure()
    
    # Add bars for IC mean
    fig.add_trace(go.Bar(
        x=[f"T+{h}" for h in ic_df['horizon']],
        y=ic_df['ic_mean'],
        marker_color=colors,
        error_y={'type': 'data', 'array': ic_df['ic_std'], 'color': '#ecf0f1', 'thickness': 1},
        hovertemplate='<b>%{x}</b><br>IC: %{y:.4f}<br>±%{error_y.array:.4f}<extra></extra>',
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    fig.update_layout(
        title={'text': 'Spearman IC by Horizon', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 50, 'r': 30, 't': 50, 'b': 60},
        xaxis={'title': 'Forward Horizon', 'gridcolor': '#3A3A3A'},
        yaxis={'title': 'IC (ρ)', 'gridcolor': '#3A3A3A', 'tickformat': '.3f'},
        showlegend=False,
    )
    
    return fig


def create_decile_spread_chart(
    decile_returns: pd.DataFrame,
    horizons: List[int] = [1, 5, 21],
    height: int = 350,
) -> Optional[Any]:
    """
    Create bar chart showing returns by decile.
    
    Args:
        decile_returns: DataFrame with columns: decile, ret_t1, ret_t5, ret_t21
        horizons: Horizons to show
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or decile_returns.empty:
        return None
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    fig = go.Figure()
    
    for i, h in enumerate(horizons):
        col = f"ret_t{h}"
        if col not in decile_returns.columns:
            continue
        
        fig.add_trace(go.Bar(
            x=decile_returns['decile'],
            y=decile_returns[col],
            name=f"T+{h}",
            marker_color=colors[i % len(colors)],
            hovertemplate='<b>Decile %{x}</b><br>Return: %{y:.3%}<extra></extra>',
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    fig.update_layout(
        title={'text': 'Returns by Weight Decile', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 50, 'r': 30, 't': 50, 'b': 60},
        xaxis={'title': 'Weight Decile (1=Highest)', 'gridcolor': '#3A3A3A', 'dtick': 1},
        yaxis={'title': 'Average Forward Return', 'gridcolor': '#3A3A3A', 'tickformat': '.2%'},
        barmode='group',
        legend={'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
    )
    
    return fig


def create_decile_spread_ts_chart(
    spread_ts: pd.DataFrame,
    height: int = 300,
) -> Optional[Any]:
    """
    Create time series chart of top-bottom decile spread.
    
    Args:
        spread_ts: DataFrame with columns like spread_t1, spread_t5, spread_t21
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or spread_ts.empty:
        return None
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    fig = go.Figure()
    
    for i, col in enumerate(spread_ts.columns):
        horizon = col.replace('spread_t', '')
        fig.add_trace(go.Scatter(
            x=spread_ts.index,
            y=spread_ts[col],
            mode='lines',
            name=f"T+{horizon}",
            line={'color': colors[i % len(colors)], 'width': 1.5},
            hovertemplate='%{y:.3%}<extra></extra>',
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    fig.update_layout(
        title={'text': 'Top-Bottom Decile Spread Over Time', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=height,
        margin={'l': 50, 'r': 30, 't': 50, 'b': 40},
        xaxis={'gridcolor': '#3A3A3A'},
        yaxis={'title': 'Spread', 'gridcolor': '#3A3A3A', 'tickformat': '.2%'},
        hovermode='x unified',
        hoverlabel={'bgcolor': '#1e1e1e', 'bordercolor': '#444', 'font': {'size': 11, 'color': '#ecf0f1'}},
        legend={'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
    )
    
    return fig
