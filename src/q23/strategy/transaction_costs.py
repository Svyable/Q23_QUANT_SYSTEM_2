"""q23.strategy.transaction_costs

Transaction cost modeling module supporting multiple TC schemes.

The default is the Quantiacs contest model:
    TC = 0.05 × ATR(14) × |ΔPosition|

This module provides:
- TransactionCostScheme enum for different TC models
- TransactionCostConfig dataclass for configuration
- TransactionCostModel class for computing costs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except ImportError:
    xr = None  # type: ignore


class TransactionCostScheme(Enum):
    """Available transaction cost models."""
    QUANTIACS_ATR = "quantiacs_atr"  # 5% of ATR(14) per position change (default)
    FLAT_BPS = "flat_bps"            # Fixed basis points on turnover
    PERCENTAGE = "percentage"         # Fixed percentage of trade value
    TIERED = "tiered"                # Tiered based on trade size
    CUSTOM = "custom"                # User-defined function


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling.
    
    Attributes:
        scheme: Which TC model to use (default: QUANTIACS_ATR)
        atr_window: ATR lookback period for Quantiacs model (default: 14)
        atr_multiplier: Multiplier for ATR model (default: 0.05 = 5%)
        flat_bps: Basis points for FLAT_BPS scheme (default: 10.0)
        percentage: Percentage for PERCENTAGE scheme (default: 0.001 = 0.1%)
        min_cost_bps: Floor cost in bps (default: 0.0)
        max_cost_bps: Ceiling cost in bps (default: 100.0)
        tiered_thresholds: List of (threshold, bps) tuples for TIERED scheme
    """
    scheme: TransactionCostScheme = TransactionCostScheme.QUANTIACS_ATR
    atr_window: int = 14
    atr_multiplier: float = 0.05
    flat_bps: float = 10.0
    percentage: float = 0.001
    min_cost_bps: float = 0.0
    max_cost_bps: float = 100.0
    tiered_thresholds: list = field(default_factory=lambda: [
        (0.01, 5.0),   # < 1% of portfolio: 5 bps
        (0.05, 10.0),  # < 5% of portfolio: 10 bps
        (0.10, 15.0),  # < 10% of portfolio: 15 bps
        (1.0, 25.0),   # >= 10% of portfolio: 25 bps
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary for persistence."""
        return {
            "scheme": self.scheme.value,
            "atr_window": self.atr_window,
            "atr_multiplier": self.atr_multiplier,
            "flat_bps": self.flat_bps,
            "percentage": self.percentage,
            "min_cost_bps": self.min_cost_bps,
            "max_cost_bps": self.max_cost_bps,
            "tiered_thresholds": self.tiered_thresholds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionCostConfig":
        """Deserialize config from dictionary."""
        scheme_val = data.get("scheme", "quantiacs_atr")
        try:
            scheme = TransactionCostScheme(scheme_val)
        except ValueError:
            scheme = TransactionCostScheme.QUANTIACS_ATR
        
        return cls(
            scheme=scheme,
            atr_window=int(data.get("atr_window", 14)),
            atr_multiplier=float(data.get("atr_multiplier", 0.05)),
            flat_bps=float(data.get("flat_bps", 10.0)),
            percentage=float(data.get("percentage", 0.001)),
            min_cost_bps=float(data.get("min_cost_bps", 0.0)),
            max_cost_bps=float(data.get("max_cost_bps", 100.0)),
            tiered_thresholds=data.get("tiered_thresholds", [
                (0.01, 5.0), (0.05, 10.0), (0.10, 15.0), (1.0, 25.0)
            ]),
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.atr_window < 1:
            raise ValueError(f"atr_window must be >= 1, got {self.atr_window}")
        if self.atr_multiplier < 0:
            raise ValueError(f"atr_multiplier must be >= 0, got {self.atr_multiplier}")
        if self.flat_bps < 0:
            raise ValueError(f"flat_bps must be >= 0, got {self.flat_bps}")
        if self.min_cost_bps > self.max_cost_bps:
            raise ValueError(f"min_cost_bps ({self.min_cost_bps}) > max_cost_bps ({self.max_cost_bps})")


def compute_atr_pandas(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Compute Average True Range (ATR) using pandas DataFrames.
    
    ATR = SMA(True Range, window)
    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
    
    Args:
        high: High prices (time x asset)
        low: Low prices (time x asset)
        close: Close prices (time x asset)
        window: Lookback period for ATR
        
    Returns:
        ATR values (time x asset)
    """
    prev_close = close.shift(1)
    
    # True Range components
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    
    # True Range = max of the three
    tr = pd.concat([hl, hc, lc], axis=1).groupby(level=0, axis=1).max()
    
    # If concat created multi-level columns, fix it
    if isinstance(tr.columns, pd.MultiIndex):
        tr.columns = tr.columns.get_level_values(0)
    
    # Handle case where columns are duplicated
    if len(tr.columns) != len(high.columns):
        # Compute element-wise max
        tr = hl.copy()
        tr = tr.where(tr >= hc, hc)
        tr = tr.where(tr >= lc, lc)
    
    # ATR = Simple Moving Average of True Range
    atr = tr.rolling(window=window, min_periods=window).mean()
    
    return atr.fillna(0.0)


def compute_atr_xarray(
    high: "xr.DataArray",
    low: "xr.DataArray",
    close: "xr.DataArray",
    window: int = 14,
) -> "xr.DataArray":
    """Compute Average True Range (ATR) using xarray DataArrays.
    
    This is the same algorithm used in factors.py but exposed as a utility.
    
    Args:
        high: High prices with dims (time, asset)
        low: Low prices with dims (time, asset)
        close: Close prices with dims (time, asset)
        window: Lookback period for ATR
        
    Returns:
        ATR values with dims (time, asset)
    """
    if xr is None:
        raise ImportError("xarray is required for compute_atr_xarray")
    
    prev = close.shift(time=1)
    hl = high - low
    hc = np.abs(high - prev)
    lc = np.abs(low - prev)
    
    # True Range = max of the three components
    tr = xr.where(hl > hc, hl, hc)
    tr = xr.where(tr > lc, tr, lc)
    
    # ATR = Simple Moving Average of True Range
    atr = tr.fillna(0.0).rolling(time=window, min_periods=window).mean()
    
    return atr.fillna(0.0)


class TransactionCostModel:
    """Transaction cost model supporting multiple schemes.
    
    Default is Quantiacs contest model:
        TC = atr_multiplier × ATR(atr_window) × |ΔPosition|
        
    Example:
        >>> config = TransactionCostConfig()  # Defaults to Quantiacs ATR
        >>> model = TransactionCostModel(config)
        >>> costs = model.compute_costs(weights_delta, close_prices, atr=atr_values)
    """
    
    def __init__(self, config: Optional[TransactionCostConfig] = None):
        """Initialize transaction cost model.
        
        Args:
            config: TC configuration. If None, uses Quantiacs defaults.
        """
        self.config = config or TransactionCostConfig()
        self.config.validate()
    
    def compute_atr(
        self,
        high: Union[pd.DataFrame, "xr.DataArray"],
        low: Union[pd.DataFrame, "xr.DataArray"],
        close: Union[pd.DataFrame, "xr.DataArray"],
        window: Optional[int] = None,
    ) -> Union[pd.DataFrame, "xr.DataArray"]:
        """Compute ATR from OHLC data.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Override ATR window (default: config.atr_window)
            
        Returns:
            ATR values in same format as input
        """
        win = window or self.config.atr_window
        
        if isinstance(high, pd.DataFrame):
            return compute_atr_pandas(high, low, close, win)
        elif xr is not None and isinstance(high, xr.DataArray):
            return compute_atr_xarray(high, low, close, win)
        else:
            raise TypeError(f"Unsupported type: {type(high)}")
    
    def compute_costs_flat_bps(
        self,
        weights_delta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute TC using flat basis points on turnover.
        
        TC = |ΔWeight| × (flat_bps / 10000)
        
        Args:
            weights_delta: Position changes (time x asset)
            
        Returns:
            Per-asset transaction costs (time x asset)
        """
        tc_rate = self.config.flat_bps / 10000.0
        costs = weights_delta.abs() * tc_rate
        return costs.fillna(0.0)
    
    def compute_costs_percentage(
        self,
        weights_delta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute TC using fixed percentage of trade value.
        
        TC = |ΔWeight| × percentage
        
        Args:
            weights_delta: Position changes (time x asset)
            
        Returns:
            Per-asset transaction costs (time x asset)
        """
        costs = weights_delta.abs() * self.config.percentage
        return costs.fillna(0.0)
    
    def compute_costs_quantiacs_atr(
        self,
        weights_delta: pd.DataFrame,
        close: pd.DataFrame,
        atr: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute TC using Quantiacs ATR model.
        
        TC = atr_multiplier × ATR(14) × |ΔPosition|
        
        The ATR is normalized by close price to get a percentage-based cost:
        TC = atr_multiplier × (ATR / Close) × |ΔWeight|
        
        Args:
            weights_delta: Position changes (time x asset)
            close: Close prices (time x asset)
            atr: Pre-computed ATR (optional, will compute if not provided)
            high: High prices (required if atr not provided)
            low: Low prices (required if atr not provided)
            
        Returns:
            Per-asset transaction costs (time x asset)
        """
        if atr is None:
            if high is None or low is None:
                raise ValueError("Must provide either atr or (high, low) for ATR computation")
            atr = compute_atr_pandas(high, low, close, self.config.atr_window)
        
        # Align indices
        common_idx = weights_delta.index.intersection(atr.index).intersection(close.index)
        common_cols = [c for c in weights_delta.columns if c in atr.columns and c in close.columns]
        
        delta = weights_delta.reindex(index=common_idx, columns=common_cols).fillna(0.0)
        atr_aligned = atr.reindex(index=common_idx, columns=common_cols).fillna(0.0)
        close_aligned = close.reindex(index=common_idx, columns=common_cols).fillna(1.0)
        
        # Normalize ATR by close price to get volatility as percentage
        atr_pct = atr_aligned / (close_aligned + 1e-12)
        
        # TC = multiplier × ATR_pct × |ΔWeight|
        costs = self.config.atr_multiplier * atr_pct * delta.abs()
        
        return costs.fillna(0.0)
    
    def compute_costs_tiered(
        self,
        weights_delta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute TC using tiered basis points based on trade size.
        
        Args:
            weights_delta: Position changes (time x asset)
            
        Returns:
            Per-asset transaction costs (time x asset)
        """
        costs = pd.DataFrame(0.0, index=weights_delta.index, columns=weights_delta.columns)
        abs_delta = weights_delta.abs()
        
        for threshold, bps in sorted(self.config.tiered_thresholds):
            mask = abs_delta <= threshold
            costs = costs.where(~mask, abs_delta * (bps / 10000.0))
        
        return costs.fillna(0.0)
    
    def compute_costs(
        self,
        weights_delta: pd.DataFrame,
        close: Optional[pd.DataFrame] = None,
        atr: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute transaction costs using configured scheme.
        
        Args:
            weights_delta: Position changes (time x asset)
            close: Close prices (required for ATR scheme)
            atr: Pre-computed ATR (optional)
            high: High prices (optional, for ATR computation)
            low: Low prices (optional, for ATR computation)
            
        Returns:
            Per-asset transaction costs (time x asset)
        """
        scheme = self.config.scheme
        
        if scheme == TransactionCostScheme.FLAT_BPS:
            costs = self.compute_costs_flat_bps(weights_delta)
        
        elif scheme == TransactionCostScheme.PERCENTAGE:
            costs = self.compute_costs_percentage(weights_delta)
        
        elif scheme == TransactionCostScheme.TIERED:
            costs = self.compute_costs_tiered(weights_delta)
        
        elif scheme == TransactionCostScheme.QUANTIACS_ATR:
            if close is None:
                raise ValueError("close prices required for QUANTIACS_ATR scheme")
            costs = self.compute_costs_quantiacs_atr(
                weights_delta, close, atr=atr, high=high, low=low
            )
        
        elif scheme == TransactionCostScheme.CUSTOM:
            # For custom scheme, fall back to flat BPS
            costs = self.compute_costs_flat_bps(weights_delta)
        
        else:
            raise ValueError(f"Unknown TC scheme: {scheme}")
        
        # Apply min/max bounds
        if self.config.min_cost_bps > 0 or self.config.max_cost_bps < 100:
            min_cost = self.config.min_cost_bps / 10000.0
            max_cost = self.config.max_cost_bps / 10000.0
            # Only apply bounds where there's actual trading
            trading_mask = weights_delta.abs() > 1e-12
            costs = costs.clip(lower=min_cost * trading_mask, upper=max_cost)
        
        return costs
    
    def total_cost(
        self,
        weights_delta: pd.DataFrame,
        close: Optional[pd.DataFrame] = None,
        atr: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute total portfolio transaction cost per time period.
        
        Args:
            weights_delta: Position changes (time x asset)
            close: Close prices (required for ATR scheme)
            atr: Pre-computed ATR (optional)
            high: High prices (optional)
            low: Low prices (optional)
            
        Returns:
            Total TC per time period (time,)
        """
        costs = self.compute_costs(weights_delta, close, atr, high, low)
        return costs.sum(axis=1)
    
    def compute_tc_drag(
        self,
        weights: pd.DataFrame,
        close: Optional[pd.DataFrame] = None,
        atr: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Compute TC drag statistics for a weight series.
        
        Args:
            weights: Portfolio weights over time (time x asset)
            close: Close prices
            atr: Pre-computed ATR (optional)
            high: High prices (optional)
            low: Low prices (optional)
            
        Returns:
            Dictionary with TC statistics
        """
        # Compute weight changes
        weights_delta = weights.diff().fillna(0.0)
        
        # Compute costs
        costs = self.compute_costs(weights_delta, close, atr, high, low)
        total_costs = costs.sum(axis=1)
        
        # Statistics
        avg_daily_tc = float(total_costs.mean())
        total_tc = float(total_costs.sum())
        annualized_tc_drag = avg_daily_tc * 252
        
        # Turnover for comparison
        turnover = weights_delta.abs().sum(axis=1)
        avg_turnover = float(turnover.mean())
        
        # TC as percentage of turnover
        tc_per_turnover = total_tc / (turnover.sum() + 1e-12)
        
        return {
            "avg_daily_tc": avg_daily_tc,
            "total_tc": total_tc,
            "annualized_tc_drag": annualized_tc_drag,
            "avg_daily_turnover": avg_turnover,
            "tc_per_turnover_pct": float(tc_per_turnover * 100),
            "scheme": self.config.scheme.value,
        }
    
    def compare_schemes(
        self,
        weights: pd.DataFrame,
        close: Optional[pd.DataFrame] = None,
        atr: Optional[pd.DataFrame] = None,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compare TC across different schemes.
        
        Args:
            weights: Portfolio weights over time
            close: Close prices
            atr: Pre-computed ATR
            high: High prices
            low: Low prices
            
        Returns:
            DataFrame comparing TC metrics across schemes
        """
        weights_delta = weights.diff().fillna(0.0)
        results = []
        
        for scheme in TransactionCostScheme:
            if scheme == TransactionCostScheme.CUSTOM:
                continue  # Skip custom
            
            try:
                config = TransactionCostConfig(scheme=scheme)
                model = TransactionCostModel(config)
                
                if scheme == TransactionCostScheme.QUANTIACS_ATR and close is None:
                    continue
                
                costs = model.compute_costs(weights_delta, close, atr, high, low)
                total_costs = costs.sum(axis=1)
                
                results.append({
                    "scheme": scheme.value,
                    "avg_daily_tc": float(total_costs.mean()),
                    "total_tc": float(total_costs.sum()),
                    "annualized_drag": float(total_costs.mean() * 252),
                })
            except Exception:
                continue
        
        return pd.DataFrame(results)


# Convenience functions for common use cases

def compute_quantiacs_tc(
    weights_delta: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    atr_window: int = 14,
    atr_multiplier: float = 0.05,
) -> pd.DataFrame:
    """Convenience function for Quantiacs TC computation.
    
    TC = atr_multiplier × ATR(atr_window) × |ΔPosition|
    
    Args:
        weights_delta: Position changes (time x asset)
        close: Close prices
        high: High prices
        low: Low prices
        atr_window: ATR lookback period (default: 14)
        atr_multiplier: ATR multiplier (default: 0.05 = 5%)
        
    Returns:
        Per-asset transaction costs
    """
    config = TransactionCostConfig(
        scheme=TransactionCostScheme.QUANTIACS_ATR,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
    )
    model = TransactionCostModel(config)
    return model.compute_costs(weights_delta, close, high=high, low=low)


def compute_flat_tc(
    weights_delta: pd.DataFrame,
    bps: float = 10.0,
) -> pd.DataFrame:
    """Convenience function for flat BPS TC computation.
    
    TC = |ΔWeight| × (bps / 10000)
    
    Args:
        weights_delta: Position changes (time x asset)
        bps: Basis points (default: 10)
        
    Returns:
        Per-asset transaction costs
    """
    config = TransactionCostConfig(
        scheme=TransactionCostScheme.FLAT_BPS,
        flat_bps=bps,
    )
    model = TransactionCostModel(config)
    return model.compute_costs(weights_delta)
