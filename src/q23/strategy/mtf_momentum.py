"""q23.strategy.mtf_momentum

Multi-Timeframe (MTF) Momentum Factor Library

Core Concept:
- Daily HLOC data aggregated to Weekly/Monthly/Quarterly/Yearly timeframes
- Returns calculated for both completed periods AND to-date (WTD/MTD/QTD/YTD)
- IC matrix tracks which timeframe best predicts forward returns
- Dynamic IC-weighting creates adaptive momentum composite

Timeframe Definitions (trading days):
- Week: 5 trading days
- Month: ~21 trading days  
- Quarter: ~63 trading days
- Year: ~252 trading days

Key Factor Categories:
1. Pure Returns: WTD, MTD, QTD, YTD, 1W, 1M, 1Q, 1Y
2. Momentum Alignment: confluence across timeframes
3. IC-Adaptive Momentum: weighted by predictive power
4. Acceleration: rate of change across timeframes
5. HLOC-Enhanced: range-based momentum signals

Usage:
    from q23.strategy.mtf_momentum import MTFMomentumLibrary, MTF_FACTORS
    
    mtf = MTFMomentumLibrary(market_data)
    F, artifacts = mtf.compute()
    
    # Get IC matrix for analysis
    ic_matrix = mtf.compute_ic_matrix(fwd_returns, horizon=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.math_utils import safe_zscore, safe_corrcoef


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.mtf_momentum")


# ==============================================================================
# Timeframe Definitions
# ==============================================================================

class Timeframe(str, Enum):
    """Standard timeframe definitions."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# Approximate trading days per period
TRADING_DAYS = {
    Timeframe.DAILY: 1,
    Timeframe.WEEKLY: 5,
    Timeframe.MONTHLY: 21,
    Timeframe.QUARTERLY: 63,
    Timeframe.YEARLY: 252,
}


# ==============================================================================
# Factor Registry for MTF Factors
# ==============================================================================

@dataclass(frozen=True)
class MTFFactorDefinition:
    """Metadata for a multi-timeframe factor."""
    name: str
    category: str
    description: str
    timeframes: Tuple[str, ...]
    sign: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "timeframes": list(self.timeframes),
            "sign": self.sign,
        }


MTF_FACTOR_REGISTRY: Dict[str, MTFFactorDefinition] = {
    # ===== To-Date Returns (current incomplete period) =====
    "ret_wtd": MTFFactorDefinition(
        name="ret_wtd",
        category="to_date_returns",
        description="Week-to-date return (from last Monday)",
        timeframes=("weekly",),
    ),
    "ret_mtd": MTFFactorDefinition(
        name="ret_mtd",
        category="to_date_returns",
        description="Month-to-date return (from 1st of month)",
        timeframes=("monthly",),
    ),
    "ret_qtd": MTFFactorDefinition(
        name="ret_qtd",
        category="to_date_returns",
        description="Quarter-to-date return",
        timeframes=("quarterly",),
    ),
    "ret_ytd": MTFFactorDefinition(
        name="ret_ytd",
        category="to_date_returns",
        description="Year-to-date return",
        timeframes=("yearly",),
    ),
    
    # ===== Completed Period Returns =====
    "ret_1w": MTFFactorDefinition(
        name="ret_1w",
        category="period_returns",
        description="Last complete week return",
        timeframes=("weekly",),
    ),
    "ret_1m": MTFFactorDefinition(
        name="ret_1m",
        category="period_returns",
        description="Last 21 trading days return (~1 month)",
        timeframes=("monthly",),
    ),
    "ret_1q": MTFFactorDefinition(
        name="ret_1q",
        category="period_returns",
        description="Last 63 trading days return (~1 quarter)",
        timeframes=("quarterly",),
    ),
    "ret_1y": MTFFactorDefinition(
        name="ret_1y",
        category="period_returns",
        description="Last 252 trading days return (~1 year)",
        timeframes=("yearly",),
    ),
    
    # ===== Multi-Period Returns (skip recent) =====
    "ret_2_12m": MTFFactorDefinition(
        name="ret_2_12m",
        category="period_returns",
        description="Months 2-12 return (classic momentum, skip recent month)",
        timeframes=("monthly", "yearly"),
    ),
    "ret_1_3m": MTFFactorDefinition(
        name="ret_1_3m",
        category="period_returns",
        description="Months 1-3 return (medium-term momentum)",
        timeframes=("monthly", "quarterly"),
    ),
    
    # ===== Momentum Alignment / Confluence =====
    "mtf_alignment": MTFFactorDefinition(
        name="mtf_alignment",
        category="alignment",
        description="Momentum direction alignment across W/M/Q/Y (0-4 score)",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    "mtf_alignment_strength": MTFFactorDefinition(
        name="mtf_alignment_strength",
        category="alignment",
        description="Alignment weighted by return magnitude",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    "trend_consistency": MTFFactorDefinition(
        name="trend_consistency",
        category="alignment",
        description="Consistency of trend across timeframes (std of signed returns)",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    
    # ===== IC-Weighted Adaptive Momentum =====
    "mtf_ic_momentum": MTFFactorDefinition(
        name="mtf_ic_momentum",
        category="ic_weighted",
        description="IC-weighted momentum composite (adapts to best timeframe)",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    "mtf_ic_regime": MTFFactorDefinition(
        name="mtf_ic_regime",
        category="ic_weighted",
        description="Which timeframe has highest IC (1=W, 2=M, 3=Q, 4=Y)",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    
    # ===== Acceleration / Rate of Change =====
    "momentum_acceleration": MTFFactorDefinition(
        name="momentum_acceleration",
        category="acceleration",
        description="Short-term vs long-term momentum (acceleration signal)",
        timeframes=("weekly", "monthly", "quarterly"),
    ),
    "mtf_slope": MTFFactorDefinition(
        name="mtf_slope",
        category="acceleration",
        description="Slope of returns across timeframes (improving vs deteriorating)",
        timeframes=("weekly", "monthly", "quarterly", "yearly"),
    ),
    
    # ===== HLOC-Enhanced Momentum =====
    "hloc_range_expansion": MTFFactorDefinition(
        name="hloc_range_expansion",
        category="hloc_enhanced",
        description="Weekly range vs monthly range (breakout signal)",
        timeframes=("weekly", "monthly"),
    ),
    "hloc_close_position": MTFFactorDefinition(
        name="hloc_close_position",
        category="hloc_enhanced",
        description="Close position within monthly HLOC range",
        timeframes=("monthly",),
    ),
    "hloc_momentum_quality": MTFFactorDefinition(
        name="hloc_momentum_quality",
        category="hloc_enhanced",
        description="Returns supported by HLOC confirmation (close near high)",
        timeframes=("weekly", "monthly"),
    ),
    
    # ===== Relative Momentum (vs market) =====
    "relative_mtf_momentum": MTFFactorDefinition(
        name="relative_mtf_momentum",
        category="relative",
        description="MTF momentum relative to market (alpha component)",
        timeframes=("weekly", "monthly", "quarterly"),
    ),
    "sector_relative_mtf": MTFFactorDefinition(
        name="sector_relative_mtf",
        category="relative",
        description="MTF momentum vs universe mean (cross-sectional)",
        timeframes=("monthly", "quarterly"),
    ),
}

# Default factor set
MTF_FACTORS = tuple(MTF_FACTOR_REGISTRY.keys())

# Core momentum factors (most useful)
MTF_CORE_FACTORS = (
    "ret_wtd", "ret_mtd", "ret_qtd", "ret_ytd",
    "ret_1w", "ret_1m", "ret_1q",
    "ret_2_12m",
    "mtf_alignment", "mtf_alignment_strength",
    "mtf_ic_momentum",
    "momentum_acceleration",
    "hloc_close_position",
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _sma(x: "xr.DataArray", win: int) -> "xr.DataArray":
    """Simple moving average."""
    return x.rolling(time=win, min_periods=max(1, win // 2)).mean()


def _rolling_max(x: "xr.DataArray", win: int) -> "xr.DataArray":
    """Rolling maximum."""
    return x.rolling(time=win, min_periods=max(1, win // 2)).max()


def _rolling_min(x: "xr.DataArray", win: int) -> "xr.DataArray":
    """Rolling minimum."""
    return x.rolling(time=win, min_periods=max(1, win // 2)).min()


def _rolling_sum(x: "xr.DataArray", win: int) -> "xr.DataArray":
    """Rolling sum."""
    return x.rolling(time=win, min_periods=max(1, win // 2)).sum()


def _robust_zscore(
    x: "xr.DataArray",
    dim: str = "asset",
    eps: float = 1e-12,
    clip: float = 5.0
) -> "xr.DataArray":
    """Robust cross-sectional z-score using median/MAD."""
    med = x.median(dim=dim, skipna=True)
    mad = np.abs(x - med).median(dim=dim, skipna=True)
    z = (x - med) / (mad * 1.4826 + eps)
    return z.fillna(0.0).clip(-clip, clip)


def _compute_period_return(close: "xr.DataArray", lookback: int) -> "xr.DataArray":
    """Compute return over lookback period."""
    return (close / close.shift(time=lookback) - 1.0).fillna(0.0)


def _log_return(close: "xr.DataArray", lookback: int) -> "xr.DataArray":
    """Compute log return over lookback period."""
    ratio = (close / close.shift(time=lookback)).clip(min=1e-8)
    return xr.apply_ufunc(np.log, ratio).fillna(0.0)


# ==============================================================================
# Calendar-Aware Period Detection (for true WTD/MTD/QTD/YTD)
# ==============================================================================

def _get_period_start_mask(times: np.ndarray, period: str) -> np.ndarray:
    """
    Create mask for period starts (Monday for week, 1st for month, etc.).
    
    Args:
        times: numpy array of datetime64 or pandas DatetimeIndex
        period: 'week', 'month', 'quarter', 'year'
    
    Returns:
        Boolean mask where True = start of new period
    """
    if pd is None:
        raise ImportError("pandas required for calendar-aware period detection")
    
    dt = pd.DatetimeIndex(times)
    
    if period == "week":
        # Monday = 0
        return (dt.dayofweek == 0).values
    elif period == "month":
        return (dt.day == 1).values
    elif period == "quarter":
        return ((dt.month - 1) % 3 == 0) & (dt.day == 1)
    elif period == "year":
        return (dt.dayofyear == 1).values
    else:
        raise ValueError(f"Unknown period: {period}")


def _compute_to_date_return(
    close: "xr.DataArray",
    period: str,
    eps: float = 1e-12
) -> "xr.DataArray":
    """
    Compute calendar-aware to-date return (WTD/MTD/QTD/YTD).
    
    This finds the start of the current period and computes return since then.
    For days where we can't determine period start, falls back to rolling approximation.
    """
    _require_xr()
    
    times = close.time.values
    
    # Try calendar-aware computation
    try:
        period_starts = _get_period_start_mask(times, period)
        
        # Convert to DataFrame for easier manipulation
        df = close.transpose("time", "asset").to_pandas()
        
        # For each row, find the most recent period start
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        
        last_start_idx = 0
        for i, is_start in enumerate(period_starts):
            if is_start:
                last_start_idx = i
            if i > 0 and last_start_idx < i:
                # Return since period start
                start_prices = df.iloc[last_start_idx]
                current_prices = df.iloc[i]
                result.iloc[i] = (current_prices / (start_prices + eps) - 1.0)
            else:
                result.iloc[i] = 0.0
        
        return xr.DataArray(
            result.fillna(0.0).values,
            coords=close.coords,
            dims=close.dims
        ).transpose("time", "asset")
        
    except Exception:
        # Fallback to rolling approximation
        approx_days = {
            "week": 5,
            "month": 21,
            "quarter": 63,
            "year": 252
        }
        return _compute_period_return(close, approx_days.get(period, 21))


# ==============================================================================
# HLOC Aggregation Functions
# ==============================================================================

def aggregate_hloc_to_timeframe(
    high: "xr.DataArray",
    low: "xr.DataArray",
    open_: "xr.DataArray",
    close: "xr.DataArray",
    vol: "xr.DataArray",
    window: int,
) -> Dict[str, "xr.DataArray"]:
    """
    Aggregate daily HLOC to a larger timeframe using rolling windows.
    
    Returns:
        Dict with 'high', 'low', 'open', 'close', 'vol' aggregated over window
    """
    _require_xr()
    
    return {
        "high": _rolling_max(high, window),
        "low": _rolling_min(low, window),
        "open": open_.shift(time=window - 1),  # Open from start of period
        "close": close,  # Current close
        "vol": _rolling_sum(vol, window),  # Total volume over period
        "range": _rolling_max(high, window) - _rolling_min(low, window),
    }


# ==============================================================================
# IC Matrix Computation
# ==============================================================================

def compute_mtf_ic_matrix(
    returns_by_tf: Dict[str, "xr.DataArray"],
    fwd_returns: "xr.DataArray",
    ic_window: int = 63,
    eps: float = 1e-12,
) -> "xr.DataArray":
    """
    Compute rolling IC for each timeframe's returns vs forward returns.
    
    Args:
        returns_by_tf: Dict mapping timeframe name to return DataArray
        fwd_returns: Forward returns to predict (time, asset)
        ic_window: Rolling window for IC calculation
    
    Returns:
        IC matrix with dims (timeframe, time)
    """
    _require_xr()
    
    ics = []
    tf_names = []
    
    for tf_name, tf_returns in returns_by_tf.items():
        # Align
        r, fwd = xr.align(tf_returns, fwd_returns, join="inner")
        
        # Rank-based IC (Spearman)
        r_rank = r.rank("asset", pct=False).fillna(0.0)
        fwd_rank = fwd.rank("asset", pct=False).fillna(0.0)
        
        # Rolling correlation
        r_df = r_rank.transpose("time", "asset").to_pandas()
        fwd_df = fwd_rank.transpose("time", "asset").to_pandas()
        
        ic_series = pd.Series(index=r_df.index, dtype=float)
        for i in range(ic_window, len(r_df)):
            window_r = r_df.iloc[i - ic_window:i].values.flatten()
            window_fwd = fwd_df.iloc[i - ic_window:i].values.flatten()
            
            # Filter valid
            mask = np.isfinite(window_r) & np.isfinite(window_fwd)
            if mask.sum() > 10:
                ic_series.iloc[i] = safe_corrcoef(window_r[mask], window_fwd[mask])
        
        ic_da = xr.DataArray(
            ic_series.fillna(0.0).values,
            coords={"time": r.time},
            dims=["time"]
        )
        ics.append(ic_da)
        tf_names.append(tf_name)
    
    return xr.concat(ics, dim="timeframe").assign_coords(timeframe=tf_names)


# ==============================================================================
# MTFMomentumParams Configuration
# ==============================================================================

@dataclass
class MTFMomentumParams:
    """Configuration for multi-timeframe momentum factors."""
    
    # Normalization
    eps: float = 1e-12
    zclip: float = 5.0
    robust_cs_z: bool = True
    
    # Timeframe windows (trading days)
    WEEK: int = 5
    MONTH: int = 21
    QUARTER: int = 63
    YEAR: int = 252
    
    # IC computation
    IC_WINDOW: int = 63          # Rolling IC window
    IC_EWMA_LAMBDA: float = 0.94  # IC smoothing
    
    # Momentum blending
    SKIP_RECENT_DAYS: int = 5    # Skip recent days for momentum (avoid reversal)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "zclip": self.zclip,
            "robust_cs_z": self.robust_cs_z,
            "WEEK": self.WEEK,
            "MONTH": self.MONTH,
            "QUARTER": self.QUARTER,
            "YEAR": self.YEAR,
            "IC_WINDOW": self.IC_WINDOW,
            "IC_EWMA_LAMBDA": self.IC_EWMA_LAMBDA,
            "SKIP_RECENT_DAYS": self.SKIP_RECENT_DAYS,
        }


# ==============================================================================
# MTFMomentumLibrary - Main computation class
# ==============================================================================

class MTFMomentumLibrary:
    """
    Multi-Timeframe Momentum Factor Library.
    
    Computes momentum factors across Weekly/Monthly/Quarterly/Yearly timeframes
    with IC-based adaptive weighting.
    
    Usage:
        mtf = MTFMomentumLibrary(market_data)
        F, artifacts = mtf.compute()
        
        # Or with custom params
        params = MTFMomentumParams(MONTH=22, IC_WINDOW=42)
        mtf = MTFMomentumLibrary(market_data, params=params)
    """
    
    def __init__(
        self,
        market_data: Any,
        *,
        params: Optional[MTFMomentumParams] = None,
        factors: Sequence[str] = MTF_CORE_FACTORS,
        asset_dim: str = "asset",
    ):
        _require_xr()
        
        self.params = params or MTFMomentumParams()
        self.factors = list(factors)
        self.asset_dim = asset_dim
        
        # Extract data
        if isinstance(market_data, xr.Dataset):
            self.data = market_data
        elif hasattr(market_data, "data") and isinstance(getattr(market_data, "data"), xr.Dataset):
            self.data = getattr(market_data, "data")
        else:
            raise TypeError("market_data must be xr.Dataset or bundle-like with .data")
        
        # Validate required variables
        for v in ("close", "high", "low", "vol"):
            if v not in self.data:
                raise ValueError(f"Dataset missing required variable '{v}'")
        
        self.data = self.data.transpose("time", asset_dim, missing_dims="ignore")
        self._artifacts: Dict[str, Any] = {}
        
        # Precompute common series
        self.close = self.data["close"].fillna(0.0)
        self.high = self.data["high"].fillna(0.0)
        self.low = self.data["low"].fillna(0.0)
        self.vol = self.data["vol"].fillna(0.0)
        
        # Handle open (may not exist)
        if "open" in self.data:
            self.open = self.data["open"].fillna(0.0)
        else:
            self.open = self.close.shift(time=1).fillna(0.0)
        
        # Daily returns
        self.ret = (self.close / self.close.shift(time=1) - 1.0).fillna(0.0).clip(-0.999, 10.0)
        
        # Market return (equal-weighted)
        self.mkt_ret = self.ret.mean("asset")
        
        # Pre-aggregate HLOC to each timeframe
        self._aggregate_hloc()
        
        # Precompute period returns
        self._compute_period_returns()
    
    def _aggregate_hloc(self) -> None:
        """Pre-aggregate HLOC data to each timeframe."""
        p = self.params
        
        self.tf_hloc = {
            "weekly": aggregate_hloc_to_timeframe(
                self.high, self.low, self.open, self.close, self.vol, p.WEEK
            ),
            "monthly": aggregate_hloc_to_timeframe(
                self.high, self.low, self.open, self.close, self.vol, p.MONTH
            ),
            "quarterly": aggregate_hloc_to_timeframe(
                self.high, self.low, self.open, self.close, self.vol, p.QUARTER
            ),
            "yearly": aggregate_hloc_to_timeframe(
                self.high, self.low, self.open, self.close, self.vol, p.YEAR
            ),
        }
        
        self._artifacts["tf_hloc"] = self.tf_hloc
    
    def _compute_period_returns(self) -> None:
        """Precompute returns for each timeframe."""
        p = self.params
        
        # Completed period returns (rolling)
        self.ret_1w = _compute_period_return(self.close, p.WEEK)
        self.ret_1m = _compute_period_return(self.close, p.MONTH)
        self.ret_1q = _compute_period_return(self.close, p.QUARTER)
        self.ret_1y = _compute_period_return(self.close, p.YEAR)
        
        # To-date returns (calendar-aware when possible)
        try:
            self.ret_wtd = _compute_to_date_return(self.close, "week", p.eps)
            self.ret_mtd = _compute_to_date_return(self.close, "month", p.eps)
            self.ret_qtd = _compute_to_date_return(self.close, "quarter", p.eps)
            self.ret_ytd = _compute_to_date_return(self.close, "year", p.eps)
        except Exception:
            # Fallback to rolling approximations
            self.ret_wtd = self.ret_1w
            self.ret_mtd = self.ret_1m
            self.ret_qtd = self.ret_1q
            self.ret_ytd = self.ret_1y
        
        # Store for IC computation
        self.returns_by_tf = {
            "1w": self.ret_1w,
            "1m": self.ret_1m,
            "1q": self.ret_1q,
            "1y": self.ret_1y,
            "wtd": self.ret_wtd,
            "mtd": self.ret_mtd,
            "qtd": self.ret_qtd,
            "ytd": self.ret_ytd,
        }
        
        self._artifacts["returns_by_tf"] = self.returns_by_tf
    
    def _cs_z(self, da: "xr.DataArray") -> "xr.DataArray":
        """Cross-sectional z-score normalization."""
        if self.params.robust_cs_z:
            return _robust_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)
        return safe_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)
    
    def compute(self) -> Tuple["xr.DataArray", Dict[str, Any]]:
        """
        Compute all requested MTF momentum factors.
        
        Returns:
            F: DataArray with dims (factor, time, asset)
            artifacts: Dict of intermediate computations
        """
        Fs: List["xr.DataArray"] = []
        names: List[str] = []
        
        for name in self.factors:
            fn = getattr(self, f"_factor_{name}", None)
            if fn is None:
                raise ValueError(f"Unknown factor '{name}'. Implement _factor_{name}.")
            raw = fn().transpose("time", self.asset_dim).fillna(0.0)
            z = self._cs_z(raw).fillna(0.0)
            z.name = name
            Fs.append(z)
            names.append(name)
        
        F = xr.concat(Fs, dim="factor").assign_coords(factor=names)
        F = F.transpose("factor", "time", self.asset_dim)
        F.name = "F_mtf"
        
        return F, dict(self._artifacts)
    
    def compute_ic_matrix(
        self,
        fwd_returns: "xr.DataArray",
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, "xr.DataArray"]:
        """
        Compute IC matrix: which timeframe best predicts forward returns.
        
        Args:
            fwd_returns: Forward returns to predict
            horizons: Forward horizons to test (default: [1, 5, 21])
        
        Returns:
            Dict of IC matrices for each horizon
        """
        if horizons is None:
            horizons = [1, 5, 21]
        
        result = {}
        
        for h in horizons:
            # Shift forward returns
            fwd = fwd_returns.shift(time=-h).fillna(0.0)
            
            ic_matrix = compute_mtf_ic_matrix(
                self.returns_by_tf,
                fwd,
                ic_window=self.params.IC_WINDOW,
                eps=self.params.eps,
            )
            result[f"ic_h{h}"] = ic_matrix
        
        self._artifacts["ic_matrices"] = result
        return result
    
    # ==========================================================================
    # TO-DATE RETURN FACTORS
    # ==========================================================================
    
    def _factor_ret_wtd(self) -> "xr.DataArray":
        """Week-to-date return."""
        return self.ret_wtd
    
    def _factor_ret_mtd(self) -> "xr.DataArray":
        """Month-to-date return."""
        return self.ret_mtd
    
    def _factor_ret_qtd(self) -> "xr.DataArray":
        """Quarter-to-date return."""
        return self.ret_qtd
    
    def _factor_ret_ytd(self) -> "xr.DataArray":
        """Year-to-date return."""
        return self.ret_ytd
    
    # ==========================================================================
    # COMPLETED PERIOD RETURN FACTORS
    # ==========================================================================
    
    def _factor_ret_1w(self) -> "xr.DataArray":
        """Last complete week return (5 trading days)."""
        return self.ret_1w
    
    def _factor_ret_1m(self) -> "xr.DataArray":
        """Last 21 trading days return (~1 month)."""
        return self.ret_1m
    
    def _factor_ret_1q(self) -> "xr.DataArray":
        """Last 63 trading days return (~1 quarter)."""
        return self.ret_1q
    
    def _factor_ret_1y(self) -> "xr.DataArray":
        """Last 252 trading days return (~1 year)."""
        return self.ret_1y
    
    def _factor_ret_2_12m(self) -> "xr.DataArray":
        """Months 2-12 return (classic momentum, skip recent month)."""
        p = self.params
        # Return from 252 days ago to 21 days ago
        ret_12m = self.close / self.close.shift(time=p.YEAR) - 1.0
        ret_1m = self.close / self.close.shift(time=p.MONTH) - 1.0
        return (ret_12m - ret_1m).fillna(0.0)
    
    def _factor_ret_1_3m(self) -> "xr.DataArray":
        """Months 1-3 return (medium-term momentum)."""
        return self.ret_1q
    
    # ==========================================================================
    # MOMENTUM ALIGNMENT / CONFLUENCE FACTORS
    # ==========================================================================
    
    def _factor_mtf_alignment(self) -> "xr.DataArray":
        """
        Momentum direction alignment across W/M/Q/Y.
        
        Score: count of timeframes with positive returns (0-4).
        High values = all timeframes agree on direction.
        """
        w_pos = (self.ret_1w > 0).astype(float)
        m_pos = (self.ret_1m > 0).astype(float)
        q_pos = (self.ret_1q > 0).astype(float)
        y_pos = (self.ret_1y > 0).astype(float)
        
        return (w_pos + m_pos + q_pos + y_pos).fillna(0.0)
    
    def _factor_mtf_alignment_strength(self) -> "xr.DataArray":
        """
        Alignment weighted by return magnitude.
        
        Returns are signed and summed - strong alignment = large magnitude.
        Normalization by vol to make comparable.
        """
        p = self.params
        
        # Normalize each return by its typical magnitude
        w_z = self.ret_1w / (self.ret_1w.rolling(time=p.QUARTER, min_periods=10).std() + p.eps)
        m_z = self.ret_1m / (self.ret_1m.rolling(time=p.QUARTER, min_periods=10).std() + p.eps)
        q_z = self.ret_1q / (self.ret_1q.rolling(time=p.YEAR, min_periods=20).std() + p.eps)
        
        # Sum of z-scores (if aligned, this is large; if mixed, cancels)
        return (w_z + m_z + q_z).fillna(0.0).clip(-10, 10)
    
    def _factor_trend_consistency(self) -> "xr.DataArray":
        """
        Consistency of trend across timeframes.
        
        Low std of returns = consistent trend
        Inverted so higher = more consistent
        """
        p = self.params
        
        # Stack returns
        ret_stack = xr.concat(
            [self.ret_1w, self.ret_1m, self.ret_1q],
            dim="tf"
        ).assign_coords(tf=["1w", "1m", "1q"])
        
        # Std across timeframes
        tf_std = ret_stack.std("tf")
        
        # Invert: low std = consistent = high score
        return (1.0 / (tf_std + p.eps)).fillna(0.0)
    
    # ==========================================================================
    # IC-WEIGHTED ADAPTIVE MOMENTUM FACTORS
    # ==========================================================================
    
    def _factor_mtf_ic_momentum(self) -> "xr.DataArray":
        """
        IC-weighted momentum composite.
        
        Dynamically weights W/M/Q returns by their rolling IC.
        Falls back to equal weights if IC not available.
        """
        p = self.params
        
        # Simple IC proxy: use autocorrelation of returns as IC estimate
        # (actual IC requires forward returns which creates look-ahead)
        
        # Use lagged returns' correlation with current returns as IC proxy
        w_ic = self.ret_1w.shift(time=p.WEEK).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1w).fillna(0.0)
        m_ic = self.ret_1m.shift(time=p.MONTH).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1m).fillna(0.0)
        q_ic = self.ret_1q.shift(time=p.QUARTER).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1q).fillna(0.0)
        
        # Positive IC only
        w_ic = w_ic.clip(min=0.0)
        m_ic = m_ic.clip(min=0.0)
        q_ic = q_ic.clip(min=0.0)
        
        # Normalize weights
        total_ic = w_ic + m_ic + q_ic + p.eps
        w_w = w_ic / total_ic
        w_m = m_ic / total_ic
        w_q = q_ic / total_ic
        
        # Weighted combination
        return (w_w * self.ret_1w + w_m * self.ret_1m + w_q * self.ret_1q).fillna(0.0)
    
    def _factor_mtf_ic_regime(self) -> "xr.DataArray":
        """
        Dominant timeframe regime (which has highest IC).
        
        Returns:
            1 = Weekly dominant
            2 = Monthly dominant
            3 = Quarterly dominant
        """
        p = self.params
        
        # IC proxies
        w_ic = np.abs(self.ret_1w.shift(time=p.WEEK).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1w)).fillna(0.0)
        m_ic = np.abs(self.ret_1m.shift(time=p.MONTH).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1m)).fillna(0.0)
        q_ic = np.abs(self.ret_1q.shift(time=p.QUARTER).rolling(time=p.IC_WINDOW, min_periods=10).corr(self.ret_1q)).fillna(0.0)
        
        # Stack and find argmax
        regime = xr.where(w_ic >= m_ic, 1.0, 0.0)
        regime = xr.where((m_ic > w_ic) & (m_ic >= q_ic), 2.0, regime)
        regime = xr.where((q_ic > w_ic) & (q_ic > m_ic), 3.0, regime)
        
        return regime.fillna(1.0)
    
    # ==========================================================================
    # ACCELERATION / RATE OF CHANGE FACTORS
    # ==========================================================================
    
    def _factor_momentum_acceleration(self) -> "xr.DataArray":
        """
        Short-term vs long-term momentum (acceleration signal).
        
        Positive = momentum accelerating (short > long)
        Negative = momentum decelerating
        """
        p = self.params
        
        # Normalize by vol for comparability
        w_vol = self.ret_1w.rolling(time=p.QUARTER, min_periods=10).std() + p.eps
        m_vol = self.ret_1m.rolling(time=p.QUARTER, min_periods=10).std() + p.eps
        
        w_z = self.ret_1w / w_vol
        m_z = self.ret_1m / m_vol
        
        return (w_z - m_z).fillna(0.0).clip(-5, 5)
    
    def _factor_mtf_slope(self) -> "xr.DataArray":
        """
        Slope of returns across timeframes.
        
        Fits a line through (log days, return) to detect trend in momentum.
        Positive slope = improving momentum with time
        Negative slope = deteriorating momentum
        """
        p = self.params
        
        # Use normalized returns
        w_norm = self.ret_1w / (np.abs(self.ret_1w).rolling(time=63, min_periods=10).mean() + p.eps)
        m_norm = self.ret_1m / (np.abs(self.ret_1m).rolling(time=63, min_periods=10).mean() + p.eps)
        q_norm = self.ret_1q / (np.abs(self.ret_1q).rolling(time=63, min_periods=10).mean() + p.eps)
        
        # X values: log(days) for each timeframe
        # log(5)≈1.6, log(21)≈3.0, log(63)≈4.1
        x = np.array([np.log(p.WEEK), np.log(p.MONTH), np.log(p.QUARTER)])
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        
        # Stack returns
        y_stack = xr.concat([w_norm, m_norm, q_norm], dim="tf")
        y_mean = y_stack.mean("tf")
        
        # Slope = Σ(x - x_mean)(y - y_mean) / Σ(x - x_mean)²
        cov = (
            (x[0] - x_mean) * (w_norm - y_mean) +
            (x[1] - x_mean) * (m_norm - y_mean) +
            (x[2] - x_mean) * (q_norm - y_mean)
        )
        
        slope = (cov / (x_var + p.eps)).fillna(0.0).clip(-5, 5)
        return slope
    
    # ==========================================================================
    # HLOC-ENHANCED MOMENTUM FACTORS
    # ==========================================================================
    
    def _factor_hloc_range_expansion(self) -> "xr.DataArray":
        """
        Weekly range vs monthly range (breakout signal).
        
        High values = weekly range is large relative to monthly
        (potential breakout / high volatility regime)
        """
        p = self.params
        
        weekly_range = self.tf_hloc["weekly"]["range"]
        monthly_range = self.tf_hloc["monthly"]["range"]
        
        return (weekly_range / (monthly_range + p.eps)).fillna(0.0).clip(0, 3)
    
    def _factor_hloc_close_position(self) -> "xr.DataArray":
        """
        Close position within monthly HLOC range.
        
        1.0 = at high (bullish)
        0.0 = at low (bearish)
        0.5 = middle
        """
        p = self.params
        
        monthly = self.tf_hloc["monthly"]
        h = monthly["high"]
        l = monthly["low"]
        c = monthly["close"]
        
        return ((c - l) / (h - l + p.eps)).fillna(0.5).clip(0, 1)
    
    def _factor_hloc_momentum_quality(self) -> "xr.DataArray":
        """
        Momentum quality: returns supported by HLOC confirmation.
        
        High quality = positive return AND close near high
        Low quality = positive return but close near low (weak buying)
        """
        p = self.params
        
        # Monthly return
        ret = self.ret_1m
        
        # Close position in range
        close_pos = self._factor_hloc_close_position()
        
        # Quality = return * close_position
        # Positive return + high close = strong quality
        # Positive return + low close = weak quality
        return (ret * (close_pos - 0.5) * 2).fillna(0.0)
    
    # ==========================================================================
    # RELATIVE MOMENTUM FACTORS
    # ==========================================================================
    
    def _factor_relative_mtf_momentum(self) -> "xr.DataArray":
        """
        MTF momentum relative to market (alpha component).
        """
        p = self.params
        
        # Asset momentum
        asset_mom = self._factor_mtf_ic_momentum()
        
        # Market momentum
        mkt_1m = _sma(self.mkt_ret, p.MONTH)
        
        return (asset_mom - mkt_1m.broadcast_like(asset_mom)).fillna(0.0)
    
    def _factor_sector_relative_mtf(self) -> "xr.DataArray":
        """
        MTF momentum vs universe mean (cross-sectional).
        """
        mom = self._factor_mtf_ic_momentum()
        universe_mean = mom.mean("asset")
        return (mom - universe_mean.broadcast_like(mom)).fillna(0.0)


# ==============================================================================
# Convenience Functions
# ==============================================================================

def get_mtf_factor_info(factor_name: str) -> Optional[MTFFactorDefinition]:
    """Get metadata for an MTF factor."""
    return MTF_FACTOR_REGISTRY.get(factor_name)


def get_mtf_factors_by_category(category: str) -> List[str]:
    """Get all MTF factors in a category."""
    return [
        name for name, defn in MTF_FACTOR_REGISTRY.items()
        if defn.category == category
    ]


def get_mtf_dashboard_config() -> Dict[str, Any]:
    """
    Get complete MTF factor configuration for dashboard display.
    """
    categories = list(set(d.category for d in MTF_FACTOR_REGISTRY.values()))
    
    return {
        "categories": sorted(categories),
        "factors": {
            name: defn.to_dict()
            for name, defn in MTF_FACTOR_REGISTRY.items()
        },
        "factor_sets": {
            "all": list(MTF_FACTORS),
            "core": list(MTF_CORE_FACTORS),
            "to_date": get_mtf_factors_by_category("to_date_returns"),
            "period": get_mtf_factors_by_category("period_returns"),
            "alignment": get_mtf_factors_by_category("alignment"),
            "ic_weighted": get_mtf_factors_by_category("ic_weighted"),
            "acceleration": get_mtf_factors_by_category("acceleration"),
            "hloc_enhanced": get_mtf_factors_by_category("hloc_enhanced"),
        },
        "params": MTFMomentumParams().to_dict(),
    }
