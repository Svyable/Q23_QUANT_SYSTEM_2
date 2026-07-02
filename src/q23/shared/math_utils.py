"""q23.shared.math_utils

Math primitives shared by BOTH strategy (xarray-heavy) and dashboard (pandas/numpy-heavy).

Design goals
- Works with numpy arrays, pandas Series/DataFrames, and xarray DataArray
- Avoids side effects and heavy imports (qnt/streamlit/matplotlib)
- Keeps numeric guardrails (NaN/inf safety, eps handling, clipping)

Conventions
- For xarray functions, default dimension conventions match the strategy:
    cross-sectional dim: "asset"
    time dim: "time"
"""

from __future__ import annotations

from typing import Union, Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"]


def _to_numpy(x) -> np.ndarray:
    """Convert pandas/numpy-like to numpy float array (best-effort)."""
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy(dtype=float, copy=False)
    return np.asarray(x, dtype=float)


# =============================================================================
# Z-scores
# =============================================================================

def robust_zscore_xr(
    x: "xr.DataArray",
    dim: str = "asset",
    eps: float = 1e-12,
    clip: float = 5.0,
) -> "xr.DataArray":
    """Robust z-score for xarray using median/MAD across `dim`.

    Returns clipped to [-clip, clip], NaNs replaced with 0.
    """
    if xr is None:
        raise ImportError("xarray is required for robust_zscore_xr")
    med = x.median(dim=dim, skipna=True)
    mad = np.abs(x - med).median(dim=dim, skipna=True)
    z = (x - med) / (mad * 1.4826 + eps)
    return z.fillna(0).clip(-clip, clip)


def safe_zscore_np(x: ArrayLike, eps: float = 1e-12, clip: float = 5.0) -> np.ndarray:
    """Robust-ish z-score for numpy/pandas using median/MAD.

    Always returns numpy array. NaNs/infs -> 0. Clipped to [-clip, clip].
    """
    x_np = _to_numpy(x).astype(float, copy=False)
    med = np.nanmedian(x_np)
    mad = np.nanmedian(np.abs(x_np - med))
    if not np.isfinite(mad) or mad < eps:
        return np.zeros_like(x_np, dtype=float)
    z = (x_np - med) / (mad * 1.4826 + eps)
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -clip, clip)


def safe_zscore(
    x: Union[ArrayLike, "xr.DataArray"],
    dim: str = "asset",
    eps: float = 1e-12,
    clip: float = 5.0,
):
    """Unified safe z-score.

    - If x is xarray.DataArray -> robust median/MAD across `dim`
    - Else -> median/MAD over the flattened numpy array
    """
    if xr is not None and isinstance(x, xr.DataArray):
        return robust_zscore_xr(x, dim=dim, eps=eps, clip=clip)
    return safe_zscore_np(x, eps=eps, clip=clip)


# =============================================================================
# EWMA
# =============================================================================

def ewma_1d(series: Union[np.ndarray, list], lam: float, eps: float = 1e-12) -> np.ndarray:
    """EWMA for 1D series with normalization to reduce startup bias."""
    x = np.asarray(series, dtype=float)
    s = 0.0
    w = 0.0
    out = np.zeros_like(x, dtype=float)

    for i, v in enumerate(x):
        if np.isfinite(v):
            s = lam * s + (1.0 - lam) * float(v)
            w = lam * w + (1.0 - lam)
        out[i] = s / (w + eps) if w > eps else 0.0
    return out


def ewma_vol_1d(returns: Union[np.ndarray, list], lam: float = 0.95) -> np.ndarray:
    """EWMA volatility (std) for 1D returns. Not annualized."""
    r = np.asarray(returns, dtype=float)
    var = 0.0
    out = np.zeros_like(r, dtype=float)
    for i, rv in enumerate(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)):
        var = lam * var + (1.0 - lam) * float(rv) * float(rv)
        out[i] = np.sqrt(max(var, 0.0))
    return out


# =============================================================================
# Capped simplex projection
# =============================================================================

def project_to_capped_simplex(
    v: Union[np.ndarray, list],
    T: float,
    lo: Union[np.ndarray, list],
    hi: Union[np.ndarray, list],
    eps: float = 1e-12,
    max_iter: int = 60,
) -> np.ndarray:
    """Project vector v onto the capped simplex: {w : sum(w)=T, lo<=w<=hi}."""
    v = np.asarray(v, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    if v.shape != lo.shape or v.shape != hi.shape:
        raise ValueError("v, lo, hi must have the same shape")

    cap = np.maximum(hi - lo, 0.0)
    T_prime = float(T) - float(np.sum(lo))

    if T_prime <= eps:
        return lo.copy()
    cap_sum = float(np.sum(cap))
    if T_prime >= cap_sum - eps:
        return (lo + cap).copy()

    a = v - lo
    tau_lo = float(np.min(a - cap))
    tau_hi = float(np.max(a))

    def sum_clip(tau: float) -> float:
        y = np.clip(a - tau, 0.0, cap)
        return float(np.sum(y))

    for _ in range(max_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        if sum_clip(tau) > T_prime:
            tau_lo = tau
        else:
            tau_hi = tau

    tau = 0.5 * (tau_lo + tau_hi)
    y = np.clip(a - tau, 0.0, cap)
    w = lo + y

    resid = float(T) - float(np.sum(w))
    if abs(resid) > 1e-10:
        free = (w > lo + eps) & (w < hi - eps)
        if np.any(free):
            w[free] += resid / float(np.sum(free))
            w = np.clip(w, lo, hi)

    return w


# =============================================================================
# Softmax tilt (numpy)
# =============================================================================

def softmax_tilt(
    weights: Union[np.ndarray, list],
    scores: Union[np.ndarray, list],
    sel_mask: Union[np.ndarray, list],
    alpha: float,
    eps: float = 1e-12,
    clip_z: float = 3.0,
) -> np.ndarray:
    """Softmax-style tilt within sel_mask, then re-normalize within sel_mask."""
    w = np.asarray(weights, dtype=float).copy()
    s = np.asarray(scores, dtype=float)
    m = np.asarray(sel_mask, dtype=bool)

    if not np.any(m):
        return w

    if alpha == 0.0:
        tot = float(np.sum(w[m]))
        if tot > eps:
            w[m] = w[m] / (tot + eps)
        return w

    s_sel = s[m]
    mu = float(np.nanmean(s_sel)) if np.any(np.isfinite(s_sel)) else 0.0
    sd = float(np.nanstd(s_sel)) + eps
    z = np.clip((s_sel - mu) / sd, -clip_z, clip_z)

    tilt = np.exp(alpha * z)
    w_sel = w[m] * tilt
    ssum = float(np.sum(w_sel))

    if ssum > eps:
        w[m] = w_sel / (ssum + eps)

    return w


# =============================================================================
# Safe correlation
# =============================================================================

def safe_corrcoef(x: np.ndarray, y: Optional[np.ndarray] = None, eps: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Safe correlation coefficient that handles zero standard deviation.
    
    Args:
        x: First array (1D or 2D)
        y: Optional second array (1D). If None, computes correlation matrix of x
        eps: Small value to avoid division by zero
    
    Returns:
        If y is None and x is 2D: correlation matrix (n x n)
        If y is provided: scalar correlation coefficient
        If x is 1D and y is None: 1.0 (self-correlation)
    """
    x = np.asarray(x, dtype=float)
    
    # Handle 2D case (correlation matrix)
    if y is None and x.ndim == 2:
        # Check if we have enough data points for correlation computation
        if x.shape[0] < 2:
            # Not enough rows for correlation - return identity matrix
            n = x.shape[1]
            return np.eye(n, dtype=float)

        # Check for constant columns (zero std)
        stds = np.nanstd(x, axis=0, ddof=1)
        valid = stds > eps

        if not np.any(valid):
            # All columns are constant
            n = x.shape[1]
            return np.eye(n, dtype=float)

        # Only compute correlation for non-constant columns
        x_valid = x[:, valid]

        if x_valid.shape[1] < 2:
            return np.array([[1.0]], dtype=float)
        
        # Compute correlation with warnings suppressed
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(x_valid, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Expand back to full size with zeros for constant columns
        if np.all(valid):
            return corr
        
        full_corr = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        valid_idx = np.where(valid)[0]
        for i, idx_i in enumerate(valid_idx):
            for j, idx_j in enumerate(valid_idx):
                full_corr[idx_i, idx_j] = corr[i, j]
        # Set diagonal to 1.0
        np.fill_diagonal(full_corr, 1.0)
        return full_corr
    
    # Handle 1D case
    if x.ndim == 1:
        if y is None:
            # Self-correlation
            return 1.0
        
        y = np.asarray(y, dtype=float)
        
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        
        # Check if we have enough data points
        if len(x) < 2:
            return 0.0  # Cannot compute correlation with < 2 points

        # Check for constant arrays
        x_std = np.nanstd(x, ddof=1)
        y_std = np.nanstd(y, ddof=1)
        
        if x_std < eps or y_std < eps:
            # At least one is constant
            if x_std < eps and y_std < eps:
                # Both constant - check if same value
                if np.abs(np.nanmean(x) - np.nanmean(y)) < eps:
                    return 1.0
                return 0.0
            return 0.0
        
        # Both have variance - compute correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(x, y)[0, 1]
            return float(np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0))
    
    raise ValueError(f"Unsupported input shape: x.shape={x.shape}, y={y}")


# =============================================================================
# Small numeric helpers
# =============================================================================

def nan_to_num0(x: Union[np.ndarray, list], copy: bool = True) -> np.ndarray:
    """Convert NaN/inf to 0.0."""
    arr = np.asarray(x, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=copy)
