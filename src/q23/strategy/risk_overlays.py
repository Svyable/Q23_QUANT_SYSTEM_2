from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.risk_overlays")


@dataclass(frozen=True)
class RiskOverlayParams:
    """Guardrails to keep realized risk within bounds."""

    vol_window: int = 63
    target_vol: float = 0.16
    max_vol: float = 0.22
    dd_window: int = 126
    dd_stretch: float = 0.60
    dd_floor: float = 0.55
    clip: Tuple[float, float] = (0.5, 1.25)
    eps: float = 1e-12


def _roll_vol(returns: "xr.DataArray", win: int, eps: float) -> "xr.DataArray":
    _require_xr()
    return returns.rolling(time=win, min_periods=1).std().fillna(0.0) * np.sqrt(252.0) + eps


def compute_vol_overlay(
    returns: "xr.DataArray",
    *,
    params: RiskOverlayParams,
) -> "xr.DataArray":
    """Scale exposure down when realized vol exceeds target."""
    _require_xr()
    vol = _roll_vol(returns, params.vol_window, params.eps)
    target = params.target_vol
    max_vol = params.max_vol

    scale = (target / vol).clip(min=params.clip[0], max=params.clip[1])
    scale = scale.where(vol <= max_vol, other=target / (max_vol + params.eps))
    return scale.fillna(1.0)


def compute_dd_overlay(
    returns: "xr.DataArray",
    *,
    params: RiskOverlayParams,
) -> "xr.DataArray":
    """Drawdown-aware throttle using market-level equity curve."""
    _require_xr()
    mkt = returns.fillna(0.0).mean("asset")
    growth = (1.0 + mkt).cumprod()
    peak = growth.rolling(time=params.dd_window, min_periods=1).max()
    dd = (peak - growth) / (peak + params.eps)
    throttle = (1.0 - params.dd_stretch * dd).clip(min=params.dd_floor, max=params.clip[1])
    return throttle.fillna(1.0)


def combined_overlay(
    returns: "xr.DataArray",
    *,
    params: RiskOverlayParams,
) -> "xr.DataArray":
    """Vol + drawdown overlay."""
    vol_ov = compute_vol_overlay(returns, params=params)
    dd_ov = compute_dd_overlay(returns, params=params)
    ov = (vol_ov * dd_ov).clip(min=params.clip[0], max=params.clip[1])
    return ov.fillna(1.0)


def apply_overlay(
    weights: "xr.DataArray",
    budget: "xr.DataArray",
    overlay: "xr.DataArray",
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Apply overlay to weights and budget while preserving sign and shape."""
    _require_xr()
    overlay_aligned = overlay
    if "time" not in overlay.dims:
        overlay_aligned = overlay.broadcast_like(weights)

    w = (weights * overlay_aligned).fillna(0.0)

    # Budget scaling uses time-only overlay (avg across assets if present)
    if "asset" in overlay_aligned.dims:
        overlay_budget = overlay_aligned.mean("asset")
    else:
        overlay_budget = overlay_aligned

    overlay_budget = overlay_budget.broadcast_like(budget)
    b = (budget * overlay_budget).fillna(0.0)
    return w, b
