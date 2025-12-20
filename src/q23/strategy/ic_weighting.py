"""q23.strategy.ic_weighting

Dynamic IC weighting for factor models (v4-compatible semantics).

Inputs
- F: xr.DataArray with dims (factor, time, asset)
- fwd_returns: xr.DataArray with dims (time, asset)

Outputs
- ic_raw: xr.DataArray(factor, time)   : cross-sectional rank IC (Spearman)
- ic_smooth: xr.DataArray(factor, time): EWMA-smoothed IC (shifted by 1 to avoid look-ahead)
- w: xr.DataArray(factor, time)        : non-negative factor weights summing to 1

Composite score
- score: xr.DataArray(time, asset) = sum_factor w[f,t] * F[f,t,a]

Design notes
- Uses xarray's ranking + correlation to compute a Spearman-style IC without scipy.
- Uses shared ewma_1d to smooth IC over time while remaining pure-python.
- Keeps logic explicit and easy to profile / cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.math_utils import ewma_1d


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.ic_weighting")


def _ewma_da(da: "xr.DataArray", lam: float, dim: str = "time") -> "xr.DataArray":
    """Apply ewma_1d along a dimension for each remaining index."""
    _require_xr()
    if dim not in da.dims:
        raise ValueError(f"dim '{dim}' not in da.dims={da.dims}")
    return xr.apply_ufunc(
        ewma_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="forbidden",
        kwargs={"lam": float(lam)},
        output_dtypes=[float],
    )


def compute_rank_ic(
    F: "xr.DataArray",
    fwd_returns: "xr.DataArray",
    *,
    asset_dim: str = "asset",
) -> "xr.DataArray":
    """Cross-sectional rank IC (Spearman) for each (factor, time).

    Implementation:
    - rank factor exposures and forward returns cross-sectionally
    - compute Pearson correlation of ranks across assets
    """
    _require_xr()
    if set(F.dims) != {"factor", "time", asset_dim}:
        # allow other orderings but require these dims
        needed = {"factor", "time", asset_dim}
        if not needed.issubset(set(F.dims)):
            raise ValueError(f"F must include dims {needed}, got {F.dims}")
        F = F.transpose("factor", "time", asset_dim)
    if set(fwd_returns.dims) != {"time", asset_dim}:
        needed = {"time", asset_dim}
        if not needed.issubset(set(fwd_returns.dims)):
            raise ValueError(f"fwd_returns must include dims {needed}, got {fwd_returns.dims}")
        fwd_returns = fwd_returns.transpose("time", asset_dim)

    # Align time/asset
    F, fwd = xr.align(F, fwd_returns, join="inner")

    Fr = F.rank(asset_dim, pct=False).fillna(0.0)
    Rr = fwd.rank(asset_dim, pct=False).fillna(0.0)

    # xarray.corr expects same dims; broadcast Rr over factor
    Rr_b = Rr.broadcast_like(Fr)
    ic = xr.corr(Fr, Rr_b, dim=asset_dim).fillna(0.0)
    ic.name = "ic_raw"
    return ic


def normalize_positive_weights(ic: "xr.DataArray", eps: float = 1e-12) -> "xr.DataArray":
    """Turn IC signal into non-negative weights that sum to 1 across factors per time."""
    _require_xr()
    raw = ic.clip(min=0.0)
    s = raw.sum("factor") + eps
    w = raw / s
    # if all zero, fall back to equal weights
    n = ic.sizes.get("factor", 1)
    w = w.where(s > eps, other=(0.0 * raw + 1.0 / float(n)))
    w.name = "factor_weights"
    return w


@dataclass
class ICWeightingParams:
    lam: float = 0.95       # EWMA lambda
    eps: float = 1e-12
    clip_ic: float = 0.20   # optional guardrail against extreme IC spikes
    use_positive_only: bool = True


@dataclass
class CorrICWeightingParams(ICWeightingParams):
    """IC weighting with correlation-aware diversification penalty."""

    corr_penalty_strength: float = 0.30  # 0=no penalty, 1=full penalty
    min_diversification: float = 0.20    # floor to prevent zeroing factors


class DynamicICWeighting:
    """Compute IC series and factor weights, and provide composite score."""

    def __init__(
        self,
        F: "xr.DataArray",
        fwd_returns: "xr.DataArray",
        *,
        params: Optional[ICWeightingParams] = None,
        asset_dim: str = "asset",
    ):
        _require_xr()
        self.F = F
        self.fwd_returns = fwd_returns
        self.asset_dim = asset_dim
        self.params = params or ICWeightingParams()

        # cache slots
        self._ic_raw: Optional["xr.DataArray"] = None
        self._ic_smooth: Optional["xr.DataArray"] = None
        self._w: Optional["xr.DataArray"] = None

    def ic_raw(self) -> "xr.DataArray":
        if self._ic_raw is None:
            self._ic_raw = compute_rank_ic(self.F, self.fwd_returns, asset_dim=self.asset_dim)
        return self._ic_raw

    def ic_smooth(self) -> "xr.DataArray":
        if self._ic_smooth is None:
            p = self.params
            ic = self.ic_raw().clip(-float(p.clip_ic), float(p.clip_ic))
            # shift by 1 to avoid look-ahead (today's weights use yesterday's IC)
            ic = ic.shift(time=1).fillna(0.0)
            self._ic_smooth = _ewma_da(ic, lam=float(p.lam), dim="time").fillna(0.0)
            self._ic_smooth.name = "ic_smooth"
        return self._ic_smooth

    def weights(self) -> "xr.DataArray":
        if self._w is None:
            p = self.params
            ic = self.ic_smooth()
            if p.use_positive_only:
                self._w = normalize_positive_weights(ic, eps=float(p.eps))
            else:
                # abs weighting, then normalize
                raw = np.abs(ic)
                s = raw.sum("factor") + float(p.eps)
                self._w = (raw / s).fillna(0.0)
                self._w.name = "factor_weights"
        return self._w

    def composite_score(self) -> "xr.DataArray":
        """Composite alpha score (time, asset)."""
        w = self.weights()
        # align to ensure matching coords
        w, F = xr.align(w, self.F, join="inner")
        score = (F * w).sum("factor").fillna(0.0)
        score.name = "composite_score"
        return score

    def artifacts(self) -> Tuple["xr.DataArray", "xr.DataArray", "xr.DataArray"]:
        """Return (ic_raw, ic_smooth, weights) for saving/debugging."""
        return self.ic_raw(), self.ic_smooth(), self.weights()


class CorrelationAwareICWeighting(DynamicICWeighting):
    """
    Dynamic IC weighting with correlation-aware diversification.

    Penalizes factors that are highly correlated with others to
    avoid overweighting redundant signals and improve Sharpe.
    """

    def __init__(
        self,
        F: "xr.DataArray",
        fwd_returns: "xr.DataArray",
        *,
        params: Optional[CorrICWeightingParams] = None,
        asset_dim: str = "asset",
    ):
        super().__init__(F, fwd_returns, params=params or CorrICWeightingParams(), asset_dim=asset_dim)

    def _correlation_penalty(self) -> "xr.DataArray":
        """Per-factor penalty in [min_diversification, 1]."""
        _require_xr()
        p: CorrICWeightingParams = self.params  # type: ignore[assignment]

        def _penalty(arr: np.ndarray) -> np.ndarray:
            # arr shape: (factor, asset)
            if arr.shape[1] < 2:
                return np.ones(arr.shape[0], dtype=float)
            corr = np.corrcoef(arr)
            np.fill_diagonal(corr, 0.0)
            penalty = 1.0 - np.nanmean(np.abs(corr), axis=1)
            return np.nan_to_num(penalty, nan=1.0, posinf=1.0, neginf=0.0)

        pen = xr.apply_ufunc(
            _penalty,
            self.F.transpose("time", "factor", self.asset_dim),
            input_core_dims=[["factor", self.asset_dim]],
            output_core_dims=[["factor"]],
            vectorize=True,
            dask="forbidden",
            output_dtypes=[float],
        )

        pen = pen.transpose("time", "factor")
        return pen.clip(min=float(p.min_diversification)).fillna(1.0)

    def weights(self) -> "xr.DataArray":
        if self._w is None:
            p: CorrICWeightingParams = self.params  # type: ignore[assignment]
            ic = self.ic_smooth()
            base_w = normalize_positive_weights(ic, eps=float(p.eps))

            penalty = self._correlation_penalty()
            base_w, penalty = xr.align(base_w, penalty, join="inner")

            # Blend: 1 - strength * (1 - penalty)
            blend = 1.0 - float(p.corr_penalty_strength) * (1.0 - penalty)
            blended = (base_w * blend).clip(min=0.0)

            s = blended.sum("factor") + float(p.eps)
            self._w = (blended / s).fillna(0.0)
            self._w.name = "factor_weights"
        return self._w
