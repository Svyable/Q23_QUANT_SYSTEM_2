"""q23.strategy.portfolio

Portfolio construction (long-only, long/short, equal-topK) with constraint handling.

This module is designed to:
- Scale: loops over *time* are OK; avoid per-asset python overhead.
- Be deterministic: explicit selection logic, pinned assets, stable ordering.
- Be testable: minimal external deps (xarray required; no qnt.* required).

Inputs
- scores: xr.DataArray(time, asset) : composite alpha score (higher = better)
- returns: xr.DataArray(time, asset) : daily returns aligned to scores
- liquidity: xr.DataArray(time, asset) or (time, asset) mask in [0,1]
- market_returns: optional pd.Series-like for risk throttle; otherwise mean(returns)

Outputs
- final_weights: xr.DataArray(time, asset) : gross-scaled weights
- budget: xr.DataArray(time) : gross target per day

Notes
- Constraint projection uses shared math_utils.project_to_capped_simplex.
- Softmax tilt uses shared math_utils.softmax_tilt.
- Transaction-cost decisioning is pluggable via tc_model.should_rebalance(prev, unit, expected_alpha).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Protocol

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception as e:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.math_utils import project_to_capped_simplex, softmax_tilt, ewma_vol_1d


# -------------------------------------------------------------------------
# Protocols (duck-typing for pluggability)
# -------------------------------------------------------------------------

class RegimeLike(Protocol):
    def params(self, date: Any) -> Dict[str, Any]: ...


class TransactionCostLike(Protocol):
    def should_rebalance(self, w0: np.ndarray, w1: np.ndarray, expected_alpha: float) -> bool: ...


# -------------------------------------------------------------------------
# Config container (kept explicit; avoid hard dependency on shared.config)
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioParams:
    # Constraints
    max_pos: float = 0.10
    min_pos: float = 0.002
    eps: float = 1e-12

    # Selection / smoothing
    topn_base: int = 20
    topn_volatile: int = 15
    score_smooth_win: int = 3
    weight_smooth_alpha: float = 0.30
    softmax_tilt_alpha: float = 0.85
    use_equal_topk: bool = False

    # Long/short
    long_seats: int = 12
    short_seats: int = 8
    fixed_long_frac: float = 0.60
    side_split_mode: str = "fixed"  # fixed | prop_mass | ic_weighted (treated as prop_mass)

    # Budget / leverage
    budget_mode: str = "unit"  # unit | use_leverage | fixed
    pm_gross_target: float = 1.00
    target_vol_base: float = 0.15
    target_vol_volatile: float = 0.10
    lev_cap: float = 1.5
    lev_min: float = 0.3

    # Risk throttle
    dd_win: int = 252
    risk_off_stretch: float = 0.60
    risk_off_floor: float = 0.50
    ewma_lambda: float = 0.95


@dataclass(frozen=True)
class PortfolioBuildResult:
    final_weights: "xr.DataArray"
    budget: "xr.DataArray"
    unit_weights: "xr.DataArray"  # before gross scaling
    meta: Dict[str, Any]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.portfolio")


def _sma_time(x: "xr.DataArray", win: int) -> "xr.DataArray":
    if win <= 1:
        return x
    return x.rolling(time=win, min_periods=1).mean()


def _as_market_returns(returns: "xr.DataArray") -> "pd.Series":
    if pd is None:
        raise ImportError("pandas is required for market return construction")
    # mean across assets; safe for NaNs
    m = returns.fillna(0).mean("asset").to_pandas()
    if not isinstance(m, pd.Series):
        m = pd.Series(m)
    return m.fillna(0.0)


def _risk_throttle(
    market_returns: "pd.Series",
    dd_win: int,
    risk_off_stretch: float,
    risk_off_floor: float,
    eps: float,
) -> "pd.Series":
    # drawdown on market equity curve
    m = market_returns.fillna(0.0)
    growth = (1.0 + m).cumprod()
    peak = growth.rolling(dd_win, min_periods=1).max()
    dd = (peak - growth) / (peak + eps)
    throttle = (1.0 - risk_off_stretch * dd).clip(lower=risk_off_floor, upper=1.0)
    return throttle.fillna(1.0)


def _target_vol_for_date(params: PortfolioParams, rg: Optional[RegimeLike], dt: Any) -> float:
    # Regime can override via params(dt) dict, but we keep a safe default.
    if rg is None:
        return float(params.target_vol_base)
    try:
        d = rg.params(dt) or {}
        # if regime provides a boolean high_vol or explicit target_vol use it
        if "target_vol" in d and d["target_vol"] is not None:
            return float(d["target_vol"])
        if bool(d.get("high_vol", False)):
            return float(params.target_vol_volatile)
    except Exception:
        pass
    return float(params.target_vol_base)


def _topn_for_date(params: PortfolioParams, rg: Optional[RegimeLike], dt: Any) -> int:
    if rg is None:
        return int(params.topn_base)
    try:
        d = rg.params(dt) or {}
        if "topn" in d and d["topn"] is not None:
            return int(d["topn"])
        if bool(d.get("high_vol", False)):
            return int(params.topn_volatile)
    except Exception:
        pass
    return int(params.topn_base)


def _regime_leverage_mult(rg: Optional[RegimeLike], dt: Any) -> float:
    if rg is None:
        return 1.0
    try:
        d = rg.params(dt) or {}
        return float(d.get("leverage_mult", 1.0))
    except Exception:
        return 1.0


# -------------------------------------------------------------------------
# Budget computation
# -------------------------------------------------------------------------

def compute_budget_series(
    unit_gross_weights: "xr.DataArray",
    returns: "xr.DataArray",
    params: PortfolioParams,
    *,
    market_returns: Optional["pd.Series"] = None,
    regime: Optional[RegimeLike] = None,
) -> "xr.DataArray":
    """Compute daily gross target series.

    unit_gross_weights: xr.DataArray(time, asset) non-negative weights whose sum is ~1.
    returns: xr.DataArray(time, asset) daily returns.

    For budget_mode:
    - unit/fixed -> constant pm_gross_target
    - use_leverage -> vol targeting with drawdown throttle and regime multiplier
    """
    _require_xr()
    if pd is None:
        raise ImportError("pandas is required for compute_budget_series")

    mode = str(params.budget_mode).lower()
    g = float(params.pm_gross_target)

    times = unit_gross_weights.time.values

    if mode in {"unit", "fixed"}:
        return xr.DataArray(
            np.full(len(times), g, dtype=float),
            coords={"time": unit_gross_weights.time},
            dims=["time"],
        )

    # use_leverage
    # Portfolio return from unit weights
    port_ret = (unit_gross_weights.shift(time=1).fillna(0.0) * returns.fillna(0.0)).sum("asset")
    pr = port_ret.to_pandas().fillna(0.0).values.astype(float, copy=False)

    # EWMA vol (annualized)
    vol = ewma_vol_1d(pr, lam=float(params.ewma_lambda)) * np.sqrt(252.0)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # Market throttle
    mkt = market_returns if market_returns is not None else _as_market_returns(returns)
    throttle = _risk_throttle(
        mkt.reindex(port_ret.time.to_pandas()).fillna(0.0),
        dd_win=int(params.dd_win),
        risk_off_stretch=float(params.risk_off_stretch),
        risk_off_floor=float(params.risk_off_floor),
        eps=float(params.eps),
    ).values.astype(float, copy=False)

    # Target vol can vary by regime per-date
    tv = np.array([_target_vol_for_date(params, regime, dt) for dt in times], dtype=float)
    base_lev = np.divide(tv, (vol + 1e-8))
    base_lev = np.clip(base_lev, float(params.lev_min), float(params.lev_cap))

    reg_mult = np.array([_regime_leverage_mult(regime, dt) for dt in times], dtype=float)
    lev = base_lev * throttle * reg_mult
    lev = np.clip(lev, float(params.lev_min), float(params.lev_cap))

    budget = lev * g
    return xr.DataArray(budget, coords={"time": unit_gross_weights.time}, dims=["time"])


# -------------------------------------------------------------------------
# Core portfolio constructor
# -------------------------------------------------------------------------

class PortfolioConstructor:
    """Constructs portfolios from composite scores."""

    def __init__(
        self,
        *,
        scores: "xr.DataArray",
        returns: "xr.DataArray",
        liquidity: Optional["xr.DataArray"] = None,
        asset_ids: Optional[Sequence[str]] = None,
        pin_idx: Optional[np.ndarray] = None,
        regime: Optional[RegimeLike] = None,
        tc_model: Optional[TransactionCostLike] = None,
        params: Optional[PortfolioParams] = None,
        market_returns: Optional["pd.Series"] = None,
    ):
        _require_xr()
        if scores.dims != ("time", "asset"):
            # allow swapped dims but enforce canonical ordering
            scores = scores.transpose("time", "asset")
        if returns.dims != ("time", "asset"):
            returns = returns.transpose("time", "asset")

        self.scores = scores
        self.returns = returns
        self.liquidity = liquidity.transpose("time", "asset") if (liquidity is not None and liquidity.dims != ("time","asset")) else liquidity
        self.asset_ids = list(asset_ids) if asset_ids is not None else list(scores.asset.values)
        self.pin_idx = np.asarray(pin_idx, dtype=int) if pin_idx is not None else np.array([], dtype=int)
        self.regime = regime
        self.tc = tc_model
        self.params = params or PortfolioParams()
        self.market_returns = market_returns

        # Sanity: match asset dimension length
        if scores.sizes.get("asset") != len(self.asset_ids):
            # fall back to coordinates
            self.asset_ids = list(scores.asset.values)

    # ---- Builders ----

    def build(self) -> PortfolioBuildResult:
        p = self.params
        if bool(p.use_equal_topk):
            return self.build_equal_topk(k=int(p.topn_base))
        if str(p.side_split_mode).lower() in {"long_short", "ls"}:
            # legacy hook; prefer build_long_short explicit
            return self.build_long_short()
        # decide by sign-ability: if you want long/short, call build_long_short explicitly
        return self.build_long_only()

    def build_long_only(self) -> PortfolioBuildResult:
        p = self.params
        S = _sma_time(self.scores.fillna(0.0), int(p.score_smooth_win))
        if self.liquidity is not None:
            S = (S * self.liquidity.fillna(0.0)).fillna(0.0)

        # shift to non-negative
        S = (S - S.min("asset") + 1e-6).fillna(0.0)

        times = S.time.values
        N = S.sizes["asset"]
        W = np.zeros((len(times), N), dtype=float)
        prev = np.zeros(N, dtype=float)

        for t, dt in enumerate(times):
            topn = _topn_for_date(p, self.regime, dt)
            s_t = S.isel(time=t).values.astype(float, copy=False)

            order = np.argsort(-s_t)
            sel = np.zeros(N, dtype=bool)
            if topn > 0:
                sel[order[:topn]] = True
            if self.pin_idx.size:
                sel[self.pin_idx] = True

            tgt = np.zeros(N, dtype=float)
            if np.any(sel):
                ss = s_t[sel]
                ss = ss - np.min(ss) + 1e-6
                tgt[sel] = ss / (float(np.sum(ss)) + p.eps)

            tilted = softmax_tilt(tgt, s_t, sel, float(p.softmax_tilt_alpha), eps=float(p.eps))

            lo = np.where(sel, float(p.min_pos), 0.0)
            hi = np.where(sel, float(p.max_pos), 0.0)
            unit = project_to_capped_simplex(tilted, T=1.0, lo=lo, hi=hi, eps=float(p.eps))

            expected_alpha = float(np.std(s_t[sel])) if np.any(sel) else 0.0

            if t > 0 and self.tc is not None and not self.tc.should_rebalance(prev, unit, expected_alpha):
                # keep previous, but project to today's feasible set (pins/topn may change)
                unit = project_to_capped_simplex(prev, T=1.0, lo=lo, hi=hi, eps=float(p.eps))
            else:
                sm = float(p.weight_smooth_alpha) * unit + (1.0 - float(p.weight_smooth_alpha)) * prev
                sm = softmax_tilt(sm, s_t, sel, float(p.softmax_tilt_alpha) * 0.5, eps=float(p.eps))
                unit = project_to_capped_simplex(sm, T=1.0, lo=lo, hi=hi, eps=float(p.eps))

            W[t] = unit
            prev = unit

        unit_da = xr.DataArray(W, coords={"time": S.time, "asset": self.asset_ids}, dims=["time", "asset"])

        budget = compute_budget_series(
            unit_gross_weights=unit_da,
            returns=self.returns,
            params=p,
            market_returns=self.market_returns,
            regime=self.regime,
        )
        scaled = (unit_da * budget.broadcast_like(unit_da)).fillna(0.0)

        # Final day-by-day cap enforcement (guardrail)
        final = []
        for t in range(scaled.sizes["time"]):
            b = float(budget.isel(time=t).values)
            row = np.nan_to_num(scaled.isel(time=t).values, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.zeros(N, dtype=float)
            hi = np.full(N, float(p.max_pos), dtype=float)
            final.append(project_to_capped_simplex(row, T=b, lo=lo, hi=hi, eps=float(p.eps)))
        final_da = xr.DataArray(np.vstack(final), coords=scaled.coords, dims=scaled.dims)

        return PortfolioBuildResult(
            final_weights=final_da,
            budget=budget,
            unit_weights=unit_da,
            meta={"mode": "long_only"},
        )

    def build_long_short(self) -> PortfolioBuildResult:
        p = self.params
        S = _sma_time(self.scores.fillna(0.0), int(p.score_smooth_win))
        if self.liquidity is not None:
            S = (S * self.liquidity.fillna(0.0)).fillna(0.0)

        times = S.time.values
        N = S.sizes["asset"]
        W = np.zeros((len(times), N), dtype=float)
        prev = np.zeros(N, dtype=float)

        L = int(p.long_seats)
        K = int(p.short_seats)

        for t, dt in enumerate(times):
            s_t = S.isel(time=t).values.astype(float, copy=False)
            order = np.argsort(-s_t)

            long_sel = np.zeros(N, dtype=bool)
            short_sel = np.zeros(N, dtype=bool)
            if L > 0:
                long_sel[order[:L]] = True
            if K > 0:
                short_sel[order[-K:]] = True

            if self.pin_idx.size:
                long_sel[self.pin_idx] = True

            # budgets per side
            mode = str(p.side_split_mode).lower()
            if mode == "fixed":
                b_long = float(np.clip(p.fixed_long_frac, 0.0, 1.0))
                b_short = 1.0 - b_long
            else:
                long_mass = float(np.sum(np.abs(s_t[long_sel]))) if np.any(long_sel) else 0.0
                short_mass = float(np.sum(np.abs(s_t[short_sel]))) if np.any(short_sel) else 0.0
                total = long_mass + short_mass + float(p.eps)
                b_long = long_mass / total if total > 0 else 0.5
                b_short = 1.0 - b_long

            # targets
            tgt_long = np.zeros(N, dtype=float)
            if np.any(long_sel) and b_long > 0:
                ss = s_t[long_sel]
                ss = ss - np.min(ss) + 1e-6
                tgt_long[long_sel] = (ss / (float(np.sum(ss)) + p.eps)) * b_long

            tgt_short = np.zeros(N, dtype=float)
            if np.any(short_sel) and b_short > 0:
                ss = -s_t[short_sel]  # higher = better short
                ss = ss - np.min(ss) + 1e-6
                tgt_short[short_sel] = -(ss / (float(np.sum(ss)) + p.eps)) * b_short

            tgt = tgt_long + tgt_short

            # softmax tilt per side (optional)
            if float(p.softmax_tilt_alpha) > 0:
                tgt2 = tgt.copy()
                if np.any(long_sel) and b_long > 0:
                    wL = softmax_tilt(np.abs(tgt_long), s_t, long_sel, float(p.softmax_tilt_alpha), eps=float(p.eps))
                    tgt2[long_sel] = wL[long_sel] * b_long
                if np.any(short_sel) and b_short > 0:
                    wS = softmax_tilt(np.abs(tgt_short), -s_t, short_sel, float(p.softmax_tilt_alpha), eps=float(p.eps))
                    tgt2[short_sel] = -wS[short_sel] * b_short
                tgt = tgt2

            # Project positive and negative legs separately
            lo_long = np.where(long_sel, float(p.min_pos), 0.0)
            hi_long = np.where(long_sel, float(p.max_pos), 0.0)

            lo_short = np.where(short_sel, float(p.min_pos), 0.0)  # abs floor
            hi_short = np.where(short_sel, float(p.max_pos), 0.0)  # abs cap

            pos_tgt = np.maximum(tgt, 0.0)
            pos_proj = project_to_capped_simplex(pos_tgt, T=b_long, lo=lo_long, hi=hi_long, eps=float(p.eps))

            neg_tgt = np.abs(np.minimum(tgt, 0.0))
            neg_proj = project_to_capped_simplex(neg_tgt, T=b_short, lo=lo_short, hi=hi_short, eps=float(p.eps))

            unit = pos_proj - neg_proj

            expected_alpha = float(np.std(s_t)) if N > 0 else 0.0

            if t > 0 and self.tc is not None and not self.tc.should_rebalance(prev, unit, expected_alpha):
                unit = prev.copy()
            else:
                sm = float(p.weight_smooth_alpha) * unit + (1.0 - float(p.weight_smooth_alpha)) * prev
                # re-project each side to keep feasibility
                pos_sm = np.maximum(sm, 0.0)
                neg_sm = np.abs(np.minimum(sm, 0.0))
                pos_proj = project_to_capped_simplex(pos_sm, T=b_long, lo=lo_long, hi=hi_long, eps=float(p.eps))
                neg_proj = project_to_capped_simplex(neg_sm, T=b_short, lo=lo_short, hi=hi_short, eps=float(p.eps))
                unit = pos_proj - neg_proj

            W[t] = unit
            prev = unit

        unit_da = xr.DataArray(W, coords={"time": S.time, "asset": self.asset_ids}, dims=["time", "asset"])

        # budget based on gross unit weights
        unit_gross = xr.DataArray(np.abs(W), coords=unit_da.coords, dims=unit_da.dims)
        budget = compute_budget_series(
            unit_gross_weights=unit_gross,
            returns=self.returns,
            params=p,
            market_returns=self.market_returns,
            regime=self.regime,
        )

        final_da = (unit_da * budget.broadcast_like(unit_da)).fillna(0.0)
        return PortfolioBuildResult(
            final_weights=final_da,
            budget=budget,
            unit_weights=unit_da,
            meta={"mode": "long_short", "b_long": float(p.fixed_long_frac)},
        )

    def build_equal_topk(self, k: int = 20) -> PortfolioBuildResult:
        p = self.params
        S = _sma_time(self.scores.fillna(0.0), int(p.score_smooth_win))
        if self.liquidity is not None:
            S = (S * self.liquidity.fillna(0.0)).fillna(0.0)
        S = (S - S.min("asset") + 1e-6).fillna(0.0)

        times = S.time.values
        N = S.sizes["asset"]
        W = np.zeros((len(times), N), dtype=float)

        for t, dt in enumerate(times):
            s_t = S.isel(time=t).values.astype(float, copy=False)
            order = np.argsort(-s_t)

            chosen = []
            # pins first
            if self.pin_idx.size:
                for idx in self.pin_idx.tolist():
                    if len(chosen) >= k:
                        break
                    if np.isfinite(s_t[idx]) and s_t[idx] > 0:
                        chosen.append(idx)

            if len(chosen) < k:
                for idx in order.tolist():
                    if len(chosen) >= k:
                        break
                    if idx in chosen:
                        continue
                    if s_t[idx] <= 0:
                        continue
                    chosen.append(idx)

            if len(chosen) == 0:
                continue

            # Equal weights sum to 1.0 unit (gross scaling done later)
            per = 1.0 / len(chosen)
            W[t, chosen] = per

        unit_da = xr.DataArray(W, coords={"time": S.time, "asset": self.asset_ids}, dims=["time", "asset"])
        budget = compute_budget_series(
            unit_gross_weights=unit_da,
            returns=self.returns,
            params=p,
            market_returns=self.market_returns,
            regime=self.regime,
        )
        final_da = (unit_da * budget.broadcast_like(unit_da)).fillna(0.0)

        return PortfolioBuildResult(
            final_weights=final_da,
            budget=budget,
            unit_weights=unit_da,
            meta={"mode": "equal_topk", "k": int(k)},
        )
