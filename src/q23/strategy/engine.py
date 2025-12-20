"""
q23.strategy.engine

StrategyEngine orchestrates:
1) data loading (MarketDataBundle)
2) factor computation (FactorLibrary)
3) IC weighting (DynamicICWeighting)
4) portfolio construction (PortfolioConstructor)
5) output writing (OutputWriter)

Ultimate enhancement additions:
- portfolio_diag_{tag}.csv      : performance/exposure/turnover diagnostics
- factor_exposure_{tag}.csv     : portfolio factor exposures over time (from F)
- factor_vectors_{tag}.csv      : per-asset factor snapshot + PM tables (from F + weights/returns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.config import cfg, PortfolioMode, TransactionCostConfig, TransactionCostScheme
from q23.strategy.factors import V4_24_FACTORS
from q23.strategy.data_loader import MarketDataBundle, load_market_data
from q23.strategy.factors import FactorLibrary, FactorParams
from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams
from q23.strategy.outputs import OutputWriter
from q23.strategy.transaction_costs import TransactionCostModel, compute_atr_pandas


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.engine")


def compute_forward_returns(
    returns: "xr.DataArray",
    horizon: int = 21,
    eps: float = 1e-12,
) -> "xr.DataArray":
    """Forward cumulative returns over [t+1, t+horizon] (no look-ahead)."""
    _require_xr()
    r = returns.fillna(0.0)
    # guard: returns must be > -1 for log1p; clip minimally
    r = r.clip(min=-0.999999)
    logret = xr.apply_ufunc(np.log1p, r)
    roll = logret.rolling(time=horizon, min_periods=horizon).sum()
    fwd_log = roll.shift(time=-horizon)
    fwd = xr.apply_ufunc(np.expm1, fwd_log).fillna(0.0)
    fwd.name = f"fwd_{horizon}"
    return fwd


def _turnover_from_weights(w: pd.DataFrame) -> pd.Series:
    d = w.fillna(0.0).diff().abs().sum(axis=1)
    if len(d) > 0:
        d.iloc[0] = 0.0
    d.name = "turnover"
    return d


def _compute_portfolio_diag(
    *,
    final_weights: "xr.DataArray",   # time x asset
    returns: "xr.DataArray",         # time x asset
    tc_bps: float = 10.0,
    tc_config: Optional[TransactionCostConfig] = None,
    ohlc: Optional[Dict[str, "xr.DataArray"]] = None,
) -> pd.DataFrame:
    """Compute portfolio diagnostics including transaction costs.
    
    Args:
        final_weights: Portfolio weights (time x asset)
        returns: Asset returns (time x asset)
        tc_bps: Legacy flat basis points (used if tc_config is None)
        tc_config: Transaction cost configuration (overrides tc_bps)
        ohlc: Dict with 'open', 'high', 'low', 'close' DataArrays for ATR-based TC
        
    Returns:
        DataFrame with portfolio diagnostics including TC-adjusted returns
    """
    _require_xr()
    w_df = final_weights.transpose("time", "asset").to_pandas()
    r_df = returns.transpose("time", "asset").to_pandas()

    if not isinstance(w_df, pd.DataFrame):
        w_df = pd.DataFrame(w_df)
    if not isinstance(r_df, pd.DataFrame):
        r_df = pd.DataFrame(r_df)

    # align
    idx = w_df.index.intersection(r_df.index)
    cols = [c for c in w_df.columns if c in r_df.columns]
    w_df = w_df.reindex(idx)[cols].fillna(0.0)
    r_df = r_df.reindex(idx)[cols].fillna(0.0)

    w_lag = w_df.shift(1).fillna(0.0)
    contrib = w_lag * r_df
    port_ret = contrib.sum(axis=1)

    mkt_ret = r_df.mean(axis=1) if len(cols) else pd.Series(index=idx, data=0.0)
    active_ret = port_ret - mkt_ret

    gross = w_df.abs().sum(axis=1)
    net = w_df.sum(axis=1)
    npos = (w_df.abs() > 1e-12).sum(axis=1)
    herf = (w_df ** 2).sum(axis=1)

    turnover = _turnover_from_weights(w_df)
    weights_delta = w_df.diff().fillna(0.0)
    
    # Compute transaction costs using the new model
    if tc_config is None:
        # Legacy flat BPS mode
        tc_rate = float(tc_bps) / 10000.0
        tc_cost = turnover * tc_rate
        tc_cost_atr = tc_cost.copy()  # Same as flat for backward compat
    else:
        tc_model = TransactionCostModel(tc_config)
        
        # Prepare OHLC DataFrames if available (for ATR-based TC)
        close_df = None
        high_df = None
        low_df = None
        atr_df = None
        
        if ohlc is not None:
            try:
                close_df = ohlc["close"].transpose("time", "asset").to_pandas()
                high_df = ohlc["high"].transpose("time", "asset").to_pandas()
                low_df = ohlc["low"].transpose("time", "asset").to_pandas()
                
                # Align OHLC with weights
                close_df = close_df.reindex(index=idx, columns=cols).fillna(method="ffill").fillna(1.0)
                high_df = high_df.reindex(index=idx, columns=cols).fillna(method="ffill").fillna(1.0)
                low_df = low_df.reindex(index=idx, columns=cols).fillna(method="ffill").fillna(1.0)
                
                # Pre-compute ATR
                atr_df = compute_atr_pandas(high_df, low_df, close_df, tc_config.atr_window)
            except Exception:
                # Fall back to flat BPS if OHLC fails
                pass
        
        # Compute TC based on scheme
        if tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR and close_df is not None:
            tc_costs_per_asset = tc_model.compute_costs(
                weights_delta, close=close_df, atr=atr_df, high=high_df, low=low_df
            )
            tc_cost_atr = tc_costs_per_asset.sum(axis=1).reindex(idx).fillna(0.0)
        else:
            # Non-ATR schemes or fallback
            tc_costs_per_asset = tc_model.compute_costs(weights_delta)
            tc_cost_atr = tc_costs_per_asset.sum(axis=1).reindex(idx).fillna(0.0)
        
        # Also compute flat BPS for comparison
        tc_rate = float(tc_config.flat_bps) / 10000.0
        tc_cost = turnover * tc_rate
    
    port_ret_net_tc = port_ret - tc_cost
    port_ret_net_atr_tc = port_ret - tc_cost_atr
    active_ret_net_tc = active_ret - tc_cost
    active_ret_net_atr_tc = active_ret - tc_cost_atr

    # EWMA vol (annualized)
    lam = 0.95
    var = 0.0
    vol = np.zeros(len(port_ret), dtype=float)
    pr = port_ret.fillna(0.0).values
    for i, rv in enumerate(pr):
        var = lam * var + (1.0 - lam) * (rv * rv)
        vol[i] = np.sqrt(max(var, 0.0)) * np.sqrt(252.0)

    diag = pd.DataFrame(
        {
            "port_ret": port_ret,
            "mkt_ret": mkt_ret.reindex(idx).fillna(0.0),
            "active_ret": active_ret,
            "gross_exposure": gross,
            "net_exposure": net,
            "n_positions": npos,
            "herfindahl": herf,
            "turnover": turnover.reindex(idx).fillna(0.0),
            "tc_cost_flat": tc_cost.reindex(idx).fillna(0.0),
            "tc_cost_atr": tc_cost_atr.reindex(idx).fillna(0.0),
            "port_ret_net_tc": port_ret_net_tc,
            "port_ret_net_atr_tc": port_ret_net_atr_tc,
            "active_ret_net_tc": active_ret_net_tc,
            "active_ret_net_atr_tc": active_ret_net_atr_tc,
            "rolling_vol": vol,
        },
        index=idx,
    )
    diag.index.name = "time"
    return diag


def _compute_factor_exposure_ts(
    *,
    F: "xr.DataArray",            # factor x time x asset
    final_weights: "xr.DataArray" # time x asset
) -> pd.DataFrame:
    _require_xr()
    # exposure[t,f] = sum_a w[t,a] * F[f,t,a]
    # align coords
    F2, w2 = xr.align(F, final_weights, join="inner")
    exp = (F2 * w2).sum("asset").transpose("time", "factor").fillna(0.0)
    df = exp.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = "time"
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _compute_factor_vectors_snapshot(
    *,
    F: "xr.DataArray",                 # factor x time x asset
    composite_score: "xr.DataArray",   # time x asset
    final_weights: "xr.DataArray",     # time x asset
    returns: "xr.DataArray",           # time x asset
) -> pd.DataFrame:
    """
    Per-asset table used by the PM dashboard:
    - factor exposures (snapshot at last date) - these are z-scores, will be rounded to 4 decimals
    - summary cols: mean_score, score_vol, days_held, avg_weight, avg_weight_when_held, pnl_per_day_held, total_pnl_contrib
    """
    _require_xr()
    # Align all
    F2, score2, w2, r2 = xr.align(F, composite_score, final_weights, returns, join="inner")

    # Snapshot exposures at last time
    t_last = F2.time.values[-1]
    expo = F2.sel(time=t_last).transpose("asset", "factor").to_pandas()
    if not isinstance(expo, pd.DataFrame):
        expo = pd.DataFrame(expo)

    # Score stats
    score_df = score2.transpose("time", "asset").to_pandas()
    w_df = w2.transpose("time", "asset").to_pandas()
    r_df = r2.transpose("time", "asset").to_pandas()
    for obj in (score_df, w_df, r_df):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Expected DataFrame conversion failure in factor_vectors snapshot.")

    idx = w_df.index.intersection(r_df.index).intersection(score_df.index)
    cols = [c for c in w_df.columns if c in r_df.columns and c in score_df.columns]
    w_df = w_df.reindex(idx)[cols].fillna(0.0)
    r_df = r_df.reindex(idx)[cols].fillna(0.0)
    score_df = score_df.reindex(idx)[cols].fillna(0.0)

    held = (w_df.abs() > 1e-12).astype(float)
    days_held = held.sum(axis=0).astype(int)  # Integer type for days

    avg_weight = w_df.mean(axis=0)
    avg_weight_when_held = (w_df.abs().where(held > 0).sum(axis=0) / (days_held.replace(0, np.nan))).fillna(0.0)

    mean_score = score_df.mean(axis=0)
    score_vol = score_df.std(axis=0, ddof=0)

    w_lag = w_df.shift(1).fillna(0.0)
    contrib = (w_lag * r_df).fillna(0.0)
    total_pnl_contrib = contrib.sum(axis=0)
    pnl_per_day_held = (total_pnl_contrib / (days_held.replace(0, np.nan))).fillna(0.0)

    # Build table
    out = expo.copy()
    out.index.name = "symbol"
    out["total_pnl_contrib"] = total_pnl_contrib.reindex(out.index).fillna(0.0)
    out["pnl_per_day_held"] = pnl_per_day_held.reindex(out.index).fillna(0.0)
    out["mean_score"] = mean_score.reindex(out.index).fillna(0.0)
    out["score_vol"] = score_vol.reindex(out.index).fillna(0.0)
    out["days_held"] = days_held.reindex(out.index).fillna(0).astype(int)  # Ensure integer
    out["avg_weight"] = avg_weight.reindex(out.index).fillna(0.0)
    out["avg_weight_when_held"] = avg_weight_when_held.reindex(out.index).fillna(0.0)

    # Replace inf/NaN (rounding will happen in write_factor_vectors)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return out


@dataclass
class EngineArtifacts:
    F: "xr.DataArray"
    composite_score: "xr.DataArray"
    ic_raw: "xr.DataArray"
    ic_smooth: "xr.DataArray"
    factor_weights: "xr.DataArray"
    unit_weights: "xr.DataArray"
    final_weights: "xr.DataArray"
    budget: "xr.DataArray"
    meta: Dict[str, Any]


class StrategyEngine:
    def __init__(
        self,
        *,
        output_writer: Optional[OutputWriter] = None,
        factor_params: Optional[FactorParams] = None,
        ic_params: Optional[ICWeightingParams] = None,
        portfolio_params: Optional[PortfolioParams] = None,
        factors: Optional[Sequence[str]] = None,
    ):
        self.output_writer = output_writer or OutputWriter(
            base_name=cfg.paths.BASE_NAME,
            output_dir=cfg.paths.OUTPUT_ROOT,
        )
        self.factor_params = factor_params or FactorParams(eps=float(cfg.strategy.EPS))
        self.ic_params = ic_params or ICWeightingParams(lam=0.95, eps=float(cfg.strategy.EPS))
        self.portfolio_params = portfolio_params or self._portfolio_params_from_cfg()
        self.factors = factors if factors is not None else list(V4_24_FACTORS)

    def _portfolio_params_from_cfg(self) -> PortfolioParams:
        s = cfg.strategy
        return PortfolioParams(
            max_pos=float(s.MAX_POS),
            min_pos=float(s.MIN_POS),
            eps=float(s.EPS),
            topn_base=int(s.TOPN_BASE),
            topn_volatile=int(s.TOPN_VOLATILE),
            score_smooth_win=int(s.SCORE_SMOOTH_WIN),
            weight_smooth_alpha=float(s.WEIGHT_SMOOTH_ALPHA),
            softmax_tilt_alpha=float(s.SOFTMAX_TILT_ALPHA),
            use_equal_topk=bool(s.USE_EQUAL_TOPK),
            long_seats=int(s.LONG_SEATS),
            short_seats=int(s.SHORT_SEATS),
            fixed_long_frac=float(s.FIXED_LONG_FRAC),
            side_split_mode=str(s.SIDE_SPLIT_MODE.value),
            budget_mode=str(s.PM_TARGET_MODE.value),
            pm_gross_target=float(s.PM_GROSS_TARGET),
            target_vol_base=float(s.TARGET_VOL_BASE),
            target_vol_volatile=float(s.TARGET_VOL_VOLATILE),
            lev_cap=float(s.LEV_CAP),
            lev_min=float(s.LEV_MIN),
        )

    def run(
        self,
        *,
        bundle: Optional[MarketDataBundle] = None,
        pinned: Optional[Sequence[str]] = None,
        exchanges: Optional[Sequence[str]] = None,
        tag: Optional[str] = None,
    ) -> EngineArtifacts:
        """Run the full pipeline and write outputs."""
        _require_xr()

        # 1) Load data
        if bundle is None:
            bundle = load_market_data(
                min_date=cfg.strategy.MIN_DATE,
                exchanges=exchanges,
                pinned=pinned,
            )

        ds = bundle.data
        returns = bundle.returns

        # 2) Factors
        flib = FactorLibrary(bundle, params=self.factor_params, factors=self.factors)
        F, _factor_art = flib.compute()

        # 3) Forward returns + IC weights
        fwd = compute_forward_returns(returns, horizon=21, eps=float(cfg.strategy.EPS))
        icw = DynamicICWeighting(F, fwd, params=self.ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

        # 4) Portfolio
        pc = PortfolioConstructor(
            scores=score,
            returns=returns,
            liquidity=ds.get("is_liquid", None),
            asset_ids=bundle.asset_ids,
            pin_idx=bundle.pin_idx,
            params=self.portfolio_params,
        )
        if cfg.strategy.PORTFOLIO_MODE == PortfolioMode.LONG_SHORT:
            res = pc.build_long_short()
        else:
            res = pc.build_long_only()

        # 5) Outputs
        tag = tag or str(ds.time.values[-1])[:10]

        # Core v4 outputs
        self.output_writer.write_wide_weights(res.final_weights, tag=tag)
        self.output_writer.write_budget(res.budget, tag=tag)
        self.output_writer.write_factor_weights(fw, tag=tag)
        self.output_writer.write_ic(ic_raw, ic_smooth, tag=tag)

        # Enhanced outputs (restore PM console richness)
        # Use new TC config (Quantiacs ATR by default)
        tc_config = cfg.tc
        tc_bps = tc_config.effective_bps  # For backward compatibility
        
        # Prepare OHLC data for ATR-based TC
        ohlc_data = {
            "open": ds["open"],
            "high": ds["high"],
            "low": ds["low"],
            "close": ds["close"],
        }
        
        diag = _compute_portfolio_diag(
            final_weights=res.final_weights,
            returns=returns,
            tc_bps=tc_bps,
            tc_config=tc_config,
            ohlc=ohlc_data,
        )
        self.output_writer.write_portfolio_diag(diag, tag=tag)

        exp_ts = _compute_factor_exposure_ts(F=F, final_weights=res.final_weights)
        self.output_writer.write_factor_exposure(exp_ts, tag=tag)

        fv = _compute_factor_vectors_snapshot(
            F=F,
            composite_score=score,
            final_weights=res.final_weights,
            returns=returns,
        )
        self.output_writer.write_factor_vectors(fv, tag=tag)

        meta = {
            **bundle.meta,
            "tag": tag,
            "base_name": cfg.paths.BASE_NAME,
            "output_dir": cfg.paths.OUTPUT_ROOT,
            "factors": list(F.factor.values),
            "tc_bps": tc_bps,
            "tc_config": tc_config.to_dict(),
            "artifacts": {
                "wide_weights": str(self.output_writer.wide_weights_path(tag)),
                "budget": str(self.output_writer.budget_path(tag)),
                "factor_weights": str(self.output_writer.factor_weights_path(tag)),
                "ic": str(self.output_writer.ic_path(tag)),
                "portfolio_diag": str(self.output_writer.portfolio_diag_path(tag)),
                "factor_exposure": str(self.output_writer.factor_exposure_path(tag)),
                "factor_vectors": str(self.output_writer.factor_vectors_path(tag)),
            },
        }
        self.output_writer.write_meta(meta, tag=tag)

        return EngineArtifacts(
            F=F,
            composite_score=score,
            ic_raw=ic_raw,
            ic_smooth=ic_smooth,
            factor_weights=fw,
            unit_weights=res.unit_weights,
            final_weights=res.final_weights,
            budget=res.budget,
            meta=meta,
        )
