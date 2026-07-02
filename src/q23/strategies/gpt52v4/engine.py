from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.shared.math_utils import ewma_1d, safe_corrcoef
from q23.shared import config as global_config
from q23.strategies.base import StrategyBase, StrategyConfig, StrategyArtifacts
from q23.strategies.registry import StrategyRegistry
from q23.strategies.gpt52v4.config import (
    gpt52v4_config,
    GPT52V4_FACTORS,
    MOM_SLEEVE,
    MR_SLEEVE,
    MS_SLEEVE,
    DEF_SLEEVE,
)
from q23.strategy.outputs import OutputWriter
from q23.strategy.factors import FactorLibrary, FactorParams
from q23.strategy.ic_weighting import CorrICWeightingParams, CorrelationAwareICWeighting
from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams
from q23.strategy.risk_overlays import RiskOverlayParams, combined_overlay, apply_overlay


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray required")


def _ewma_time(x: "xr.DataArray", lam: float) -> "xr.DataArray":
    _require_xr()
    return xr.apply_ufunc(
        ewma_1d,
        x,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="forbidden",
        kwargs={"lam": float(lam)},
        output_dtypes=[float],
    ).fillna(0.0)


def _normalize_positive(x: "xr.DataArray", dim: str, eps: float, min_floor: float = 0.0) -> "xr.DataArray":
    _require_xr()
    raw = x.clip(min=0.0)
    if min_floor > 0:
        raw = raw.clip(min=float(min_floor))
    s = raw.sum(dim) + eps
    out = raw / s
    n = raw.sizes.get(dim, 1)
    out = out.where(s > eps, other=(0.0 * raw + 1.0 / float(n)))
    return out.fillna(0.0)


def _min_weight_floor(fw: "xr.DataArray", min_w: float, eps: float) -> "xr.DataArray":
    fw_sum = fw.sum("factor")
    floored = fw.clip(min=float(min_w))
    floored_sum = floored.sum("factor") + eps
    out = xr.where(fw_sum > eps, floored / floored_sum, fw)
    return out.fillna(0.0)


def _rank_ic(score: "xr.DataArray", fwd: "xr.DataArray", *, asset_dim: str = "asset") -> "xr.DataArray":
    """Spearman-like IC: corr(rank(score), rank(fwd)) across assets per time."""
    _require_xr()
    s, r = xr.align(score, fwd, join="inner")
    sr = s.rank(asset_dim, pct=False).fillna(0.0)
    rr = r.rank(asset_dim, pct=False).fillna(0.0)
    return xr.corr(sr, rr, dim=asset_dim).fillna(0.0)


def _market_drawdown(returns: "xr.DataArray", dd_win: int, eps: float) -> "xr.DataArray":
    """Market-level rolling drawdown series (time,) from mean return."""
    _require_xr()
    mkt = returns.fillna(0.0).mean("asset")
    growth = (1.0 + mkt).cumprod()
    peak = growth.rolling(time=dd_win, min_periods=1).max()
    dd = (peak - growth) / (peak + eps)
    return dd.fillna(0.0).clip(0.0, 1.0)


def _sleeve_score(
    F: "xr.DataArray",
    returns: "xr.DataArray",
    *,
    factors: Sequence[str],
    horizons: Sequence[int],
    lam: float,
    corr_penalty: float,
    min_div: float,
    min_factor_weight: float,
    eps: float,
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Compute sleeve score via correlation-aware IC weights, averaged across horizons."""
    _require_xr()
    from q23.strategy.engine import compute_forward_returns

    factors = [f for f in factors if f in set(F.factor.values)]
    F_s = F.sel(factor=factors)

    scores: List["xr.DataArray"] = []
    fws: List["xr.DataArray"] = []

    for h in horizons:
        fwd = compute_forward_returns(returns, horizon=int(h), eps=eps)
        ic_params = CorrICWeightingParams(
            lam=float(lam),
            eps=float(eps),
            clip_ic=0.25,
            use_positive_only=True,
            corr_penalty_strength=float(corr_penalty),
            min_diversification=float(min_div),
        )
        icw = CorrelationAwareICWeighting(F_s, fwd, params=ic_params)
        _, _, fw_h = icw.artifacts()
        fw_h = _min_weight_floor(fw_h, float(min_factor_weight), eps)

        w_aligned, F_aligned = xr.align(fw_h, F_s, join="inner")
        score_h = (F_aligned * w_aligned).sum("factor").fillna(0.0)

        scores.append(score_h)
        fws.append(fw_h)

    score = xr.concat(scores, dim="h").mean("h").fillna(0.0)
    fw = xr.concat(fws, dim="h").mean("h").fillna(0.0)
    return score, fw


def _sleeve_corr_penalty(S: "xr.DataArray", strength: float, min_floor: float, eps: float) -> "xr.DataArray":
    """Penalize sleeves that are highly correlated with each other (per time)."""
    _require_xr()

    def _pen(arr: np.ndarray) -> np.ndarray:
        # arr shape (sleeve, asset)
        if arr.shape[1] < 2 or arr.shape[0] < 2:
            return np.ones(arr.shape[0], dtype=float)
        corr = safe_corrcoef(arr.T, eps=eps)  # (asset, sleeve) -> corr sleeves
        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            return np.ones(arr.shape[0], dtype=float)
        np.fill_diagonal(corr, 0.0)
        pen = 1.0 - np.nanmean(np.abs(corr), axis=1)
        return np.nan_to_num(pen, nan=1.0, posinf=1.0, neginf=0.0)

    pen = xr.apply_ufunc(
        _pen,
        S.transpose("time", "sleeve", "asset"),
        input_core_dims=[["sleeve", "asset"]],
        output_core_dims=[["sleeve"]],
        vectorize=True,
        dask="forbidden",
        output_dtypes=[float],
    ).transpose("sleeve", "time")

    pen = pen.clip(min=float(min_floor), max=1.0).fillna(1.0)
    return (1.0 - float(strength) * (1.0 - pen)).clip(min=float(min_floor), max=1.0).fillna(1.0)


@StrategyRegistry.register
class GPT52V4Strategy(StrategyBase):
    """GPT5.2 v4: sleeve ensemble + stress-adaptive sleeve priors."""

    @classmethod
    def strategy_id(cls) -> str:
        return "gpt52v4"

    @property
    def config(self) -> StrategyConfig:
        cfg = gpt52v4_config
        return StrategyConfig(
            name="gpt52v4",
            display_name="GPT52 v4 (Adaptive Sleeve Ensemble)",
            version="4.0",
            min_date=cfg.MIN_DATE,
            exchanges=list(cfg.EXCHANGES),
            factors=GPT52V4_FACTORS,
            topn=cfg.TOPN_BASE,
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            target_vol=cfg.TARGET_VOL_BASE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            tc_bps=cfg.TC_BPS,
            description=(
                "Adaptive sleeve ensemble. Builds four independent sleeves (momentum, mean-reversion, "
                "microstructure, defensive) and dynamically weights sleeves via IC + correlation penalty, "
                "with an additional stress prior that shifts weight toward defensive/MR during drawdowns."
            ),
            long_only=cfg.LONG_ONLY,
            long_seats=cfg.LONG_SEATS,
            short_seats=cfg.SHORT_SEATS,
            weight_smooth_alpha=cfg.WEIGHT_SMOOTH_ALPHA,
            score_smooth_win=cfg.SCORE_SMOOTH_WIN,
            softmax_tilt_alpha=cfg.SOFTMAX_TILT_ALPHA,
            risk_off_stretch=cfg.RISK_OFF_STRETCH,
            risk_off_floor=cfg.RISK_OFF_FLOOR,
            dd_win=cfg.DD_WIN,
        )

    def get_factors(self) -> Dict[str, "xr.DataArray"]:
        return self._factors if hasattr(self, "_factors") else {}

    def run(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        tag: Optional[str] = None,
        write_outputs: bool = True,
        force_live_data: bool = False,
    ) -> StrategyArtifacts:
        _require_xr()

        from q23.strategy.engine import (
            _compute_portfolio_diag,
            _compute_factor_exposure_ts,
            _compute_factor_vectors_snapshot,
            compute_forward_returns,
        )
        from q23.strategy.data_loader import load_market_data

        cfg = gpt52v4_config
        use_min_date = min_date or cfg.MIN_DATE

        bundle = load_market_data(
            min_date=use_min_date,
            max_date=max_date,
            exchanges=list(cfg.EXCHANGES),
            pinned=None,
            fill_recent_days=global_config.cfg.marketstack.DEFAULT_LOOKBACK_DAYS,
            use_marketstack=global_config.cfg.marketstack.ENABLED,
            force_live_data=force_live_data,
            strategy_id=self.strategy_id(),
        )

        ds = bundle.data
        returns = bundle.returns

        liq = ds.get("is_liquid", None)
        if liq is None:
            vol = ds.get("vol", None)
            if vol is not None:
                liq = xr.where(np.isfinite(vol) & (vol > 0), 1.0, 0.0)
            else:
                liq = xr.ones_like(returns, dtype=float)
        else:
            liq = liq.fillna(1.0)

        factor_params = FactorParams(
            eps=cfg.EPS,
            BETA_WIN=cfg.BETA_WIN,
            IDIO_WIN=cfg.IDIO_WIN,
            DOWN_WIN=cfg.DOWN_WIN,
            CORR_WIN=cfg.CORR_WIN,
            ADV_WIN_LONG=cfg.ADV_WIN_LONG,
            ADV_WIN_SHORT=cfg.ADV_WIN_SHORT,
            MOM_WIN=cfg.MOM_WIN,
            MOM_SHORT=cfg.MOM_SHORT,
            MOM_LONG=cfg.MOM_LONG,
            REV_WIN=cfg.REV_WIN,
            REV_LONG=cfg.REV_LONG,
            DON_WIN=cfg.DON_WIN,
            EMA_FAST=cfg.EMA_FAST,
            EMA_SLOW=cfg.EMA_SLOW,
            ATR_WIN=cfg.ATR_WIN,
            SECTOR_MOM_WIN=cfg.SECTOR_MOM_WIN,
            VALUE_WIN=cfg.VALUE_WIN,
            # Neural
            EFFICIENCY_WIN=cfg.EFFICIENCY_WIN,
            ATTENTION_WIN=cfg.ATTENTION_WIN,
            DISPOSITION_WIN=cfg.DISPOSITION_WIN,
            SKEW_WIN=cfg.SKEW_WIN,
            KURT_WIN=cfg.KURT_WIN,
            VOV_WIN=cfg.VOV_WIN,
            PERSISTENCE_WIN=cfg.PERSISTENCE_WIN,
            # OU
            OU_SHORT_WIN=cfg.OU_SHORT_WIN,
            OU_MED_WIN=cfg.OU_MED_WIN,
            OU_HALFLIFE_MIN=cfg.OU_HALFLIFE_MIN,
            OU_HALFLIFE_MAX=cfg.OU_HALFLIFE_MAX,
            OU_ZSCORE_CLIP=cfg.OU_ZSCORE_CLIP,
            # GLFT
            OFI_SHORT_WIN=cfg.OFI_SHORT_WIN,
            OFI_MED_WIN=cfg.OFI_MED_WIN,
            OFI_LONG_WIN=cfg.OFI_LONG_WIN,
            VPIN_WIN=cfg.VPIN_WIN,
            SPREAD_EST_WIN=cfg.SPREAD_EST_WIN,
            IMPACT_WIN=cfg.IMPACT_WIN,
            INVENTORY_DECAY=cfg.INVENTORY_DECAY,
            # MTF
            WEEK=5,
            MONTH=21,
            QUARTER=63,
            YEAR=252,
        )

        flib = FactorLibrary(bundle, params=factor_params, factors=GPT52V4_FACTORS)
        F, _ = flib.compute()
        self._factors = {nm: F.sel(factor=nm) for nm in GPT52V4_FACTORS if nm in set(F.factor.values)}

        # Sleeve scores
        mom_score, mom_fw = _sleeve_score(
            F,
            returns,
            factors=MOM_SLEEVE,
            horizons=cfg.MOM_HORIZONS,
            lam=cfg.IC_LAMBDA,
            corr_penalty=cfg.FACTOR_CORR_PENALTY,
            min_div=cfg.FACTOR_MIN_DIVERSIFICATION,
            min_factor_weight=cfg.FACTOR_MIN_WEIGHT,
            eps=cfg.EPS,
        )
        mr_score, mr_fw = _sleeve_score(
            F,
            returns,
            factors=MR_SLEEVE,
            horizons=cfg.MR_HORIZONS,
            lam=cfg.IC_LAMBDA,
            corr_penalty=cfg.FACTOR_CORR_PENALTY,
            min_div=cfg.FACTOR_MIN_DIVERSIFICATION,
            min_factor_weight=cfg.FACTOR_MIN_WEIGHT,
            eps=cfg.EPS,
        )
        ms_score, ms_fw = _sleeve_score(
            F,
            returns,
            factors=MS_SLEEVE,
            horizons=cfg.MS_HORIZONS,
            lam=cfg.IC_LAMBDA,
            corr_penalty=cfg.FACTOR_CORR_PENALTY,
            min_div=cfg.FACTOR_MIN_DIVERSIFICATION,
            min_factor_weight=cfg.FACTOR_MIN_WEIGHT,
            eps=cfg.EPS,
        )
        def_score, def_fw = _sleeve_score(
            F,
            returns,
            factors=DEF_SLEEVE,
            horizons=cfg.DEF_HORIZONS,
            lam=cfg.IC_LAMBDA,
            corr_penalty=cfg.FACTOR_CORR_PENALTY,
            min_div=cfg.FACTOR_MIN_DIVERSIFICATION,
            min_factor_weight=cfg.FACTOR_MIN_WEIGHT,
            eps=cfg.EPS,
        )

        # Dynamic sleeve weights based on sleeve IC (21d forward)
        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)
        sleeve_names = ["mom", "mr", "ms", "def"]
        S = xr.concat([mom_score, mr_score, ms_score, def_score], dim="sleeve").assign_coords(sleeve=sleeve_names)

        ic = xr.concat(
            [
                _rank_ic(mom_score, fwd21),
                _rank_ic(mr_score, fwd21),
                _rank_ic(ms_score, fwd21),
                _rank_ic(def_score, fwd21),
            ],
            dim="sleeve",
        ).assign_coords(sleeve=sleeve_names)
        ic = ic.shift(time=1).fillna(0.0)
        ic_smooth = _ewma_time(ic, lam=cfg.SLEEVE_IC_LAMBDA).fillna(0.0)

        # Sleeve correlation penalty
        pen = _sleeve_corr_penalty(S, strength=cfg.SLEEVE_CORR_PENALTY, min_floor=0.25, eps=cfg.EPS)

        # Stress prior: shift weight away from momentum in drawdowns
        dd = _market_drawdown(returns, dd_win=cfg.DD_WIN, eps=cfg.EPS)
        stress = (dd / (cfg.STRESS_DD_CAP + cfg.EPS)).clip(0.0, 1.0)
        stress = stress.broadcast_like(ic_smooth)

        stress_mult = xr.concat(
            [
                1.0 - cfg.STRESS_MOM_MULT * stress.sel(sleeve="mom"),
                1.0 + cfg.STRESS_MR_MULT * stress.sel(sleeve="mr"),
                1.0 + cfg.STRESS_MS_MULT * stress.sel(sleeve="ms"),
                1.0 + cfg.STRESS_DEF_MULT * stress.sel(sleeve="def"),
            ],
            dim="sleeve",
        ).assign_coords(sleeve=sleeve_names)

        raw_w = (ic_smooth.clip(min=0.0) * pen * stress_mult).fillna(0.0)
        w = _normalize_positive(raw_w, dim="sleeve", eps=cfg.EPS, min_floor=cfg.SLEEVE_MIN_WEIGHT)

        # Combine sleeve scores
        score = (S * w.broadcast_like(S)).sum("sleeve").fillna(0.0)

        # Combine factor weights for output
        all_factors = list(F.factor.values)

        def _expand_fw(fw_s: "xr.DataArray") -> "xr.DataArray":
            return fw_s.fillna(0.0).reindex(factor=all_factors, fill_value=0.0)

        fw_m = _expand_fw(mom_fw)
        fw_r = _expand_fw(mr_fw)
        fw_s = _expand_fw(ms_fw)
        fw_d = _expand_fw(def_fw)

        w_m = w.sel(sleeve="mom").broadcast_like(fw_m.sum("factor"))
        w_r = w.sel(sleeve="mr").broadcast_like(fw_r.sum("factor"))
        w_s2 = w.sel(sleeve="ms").broadcast_like(fw_s.sum("factor"))
        w_d2 = w.sel(sleeve="def").broadcast_like(fw_d.sum("factor"))

        fw_total = (fw_m * w_m + fw_r * w_r + fw_s * w_s2 + fw_d * w_d2).fillna(0.0)
        fw_total = fw_total / (fw_total.sum("factor") + cfg.EPS)

        # Portfolio construction
        portfolio_params = PortfolioParams(
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            eps=cfg.EPS,
            topn_base=cfg.TOPN_BASE,
            topn_volatile=cfg.TOPN_VOLATILE,
            score_smooth_win=cfg.SCORE_SMOOTH_WIN,
            weight_smooth_alpha=cfg.WEIGHT_SMOOTH_ALPHA,
            softmax_tilt_alpha=cfg.SOFTMAX_TILT_ALPHA,
            use_equal_topk=False,
            long_seats=cfg.LONG_SEATS,
            short_seats=cfg.SHORT_SEATS,
            fixed_long_frac=0.75,
            side_split_mode="prop_mass",
            budget_mode="vol_target",
            pm_gross_target=1.0,
            target_vol_base=cfg.TARGET_VOL_BASE,
            target_vol_volatile=cfg.TARGET_VOL_VOLATILE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            dd_win=cfg.DD_WIN,
            risk_off_stretch=cfg.RISK_OFF_STRETCH,
            risk_off_floor=cfg.RISK_OFF_FLOOR,
        )

        pc = PortfolioConstructor(
            scores=score,
            returns=returns,
            liquidity=liq,
            asset_ids=bundle.asset_ids,
            pin_idx=bundle.pin_idx,
            params=portfolio_params,
        )
        res = pc.build_long_short()
        final_weights = res.final_weights
        budget = res.budget

        # Overlay
        overlay_params = RiskOverlayParams(
            vol_window=cfg.VOL_OVERLAY_WIN,
            target_vol=cfg.TARGET_VOL_BASE,
            max_vol=cfg.VOL_OVERLAY_MAX,
            dd_window=cfg.DD_WIN,
            dd_stretch=cfg.RISK_OFF_STRETCH,
            dd_floor=cfg.RISK_OFF_FLOOR,
            clip=(0.5, 1.25),
            eps=cfg.EPS,
        )
        overlay = combined_overlay(returns, params=overlay_params)
        final_weights, budget = apply_overlay(final_weights, budget, overlay)

        use_tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        meta = {}
        if write_outputs:
            output_dir = self.get_output_dir()
            writer = OutputWriter(base_name=self.get_base_name(), output_dir=output_dir)

            writer.write_wide_weights(final_weights, tag=use_tag)
            writer.write_budget(budget, tag=use_tag)
            writer.write_factor_weights(fw_total, tag=use_tag)

            diag = _compute_portfolio_diag(final_weights=final_weights, returns=returns, tc_bps=cfg.TC_BPS)
            writer.write_portfolio_diag(diag, tag=use_tag)

            exp_ts = _compute_factor_exposure_ts(F=F, final_weights=final_weights)
            writer.write_factor_exposure(exp_ts, tag=use_tag)

            fv = _compute_factor_vectors_snapshot(F=F, composite_score=score, final_weights=final_weights, returns=returns)
            writer.write_factor_vectors(fv, tag=use_tag)

            # last sleeve weights snapshot for debugging
            last_w = {}
            if w.sizes.get("time", 0) > 0:
                for sid in sleeve_names:
                    try:
                        last_w[sid] = float(w.sel(sleeve=sid).isel(time=-1).values)
                    except Exception:
                        pass

            meta = {
                "strategy_id": self.strategy_id(),
                "strategy_version": self.config.version,
                "display_name": self.config.display_name,
                "tag": use_tag,
                "run_timestamp": datetime.now().isoformat(),
                "date_range": [str(final_weights.time.values[0])[:10], str(final_weights.time.values[-1])[:10]],
                "sleeves": sleeve_names,
                "sleeve_weights_last": last_w,
                "config": {
                    "target_vol_base": cfg.TARGET_VOL_BASE,
                    "corr_penalty_factor": cfg.FACTOR_CORR_PENALTY,
                    "corr_penalty_sleeve": cfg.SLEEVE_CORR_PENALTY,
                    "stress_dd_cap": cfg.STRESS_DD_CAP,
                },
            }
            writer.write_meta(meta, tag=use_tag)

        return StrategyArtifacts(
            weights=final_weights,
            budget=budget,
            factor_weights=fw_total,
            ic_raw=xr.DataArray(),
            ic_smooth=xr.DataArray(),
            composite_score=score,
            F=F,
            meta=meta,
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )

