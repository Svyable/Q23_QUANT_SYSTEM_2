from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.strategies.base import StrategyBase, StrategyConfig, StrategyArtifacts
from q23.strategies.registry import StrategyRegistry
from q23.strategies.gtp51max.config import gtp51max_config, GTP51MAX_FACTORS
from q23.strategy.outputs import OutputWriter
from q23.strategy.factors import FactorLibrary, FactorParams
from q23.strategy.ic_weighting import (
    CorrICWeightingParams,
    CorrelationAwareICWeighting,
)
from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams
from q23.strategy.risk_overlays import RiskOverlayParams, combined_overlay, apply_overlay
from q23.shared import config as global_config


@StrategyRegistry.register
class GTP51MAXStrategy(StrategyBase):
    """Sharpe-first ensemble with diversification-aware IC weighting."""

    @classmethod
    def strategy_id(cls) -> str:
        return "gtp51max"

    @property
    def config(self) -> StrategyConfig:
        cfg = gtp51max_config
        return StrategyConfig(
            name="gtp51max",
            display_name="GTP51MAX (Sharpe Optimized)",
            version="1.0",
            min_date=cfg.MIN_DATE,
            exchanges=list(cfg.EXCHANGES),
            factors=GTP51MAX_FACTORS,
            topn=cfg.TOPN_BASE,
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            target_vol=cfg.TARGET_VOL_BASE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            tc_bps=cfg.TC_BPS,
            description=(
                "Correlation-aware IC weighted ensemble combining defensive, "
                "multi-timeframe momentum, quality, and breakout signals with "
                "vol/drawdown overlays."
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
        if xr is None:
            raise ImportError("xarray required")

        from q23.strategy.engine import (
            _compute_portfolio_diag,
            _compute_factor_exposure_ts,
            _compute_factor_vectors_snapshot,
            compute_forward_returns,
        )
        from q23.strategy.data_loader import load_market_data

        cfg = gtp51max_config
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

        # Liquidity mask
        liq = ds.get("is_liquid", None)
        if liq is None:
            vol = ds.get("vol", None)
            if vol is not None:
                liq = xr.where(xr.ufuncs.isfinite(vol) & (vol > 0), 1.0, 0.0)
            else:
                liq = xr.ones_like(returns, dtype=float)
        else:
            liq = liq.fillna(1.0)

        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

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
            WEEK=5,
            MONTH=21,
            QUARTER=63,
            YEAR=252,
        )

        flib = FactorLibrary(bundle, params=factor_params, factors=GTP51MAX_FACTORS)
        F, _factor_art = flib.compute()
        self._factors = {nm: F.sel(factor=nm) for nm in GTP51MAX_FACTORS}

        ic_params = CorrICWeightingParams(
            lam=cfg.IC_LAMBDA,
            eps=cfg.EPS,
            clip_ic=0.20,
            use_positive_only=True,
            corr_penalty_strength=cfg.CORR_PENALTY,
            min_diversification=cfg.MIN_DIVERSIFICATION,
        )
        icw = CorrelationAwareICWeighting(F, fwd21, params=ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

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
            fixed_long_frac=1.0,
            side_split_mode="fixed",
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
        res = pc.build_long_only()

        final_weights = res.final_weights
        budget = res.budget

        # Drawdown/vol overlays on top of vol targeting from constructor
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

            writer = OutputWriter(
                base_name=self.get_base_name(),
                output_dir=output_dir,
            )

            writer.write_wide_weights(final_weights, tag=use_tag)
            writer.write_budget(budget, tag=use_tag)
            writer.write_factor_weights(fw, tag=use_tag)
            writer.write_ic(ic_raw, ic_smooth, tag=use_tag)

            diag = _compute_portfolio_diag(
                final_weights=final_weights,
                returns=returns,
                tc_bps=cfg.TC_BPS,
            )
            writer.write_portfolio_diag(diag, tag=use_tag)

            exp_ts = _compute_factor_exposure_ts(F=F, final_weights=final_weights)
            writer.write_factor_exposure(exp_ts, tag=use_tag)

            fv = _compute_factor_vectors_snapshot(
                F=F, composite_score=score, final_weights=final_weights, returns=returns
            )
            writer.write_factor_vectors(fv, tag=use_tag)

            meta = {
                "strategy_id": self.strategy_id(),
                "strategy_version": self.config.version,
                "display_name": self.config.display_name,
                "tag": use_tag,
                "run_timestamp": datetime.now().isoformat(),
                "date_range": [
                    str(final_weights.time.values[0])[:10],
                    str(final_weights.time.values[-1])[:10],
                ],
                "factors": GTP51MAX_FACTORS,
                "config": {
                    "topn_base": cfg.TOPN_BASE,
                    "topn_volatile": cfg.TOPN_VOLATILE,
                    "max_pos": cfg.MAX_POS,
                    "target_vol_base": cfg.TARGET_VOL_BASE,
                    "target_vol_volatile": cfg.TARGET_VOL_VOLATILE,
                    "lev_cap": cfg.LEV_CAP,
                    "lev_min": cfg.LEV_MIN,
                    "tc_bps": cfg.TC_BPS,
                    "exchanges": list(cfg.EXCHANGES),
                    "ic_lambda": cfg.IC_LAMBDA,
                    "corr_penalty": cfg.CORR_PENALTY,
                    "min_diversification": cfg.MIN_DIVERSIFICATION,
                    "vol_overlay_win": cfg.VOL_OVERLAY_WIN,
                    "vol_overlay_max": cfg.VOL_OVERLAY_MAX,
                },
            }
            writer.write_meta(meta, tag=use_tag)

        return StrategyArtifacts(
            weights=final_weights,
            budget=budget,
            factor_weights=fw,
            ic_raw=ic_raw,
            ic_smooth=ic_smooth,
            composite_score=score,
            F=F,
            meta=meta,
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )
