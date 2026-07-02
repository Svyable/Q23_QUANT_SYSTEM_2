from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.strategies.base import StrategyBase, StrategyConfig, StrategyArtifacts
from q23.strategies.registry import StrategyRegistry
from q23.strategies.qs23_hybrid_alpha.config import qs23_config, QS23_FACTORS
from q23.strategy.outputs import OutputWriter
from q23.shared import config as global_config


@StrategyRegistry.register
class QS23HybridAlphaStrategy(StrategyBase):

    @classmethod
    def strategy_id(cls) -> str:
        return "qs23_hybrid_alpha"

    @property
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="qs23_hybrid_alpha",
            display_name="QS23 Hybrid Alpha (NAS)",
            version="3.0",
            min_date=qs23_config.MIN_DATE,
            exchanges=[qs23_config.EXCHANGE],
            factors=QS23_FACTORS,
            topn=qs23_config.TOPN,
            max_pos=qs23_config.MAX_POS,
            min_pos=qs23_config.MIN_POS,
            target_vol=qs23_config.TARGET_VOL,
            lev_cap=qs23_config.LEV_CAP,
            lev_min=qs23_config.LEV_MIN,
            tc_bps=qs23_config.TC_BPS,
            description="Daily rebalance hybrid alpha with IC weighting, NAS-focused, 2024+ data",
            long_only=True,
            long_seats=qs23_config.TOPN,
            short_seats=0,
            weight_smooth_alpha=qs23_config.WEIGHT_SMOOTH_ALPHA,
            score_smooth_win=qs23_config.SCORE_SMOOTH_WIN,
            risk_off_stretch=qs23_config.RISK_OFF_STRETCH,
            risk_off_floor=qs23_config.RISK_OFF_FLOOR,
            dd_win=qs23_config.DD_WIN,
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

        from q23.strategy.engine import StrategyEngine, _compute_portfolio_diag, _compute_factor_exposure_ts, _compute_factor_vectors_snapshot
        from q23.strategy.data_loader import load_market_data
        from q23.strategy.factors import FactorLibrary, FactorParams
        from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
        from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams

        cfg = qs23_config
        use_min_date = min_date or cfg.MIN_DATE

        bundle = load_market_data(
            min_date=use_min_date,
            max_date=max_date,
            exchanges=[cfg.EXCHANGE],
            pinned=None,
            fill_recent_days=global_config.cfg.marketstack.DEFAULT_LOOKBACK_DAYS,
            use_marketstack=global_config.cfg.marketstack.ENABLED,
            force_live_data=force_live_data,
            strategy_id=self.strategy_id(),
        )

        ds = bundle.data
        returns = bundle.returns

        # Get liquidity mask
        liq = ds.get("is_liquid", None)
        if liq is None:
            vol = ds.get("vol", None)
            if vol is not None:
                liq = xr.where(np.isfinite(vol) & (vol > 0), 1.0, 0.0)
            else:
                liq = xr.ones_like(returns, dtype=float)
        else:
            liq = liq.fillna(0)

        # Compute forward returns for IC
        from q23.strategy.engine import compute_forward_returns
        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

        # Use FactorLibrary for standard factors
        factor_params = FactorParams(
            eps=cfg.EPS,
            BETA_WIN=cfg.BETA_WIN,
            IDIO_WIN=cfg.IDIO_WIN,
            DOWN_WIN=cfg.DOWN_WIN,
            CORR_WIN=cfg.CORR_WIN,
            ADV_WIN_LONG=cfg.ADV_WIN_LONG,
            ADV_WIN_SHORT=cfg.ADV_WIN_SHORT,
            MOM_WIN=cfg.MOM_WIN,
            REV_WIN=cfg.REV_WIN,
            DON_WIN=cfg.DON_WIN,
            EMA_FAST=cfg.EMA_FAST,
            EMA_SLOW=cfg.EMA_SLOW,
            ATR_WIN=cfg.ATR_WIN,
        )

        flib = FactorLibrary(bundle, params=factor_params, factors=QS23_FACTORS)
        F, _factor_art = flib.compute()

        self._factors = {nm: F.sel(factor=nm) for nm in QS23_FACTORS}

        # Use DynamicICWeighting for IC computation
        ic_params = ICWeightingParams(lam=cfg.IC_LAMBDA, eps=cfg.EPS)
        icw = DynamicICWeighting(F, fwd21, params=ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

        # Portfolio construction using PortfolioConstructor
        portfolio_params = PortfolioParams(
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            eps=cfg.EPS,
            topn_base=cfg.TOPN,
            topn_volatile=cfg.TOPN,
            score_smooth_win=cfg.SCORE_SMOOTH_WIN,
            weight_smooth_alpha=cfg.WEIGHT_SMOOTH_ALPHA,
            softmax_tilt_alpha=0.90,
            use_equal_topk=False,
            long_seats=cfg.TOPN,
            short_seats=0,
            fixed_long_frac=1.0,
            side_split_mode="fixed",
            budget_mode="vol_target",
            pm_gross_target=1.0,
            target_vol_base=cfg.TARGET_VOL,
            target_vol_volatile=cfg.TARGET_VOL,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
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
                final_weights=final_weights, returns=returns, tc_bps=cfg.TC_BPS
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
                "date_range": [str(final_weights.time.values[0])[:10], str(final_weights.time.values[-1])[:10]],
                "factors": QS23_FACTORS,
                "config": {
                    "topn": cfg.TOPN,
                    "max_pos": cfg.MAX_POS,
                    "target_vol": cfg.TARGET_VOL,
                    "lev_cap": cfg.LEV_CAP,
                    "lev_min": cfg.LEV_MIN,
                    "tc_bps": cfg.TC_BPS,
                    "exchange": cfg.EXCHANGE,
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
            meta=meta if write_outputs else {},
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )
