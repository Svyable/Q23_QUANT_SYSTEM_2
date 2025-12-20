from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.strategies.base import StrategyBase, StrategyConfig, StrategyArtifacts
from q23.strategies.registry import StrategyRegistry
from q23.strategies.nasnys_v4.config import v4_config, V4_24_FACTORS
from q23.strategy.outputs import OutputWriter


@StrategyRegistry.register
class NASNYSV4Strategy(StrategyBase):

    @classmethod
    def strategy_id(cls) -> str:
        return "nasnys_v4"

    @property
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="nasnys_v4",
            display_name="NASNYS V4 (24 Factors)",
            version="4.0",
            min_date=v4_config.MIN_DATE,
            exchanges=list(v4_config.EXCHANGES),
            factors=V4_24_FACTORS,
            topn=v4_config.TOPN_BASE,
            max_pos=v4_config.MAX_POS,
            min_pos=v4_config.MIN_POS,
            target_vol=v4_config.TARGET_VOL_BASE,
            lev_cap=v4_config.LEV_CAP,
            lev_min=v4_config.LEV_MIN,
            tc_bps=v4_config.TC_BPS,
            description="24-factor IC-weighted blend with NAS+NYS universe, long-only capability",
            long_only=v4_config.LONG_ONLY,
            long_seats=v4_config.LONG_SEATS,
            short_seats=v4_config.SHORT_SEATS,
            weight_smooth_alpha=v4_config.WEIGHT_SMOOTH_ALPHA,
            score_smooth_win=v4_config.SCORE_SMOOTH_WIN,
            softmax_tilt_alpha=v4_config.SOFTMAX_TILT_ALPHA,
            risk_off_stretch=v4_config.RISK_OFF_STRETCH,
            risk_off_floor=v4_config.RISK_OFF_FLOOR,
            dd_win=v4_config.DD_WIN,
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
    ) -> StrategyArtifacts:
        if xr is None:
            raise ImportError("xarray required")

        from q23.strategy.engine import StrategyEngine, _compute_portfolio_diag, _compute_factor_exposure_ts, _compute_factor_vectors_snapshot
        from q23.strategy.data_loader import load_market_data
        from q23.strategy.factors import FactorLibrary, FactorParams
        from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
        from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams

        cfg = v4_config
        use_min_date = min_date or cfg.MIN_DATE

        bundle = load_market_data(
            min_date=use_min_date,
            exchanges=list(cfg.EXCHANGES),
            pinned=None,
        )

        ds = bundle.data
        returns = bundle.returns

        factor_params = FactorParams(eps=cfg.EPS)
        flib = FactorLibrary(bundle, params=factor_params, factors=V4_24_FACTORS)
        F, _factor_art = flib.compute()

        self._factors = {nm: F.sel(factor=nm) for nm in V4_24_FACTORS}

        from q23.strategy.engine import compute_forward_returns
        fwd = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

        ic_params = ICWeightingParams(lam=0.95, eps=cfg.EPS)
        icw = DynamicICWeighting(F, fwd, params=ic_params)
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
        )

        pc = PortfolioConstructor(
            scores=score,
            returns=returns,
            liquidity=ds.get("is_liquid", None),
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
                "factors": V4_24_FACTORS,
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
