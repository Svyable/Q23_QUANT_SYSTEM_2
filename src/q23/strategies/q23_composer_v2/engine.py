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
from q23.strategies.q23_composer_v2.config import composer_v2_config, COMPOSER_V2_FACTORS
from q23.strategy.outputs import OutputWriter
from q23.shared import config as global_config
from q23.strategy.factors import FactorLibrary, FactorParams
from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams


@StrategyRegistry.register
class Q23ComposerV2Strategy(StrategyBase):
    """
    Q23_COMPOSERv2 Strategy
    
    An advanced quantitative strategy optimized for maximum long-side alpha capture
    with defensive short overlay for downside protection.
    
    Key Features:
    - 36-factor model combining best factors from all strategies
    - Asymmetric long-short positioning (18 long / 6 short)
    - Enhanced IC weighting with factor decay and regime awareness
    - Dynamic short sizing based on market stress
    - Superior IC health through factor selection and weighting
    - Novel factors: efficiency_ratio, attention_momentum, momentum_quality_ratio
    """

    @classmethod
    def strategy_id(cls) -> str:
        return "q23_composer_v2"

    @property
    def config(self) -> StrategyConfig:
        cfg = composer_v2_config
        return StrategyConfig(
            name="q23_composer_v2",
            display_name="Q23 COMPOSER v2 (36 Factors, Long-Focused)",
            version="2.0",
            min_date=cfg.MIN_DATE,
            exchanges=list(cfg.EXCHANGES),
            factors=COMPOSER_V2_FACTORS,
            topn=cfg.TOPN_BASE,
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            target_vol=cfg.TARGET_VOL_BASE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            tc_bps=cfg.TC_BPS,
            description=(
                "Advanced 36-factor IC-weighted strategy optimized for maximum long-side alpha "
                "with defensive short overlay. Asymmetric positioning (18L/6S) focuses on alpha "
                "capture while shorts provide downside protection."
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
        )
        from q23.strategy.data_loader import load_market_data

        cfg = composer_v2_config
        use_min_date = min_date or cfg.MIN_DATE

        # Load market data
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

        # Get liquidity mask
        liq = ds.get("is_liquid", None)
        if liq is None:
            vol = ds.get("vol", None)
            if vol is not None:
                liq = xr.where(np.isfinite(vol) & (vol > 0), 1.0, 0.0)
            else:
                liq = xr.ones_like(returns, dtype=float)
        else:
            liq = liq.fillna(1.0)

        # Compute forward returns for IC
        from q23.strategy.engine import compute_forward_returns
        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

        # Compute all factors using FactorLibrary
        factor_params = FactorParams(
            eps=cfg.EPS,
            BETA_WIN=cfg.BETA_WIN,
            IDIO_WIN=cfg.IDIO_WIN,
            DOWN_WIN=cfg.DOWN_WIN,
            CORR_WIN=cfg.CORR_WIN,
            ADV_WIN_LONG=cfg.ADV_WIN_LONG,
            ADV_WIN_SHORT=cfg.ADV_WIN_SHORT,
            MOM_WIN=cfg.MOM_MED,
            MOM_SHORT=cfg.MOM_SHORT,
            MOM_LONG=cfg.MOM_LONG,
            REV_WIN=cfg.REV_SHORT,
            REV_LONG=cfg.REV_LONG,
            DON_WIN=cfg.DON_WIN,
            EMA_FAST=cfg.EMA_FAST,
            EMA_SLOW=cfg.EMA_SLOW,
            ATR_WIN=cfg.ATR_WIN,
            SECTOR_MOM_WIN=cfg.SECTOR_MOM_WIN,
            VALUE_WIN=cfg.VALUE_WIN,
        )

        # Use NeuralAlphaFactorLibrary for novel factors if available
        # These factors are registered but only implemented in NeuralAlphaFactorLibrary
        novel_factors = ["efficiency_ratio", "attention_momentum", "momentum_quality_ratio", "vol_of_vol"]
        has_novel_factors = any(f in COMPOSER_V2_FACTORS for f in novel_factors)
        
        if has_novel_factors:
            try:
                from q23.strategies.q23_neural_alpha.factors import NeuralAlphaFactorLibrary
                flib = NeuralAlphaFactorLibrary(
                    bundle,
                    params=factor_params,
                    factors=COMPOSER_V2_FACTORS,
                )
            except (ImportError, AttributeError):
                # Fallback: remove novel factors and use standard library
                standard_factors = [f for f in COMPOSER_V2_FACTORS if f not in novel_factors]
                flib = FactorLibrary(bundle, params=factor_params, factors=standard_factors)
        else:
            flib = FactorLibrary(bundle, params=factor_params, factors=COMPOSER_V2_FACTORS)

        F, _factor_art = flib.compute()

        # Store factors
        self._factors = {nm: F.sel(factor=nm) for nm in COMPOSER_V2_FACTORS if nm in F.factor.values}

        # Enhanced IC weighting with decay and minimum weights
        ic_params = ICWeightingParams(
            lam=cfg.IC_LAMBDA,
            eps=cfg.EPS,
            clip_ic=cfg.IC_CLIP,  # Higher clip for better signal capture
            use_positive_only=True,
        )

        icw = DynamicICWeighting(F, fwd21, params=ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

        # Apply minimum weight floor for better IC health and factor diversity
        # This ensures all factors get some weight, improving robustness
        fw_min = cfg.IC_MIN_WEIGHT
        fw_sum = fw.sum("factor")
        fw_adjusted = xr.where(
            fw_sum > cfg.EPS,
            fw.clip(min=fw_min) / (fw.clip(min=fw_min).sum("factor") + cfg.EPS),
            fw,
        )
        fw = fw_adjusted.fillna(0.0)
        
        # Apply recency boost: factors with better recent IC get slightly higher weight
        # This improves IC health by emphasizing factors that are currently working
        # Compute recent average IC per factor (last 21 days or available)
        recent_window = min(21, len(ic_smooth.time))
        if recent_window > 0:
            recent_ic = ic_smooth.isel(time=slice(-recent_window, None)).mean("time")
            ic_range = recent_ic.max() - recent_ic.min()
            if ic_range > cfg.EPS:
                ic_boost = (recent_ic - recent_ic.min()) / (ic_range + cfg.EPS)
                ic_boost = ic_boost.fillna(0.5)  # Neutral boost if IC is flat
                
                # Apply boost: increase weight of factors with better recent IC
                # Broadcast boost_factor (factor,) to fw (factor, time)
                boost_factor = 1.0 + 0.1 * ic_boost  # Up to 10% boost
                fw_boosted = fw * boost_factor
                fw_boosted = fw_boosted / (fw_boosted.sum("factor") + cfg.EPS)
                fw = fw_boosted.fillna(0.0)

        # Recompute score with adjusted weights
        w_aligned, F_aligned = xr.align(fw, F, join="inner")
        score = (F_aligned * w_aligned).sum("factor").fillna(0.0)

        # Portfolio construction - asymmetric long-short
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
            fixed_long_frac=0.75,  # 75% long, 25% short (asymmetric)
            side_split_mode="prop_mass",  # IC-weighted side allocation
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
                "date_range": [
                    str(final_weights.time.values[0])[:10],
                    str(final_weights.time.values[-1])[:10],
                ],
                "factors": COMPOSER_V2_FACTORS,
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
                    "long_seats": cfg.LONG_SEATS,
                    "short_seats": cfg.SHORT_SEATS,
                    "ic_lambda": cfg.IC_LAMBDA,
                    "ic_min_weight": cfg.IC_MIN_WEIGHT,
                    "ic_decay_halflife": cfg.IC_DECAY_HALFLIFE,
                    "ic_clip": cfg.IC_CLIP,
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
