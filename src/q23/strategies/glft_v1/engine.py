"""GLFTv1 - Guéant–Lehalle–Fernandez-Tapia Microstructure Strategy Engine.

A systematic strategy based on market microstructure insights from the GLFT
optimal market-making framework.

Mathematical Foundation:
The GLFT model optimizes market-making under inventory risk:
- Optimal spread: δ = γσ²τ + (2/γ)ln(1 + γ/k)
- Reservation price: r = s - q·γ·σ²·τ

For alpha generation, we exploit:
1. Order flow information (buying/selling pressure)
2. Flow toxicity (probability of informed trading)
3. Inventory dynamics (mean-reversion pressure)
4. Price impact asymmetry (information edge)

This strategy is NOT market-making but uses MM insights for alpha:
- Low toxicity + strong OFI = momentum signal
- High inventory + reversal = mean-reversion signal
- Spread-adjusted returns filter noise
"""

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
from q23.strategies.glft_v1.config import (
    glft_v1_config,
    GLFT_V1_FACTORS,
    GLFT_NOVEL_FACTORS,
)
from q23.strategy.outputs import OutputWriter


@StrategyRegistry.register
class GLFTv1Strategy(StrategyBase):
    """GLFTv1 - Market Microstructure Alpha Strategy.
    
    A systematic strategy that applies insights from the GLFT market-making
    framework to generate alpha from microstructure inefficiencies.
    
    Key features:
    - 10 novel microstructure factors (OFI, VPIN, inventory, impact)
    - 15 proven complementary factors (liquidity, momentum, defensive)
    - Long-Short construction exploiting microstructure signals
    - Transaction-cost aware signal construction
    """

    @classmethod
    def strategy_id(cls) -> str:
        return "glft_v1"

    @property
    def config(self) -> StrategyConfig:
        cfg = glft_v1_config
        return StrategyConfig(
            name="glft_v1",
            display_name="GLFTv1 (Microstructure)",
            version="1.0",
            min_date=cfg.MIN_DATE,
            exchanges=list(cfg.EXCHANGES),
            factors=GLFT_V1_FACTORS,
            topn=cfg.TOPN_BASE,
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            target_vol=cfg.TARGET_VOL_BASE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            tc_bps=cfg.TC_BPS,
            description=(
                "Market microstructure strategy based on GLFT framework. "
                "Uses order flow, toxicity, and inventory signals. "
                "25 factors including 10 novel microstructure metrics. "
                "16 long / 8 short positions."
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
        """Return computed factors if available."""
        return self._factors if hasattr(self, "_factors") else {}

    def run(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        tag: Optional[str] = None,
        write_outputs: bool = True,
    ) -> StrategyArtifacts:
        """Execute the GLFTv1 strategy."""
        if xr is None:
            raise ImportError("xarray required for GLFTv1 strategy")

        from q23.strategy.engine import (
            _compute_portfolio_diag,
            _compute_factor_exposure_ts,
            _compute_factor_vectors_snapshot,
            compute_forward_returns,
        )
        from q23.strategy.data_loader import load_market_data
        from q23.strategy.factors import FactorParams
        from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
        from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams
        from q23.strategies.glft_v1.factors import GLFTFactorLibrary

        cfg = glft_v1_config
        use_min_date = min_date or cfg.MIN_DATE

        # 1) Load market data
        bundle = load_market_data(
            min_date=use_min_date,
            max_date=max_date,
            exchanges=list(cfg.EXCHANGES),
            pinned=None,
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

        # 2) Compute factors using GLFT-extended factor library
        # Include all params in initial creation (FactorParams is frozen)
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
            # GLFT-specific params
            OFI_SHORT_WIN=cfg.OFI_SHORT_WIN,
            OFI_MED_WIN=cfg.OFI_MED_WIN,
            OFI_LONG_WIN=cfg.OFI_LONG_WIN,
            VPIN_WIN=cfg.VPIN_WIN,
            SPREAD_EST_WIN=cfg.SPREAD_EST_WIN,
            IMPACT_WIN=cfg.IMPACT_WIN,
            INVENTORY_DECAY=cfg.INVENTORY_DECAY,
        )

        flib = GLFTFactorLibrary(
            bundle,
            params=factor_params,
            factors=GLFT_V1_FACTORS,
        )
        F, _factor_art = flib.compute()

        # Store factors
        self._factors = {nm: F.sel(factor=nm) for nm in GLFT_V1_FACTORS if nm in F.factor.values}

        # 3) Forward returns for IC
        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

        # 4) IC weighting
        ic_params = ICWeightingParams(lam=cfg.IC_LAMBDA, eps=cfg.EPS)
        icw = DynamicICWeighting(F, fwd21, params=ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

        # 5) Portfolio construction - Long-Short
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
            fixed_long_frac=0.65,  # Slightly more long-biased
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

        # 6) Output generation
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
                "model": "GLFT Market Microstructure",
                "factors": {
                    "total": len(GLFT_V1_FACTORS),
                    "glft_novel": len(GLFT_NOVEL_FACTORS),
                    "glft_novel_list": GLFT_NOVEL_FACTORS,
                    "all": GLFT_V1_FACTORS,
                },
                "glft_params": {
                    "ofi_windows": [cfg.OFI_SHORT_WIN, cfg.OFI_MED_WIN, cfg.OFI_LONG_WIN],
                    "vpin_win": cfg.VPIN_WIN,
                    "impact_win": cfg.IMPACT_WIN,
                    "inventory_decay": cfg.INVENTORY_DECAY,
                },
                "config": {
                    "long_seats": cfg.LONG_SEATS,
                    "short_seats": cfg.SHORT_SEATS,
                    "topn_base": cfg.TOPN_BASE,
                    "max_pos": cfg.MAX_POS,
                    "target_vol_base": cfg.TARGET_VOL_BASE,
                    "lev_cap": cfg.LEV_CAP,
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
            meta=meta if write_outputs else {},
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )
