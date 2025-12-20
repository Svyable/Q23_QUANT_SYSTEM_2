"""
Strategy Generator

Generates new Python strategy code from UI parameters, enabling PMs to create
deployable strategies from the Live Strategy Lab without manual coding.

Generated strategies include:
- config.py: Configuration dataclass and factor list
- engine.py: Strategy execution engine
- __init__.py: Module exports

The generator also updates the registry auto-discovery to include the new strategy.
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StrategyGenerator:
    """
    Generate new strategy Python code from configuration.
    
    Usage:
        generator = StrategyGenerator()
        strategy_dir = generator.generate(
            variant_name="neural_alpha_conservative",
            base_strategy_id="q23_neural_alpha",
            factors=["inv_vol", "resid_mom", ...],
            config_overrides={"max_pos": 0.08, "long_only": True, ...},
        )
    """
    
    def __init__(self, strategies_dir: Optional[Path] = None):
        """
        Initialize the generator.
        
        Args:
            strategies_dir: Directory where strategies are stored.
                           Defaults to src/q23/strategies/
        """
        if strategies_dir is None:
            # Find strategies directory relative to this file
            strategies_dir = Path(__file__).parent
        self.strategies_dir = Path(strategies_dir)
    
    def generate(
        self,
        variant_name: str,
        base_strategy_id: Optional[str],
        factors: List[str],
        config_overrides: Dict[str, Any],
    ) -> Path:
        """
        Generate a new strategy variant.
        
        Args:
            variant_name: Name for the new strategy (e.g., "neural_alpha_conservative")
            base_strategy_id: ID of the base strategy to inherit from, or None for new
            factors: List of factor names to include
            config_overrides: Dict of config parameters to override
            
        Returns:
            Path to the generated strategy directory
        """
        # Sanitize name
        safe_name = self._sanitize_name(variant_name)
        strategy_dir = self.strategies_dir / safe_name
        
        # Create directory
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        self._write_config(strategy_dir, safe_name, factors, config_overrides, base_strategy_id)
        self._write_engine(strategy_dir, safe_name, base_strategy_id)
        self._write_init(strategy_dir, safe_name)
        
        # Update registry auto-discovery
        self._update_registry(safe_name)
        
        return strategy_dir
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize strategy name for use as Python identifier."""
        # Replace non-alphanumeric with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'strategy_' + sanitized
        return sanitized or 'custom_strategy'
    
    def _to_class_name(self, name: str) -> str:
        """Convert snake_case to PascalCase for class name."""
        parts = name.split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def _write_config(
        self,
        dir: Path,
        name: str,
        factors: List[str],
        overrides: Dict[str, Any],
        base_strategy_id: Optional[str],
    ) -> None:
        """Write the config.py file."""
        class_name = self._to_class_name(name)
        timestamp = datetime.now().isoformat()
        
        # Extract configuration values with defaults
        min_date = overrides.get('min_date', '2020-01-01')
        exchanges = overrides.get('exchanges', ['NAS', 'NYS'])
        long_seats = overrides.get('long_seats', 12)
        short_seats = overrides.get('short_seats', 8)
        long_only = overrides.get('long_only', False)
        topn_base = overrides.get('topn_base', 20)
        topn_vol = overrides.get('topn_vol', 15)
        max_pos = overrides.get('max_pos', 0.10)
        min_pos = overrides.get('min_pos', 0.003)
        target_vol = overrides.get('target_vol', 0.15)
        target_vol_volatile = overrides.get('target_vol_volatile', target_vol * 0.7)
        lev_cap = overrides.get('lev_cap', 1.5)
        lev_min = overrides.get('lev_min', 0.3)
        tc_bps = overrides.get('tc_bps', 10.0)
        w_smooth = overrides.get('w_smooth', 0.30)
        softmax_alpha = overrides.get('softmax_alpha', 0.85)
        score_smooth_win = overrides.get('score_smooth_win', 3)
        risk_off_stretch = overrides.get('risk_off_stretch', 0.60)
        risk_off_floor = overrides.get('risk_off_floor', 0.50)
        dd_win = overrides.get('dd_win', 252)
        ic_lambda = overrides.get('ic_lambda', 0.94)
        eps = overrides.get('eps', 1e-12)
        
        # Format factors list for Python code
        factors_str = ',\n    '.join(f'"{f}"' for f in factors)
        
        base_info = f"Based on: {base_strategy_id}" if base_strategy_id else "Created from scratch"
        
        config_code = f'''"""Auto-generated strategy configuration: {name}

{base_info}
Created: {timestamp}

This file was generated by the Strategy Generator from the Live Strategy Lab.
You can modify these parameters and re-run the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class {class_name}Config:
    """Configuration for {name} strategy."""
    
    # Data configuration
    MIN_DATE: str = "{min_date}"
    EXCHANGES: List[str] = {repr(exchanges)}
    
    # Portfolio style
    LONG_SEATS: int = {long_seats}
    SHORT_SEATS: int = {short_seats}
    LONG_ONLY: bool = {long_only}
    
    # Position sizing
    TOPN_BASE: int = {topn_base}
    TOPN_VOLATILE: int = {topn_vol}
    MAX_POS: float = {max_pos}
    MIN_POS: float = {min_pos}
    
    # Risk parameters
    TARGET_VOL_BASE: float = {target_vol}
    TARGET_VOL_VOLATILE: float = {target_vol_volatile}
    LEV_CAP: float = {lev_cap}
    LEV_MIN: float = {lev_min}
    
    # Transaction costs
    TC_BPS: float = {tc_bps}
    
    # Smoothing parameters
    WEIGHT_SMOOTH_ALPHA: float = {w_smooth}
    SCORE_SMOOTH_WIN: int = {score_smooth_win}
    SOFTMAX_TILT_ALPHA: float = {softmax_alpha}
    
    # Risk throttle
    RISK_OFF_STRETCH: float = {risk_off_stretch}
    RISK_OFF_FLOOR: float = {risk_off_floor}
    DD_WIN: int = {dd_win}
    
    # IC weighting
    IC_LAMBDA: float = {ic_lambda}
    
    # Numeric precision
    EPS: float = {eps}
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


# Default configuration instance
{name}_config = {class_name}Config()


# Factor list for this strategy ({len(factors)} factors)
{name.upper()}_FACTORS: List[str] = [
    {factors_str}
]
'''
        
        (dir / "config.py").write_text(config_code)
    
    def _write_engine(
        self,
        dir: Path,
        name: str,
        base_strategy_id: Optional[str],
    ) -> None:
        """Write the engine.py file."""
        class_name = self._to_class_name(name)
        timestamp = datetime.now().isoformat()
        
        base_info = f"Based on: {base_strategy_id}" if base_strategy_id else "New strategy"
        
        engine_code = f'''"""Auto-generated strategy engine: {name}

{base_info}
Created: {timestamp}
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
from q23.strategies.{name}.config import (
    {name}_config,
    {name.upper()}_FACTORS,
)
from q23.strategy.outputs import OutputWriter


@StrategyRegistry.register
class {class_name}Strategy(StrategyBase):
    """Auto-generated strategy: {name}"""

    @classmethod
    def strategy_id(cls) -> str:
        return "{name}"

    @property
    def config(self) -> StrategyConfig:
        cfg = {name}_config
        return StrategyConfig(
            name="{name}",
            display_name="{class_name.replace('_', ' ')}",
            version="1.0",
            min_date=cfg.MIN_DATE,
            exchanges=list(cfg.EXCHANGES),
            factors={name.upper()}_FACTORS,
            topn=cfg.TOPN_BASE,
            max_pos=cfg.MAX_POS,
            min_pos=cfg.MIN_POS,
            target_vol=cfg.TARGET_VOL_BASE,
            lev_cap=cfg.LEV_CAP,
            lev_min=cfg.LEV_MIN,
            tc_bps=cfg.TC_BPS,
            description="Auto-generated strategy from Live Strategy Lab",
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
        return self._factors if hasattr(self, "_factors") else {{}}

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

        from q23.strategy.engine import (
            _compute_portfolio_diag,
            _compute_factor_exposure_ts,
            _compute_factor_vectors_snapshot,
            compute_forward_returns,
        )
        from q23.strategy.data_loader import load_market_data
        from q23.strategy.factors import FactorLibrary, FactorParams
        from q23.strategy.ic_weighting import DynamicICWeighting, ICWeightingParams
        from q23.strategy.portfolio import PortfolioConstructor, PortfolioParams

        cfg = {name}_config
        use_min_date = min_date or cfg.MIN_DATE

        # Load market data
        bundle = load_market_data(
            min_date=use_min_date,
            max_date=max_date,
            exchanges=list(cfg.EXCHANGES),
            pinned=None,
        )

        ds = bundle.data
        returns = bundle.returns

        # Liquidity mask
        liq = ds.get("is_liquid", None)
        if liq is None:
            vol = ds.get("vol", None)
            if vol is not None:
                liq = xr.where(np.isfinite(vol) & (vol > 0), 1.0, 0.0)
            else:
                liq = xr.ones_like(returns, dtype=float)
        else:
            liq = liq.fillna(0)

        # Compute factors
        factor_params = FactorParams(eps=cfg.EPS)
        flib = FactorLibrary(bundle, params=factor_params, factors={name.upper()}_FACTORS)
        F, _factor_art = flib.compute()

        self._factors = {{nm: F.sel(factor=nm) for nm in {name.upper()}_FACTORS if nm in F.factor.values}}

        # Forward returns for IC
        fwd21 = compute_forward_returns(returns, horizon=21, eps=cfg.EPS)

        # IC weighting
        ic_params = ICWeightingParams(lam=cfg.IC_LAMBDA, eps=cfg.EPS)
        icw = DynamicICWeighting(F, fwd21, params=ic_params)
        score = icw.composite_score()
        ic_raw, ic_smooth, fw = icw.artifacts()

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
            fixed_long_frac=0.60,
            side_split_mode="fixed" if cfg.LONG_ONLY else "prop_mass",
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
            liquidity=liq,
            asset_ids=bundle.asset_ids,
            pin_idx=bundle.pin_idx,
            params=portfolio_params,
        )
        
        if cfg.LONG_ONLY:
            res = pc.build_long_only()
        else:
            res = pc.build_long_short()

        final_weights = res.final_weights
        budget = res.budget

        use_tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        meta = {{}}
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

            meta = {{
                "strategy_id": self.strategy_id(),
                "strategy_version": self.config.version,
                "display_name": self.config.display_name,
                "tag": use_tag,
                "run_timestamp": datetime.now().isoformat(),
                "date_range": [
                    str(final_weights.time.values[0])[:10],
                    str(final_weights.time.values[-1])[:10],
                ],
                "factors": {name.upper()}_FACTORS,
                "config": {{
                    "long_seats": cfg.LONG_SEATS,
                    "short_seats": cfg.SHORT_SEATS,
                    "topn_base": cfg.TOPN_BASE,
                    "max_pos": cfg.MAX_POS,
                    "target_vol_base": cfg.TARGET_VOL_BASE,
                    "lev_cap": cfg.LEV_CAP,
                    "tc_bps": cfg.TC_BPS,
                    "exchanges": list(cfg.EXCHANGES),
                }},
            }}
            writer.write_meta(meta, tag=use_tag)

        return StrategyArtifacts(
            weights=final_weights,
            budget=budget,
            factor_weights=fw,
            ic_raw=ic_raw,
            ic_smooth=ic_smooth,
            composite_score=score,
            F=F,
            meta=meta if write_outputs else {{}},
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )
'''
        
        (dir / "engine.py").write_text(engine_code)
    
    def _write_init(self, dir: Path, name: str) -> None:
        """Write the __init__.py file."""
        class_name = self._to_class_name(name)
        timestamp = datetime.now().isoformat()
        
        init_code = f'''"""Auto-generated strategy: {name}

Created: {timestamp}

Usage:
    from q23.strategies.{name} import {class_name}Strategy
    
    strategy = {class_name}Strategy()
    artifacts = strategy.run()
"""

from q23.strategies.{name}.config import (
    {class_name}Config,
    {name}_config,
    {name.upper()}_FACTORS,
)
from q23.strategies.{name}.engine import {class_name}Strategy

__all__ = [
    "{class_name}Strategy",
    "{class_name}Config",
    "{name}_config",
    "{name.upper()}_FACTORS",
]
'''
        
        (dir / "__init__.py").write_text(init_code)
    
    def _update_registry(self, name: str) -> None:
        """Update the registry to auto-discover the new strategy."""
        registry_path = self.strategies_dir / "registry.py"
        
        if not registry_path.exists():
            return
        
        content = registry_path.read_text()
        
        # Check if already registered
        import_line = f"from q23.strategies import {name}"
        if import_line in content:
            return
        
        # Find the last try/except block for auto-discovery
        # Add our new strategy import there
        new_import = f'''
        try:
            from q23.strategies import {name}  # Auto-registered
        except ImportError:
            pass'''
        
        # Find the _auto_discover method and add before its last line
        if "_auto_discover" in content and "pass" in content:
            # Find the last occurrence of the benchmark import block
            benchmark_pattern = 'from q23.strategies import benchmarks'
            if benchmark_pattern in content:
                # Insert after benchmarks
                content = content.replace(
                    f'''        try:
            from q23.strategies import benchmarks  # Auto-registers benchmark strategies
        except ImportError:
            pass''',
                    f'''        try:
            from q23.strategies import benchmarks  # Auto-registers benchmark strategies
        except ImportError:
            pass
{new_import}'''
                )
                registry_path.write_text(content)
