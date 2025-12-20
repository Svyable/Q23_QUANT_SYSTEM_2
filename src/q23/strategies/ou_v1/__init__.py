"""OUv1 - Ornstein-Uhlenbeck Mean-Reversion Strategy.

A systematic mean-reversion strategy based on the canonical OU process:
dX = θ(μ - X)dt + σdW

Key concepts:
- θ (kappa): Mean-reversion speed
- μ (mu): Long-term equilibrium
- σ (sigma): Volatility of innovations
- Half-life = ln(2)/θ

The strategy estimates OU parameters cross-sectionally and:
1. Goes long stocks below equilibrium with short half-lives
2. Goes short stocks above equilibrium with short half-lives
3. Weights signals by reversion strength (θ)

Usage:
    from q23.strategies.ou_v1 import OUv1Strategy
    
    strategy = OUv1Strategy()
    artifacts = strategy.run()
"""

from q23.strategies.ou_v1.config import (
    OUv1Config,
    ou_v1_config,
    OU_V1_FACTORS,
    OU_NOVEL_FACTORS,
    OU_COMPLEMENTARY_FACTORS,
)
from q23.strategies.ou_v1.engine import OUv1Strategy

__all__ = [
    "OUv1Strategy",
    "OUv1Config",
    "ou_v1_config",
    "OU_V1_FACTORS",
    "OU_NOVEL_FACTORS",
    "OU_COMPLEMENTARY_FACTORS",
]
