"""Q23 Neural Alpha v2 Strategy.

An enhanced version of the Neural Alpha strategy combining the best aspects of:
- Neural Alpha v1: Novel factors, best Sharpe/Calmar
- GTP51MAX: Lowest drawdown, superior risk controls  
- NASNYS V4: Highest CAGR, proven factors
- COMPOSER v1: Balanced Sharpe/Sortino

Key improvements:
- 40 factors (15 novel + 25 proven) vs v1's 32
- Faster drawdown response (DD_WIN: 252 → 126)
- Better diversification (MAX_POS: 0.12 → 0.09)
- Lower target volatility (TARGET_VOL: 0.18 → 0.16)
- More balanced L/S (14L/6S vs 12L/8S)

Usage:
    from q23.strategies.q23_neural_alpha_v2 import Q23NeuralAlphaV2Strategy
    
    strategy = Q23NeuralAlphaV2Strategy()
    artifacts = strategy.run()
"""

from q23.strategies.q23_neural_alpha_v2.config import (
    NeuralAlphaV2Config,
    neural_alpha_v2_config,
    NEURAL_ALPHA_V2_FACTORS,
    NEURAL_ALPHA_V2_NOVEL_FACTORS,
    NEURAL_ALPHA_V2_DEFENSIVE_FACTORS,
    NEURAL_ALPHA_V2_MOMENTUM_FACTORS,
    NEURAL_ALPHA_V2_TECHNICAL_FACTORS,
    NEURAL_ALPHA_V2_QUALITY_FACTORS,
)
from q23.strategies.q23_neural_alpha_v2.engine import Q23NeuralAlphaV2Strategy

__all__ = [
    "Q23NeuralAlphaV2Strategy",
    "NeuralAlphaV2Config",
    "neural_alpha_v2_config",
    "NEURAL_ALPHA_V2_FACTORS",
    "NEURAL_ALPHA_V2_NOVEL_FACTORS",
    "NEURAL_ALPHA_V2_DEFENSIVE_FACTORS",
    "NEURAL_ALPHA_V2_MOMENTUM_FACTORS",
    "NEURAL_ALPHA_V2_TECHNICAL_FACTORS",
    "NEURAL_ALPHA_V2_QUALITY_FACTORS",
]
