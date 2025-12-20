"""Q23 Neural Alpha Strategy.

A state-of-the-art Pure Alpha Long-Short strategy incorporating 15 novel factors
derived from behavioral finance, market microstructure, and machine learning research.

Usage:
    from q23.strategies.q23_neural_alpha import Q23NeuralAlphaStrategy
    
    strategy = Q23NeuralAlphaStrategy()
    artifacts = strategy.run()
    
    # Or via registry
    from q23.strategies.registry import StrategyRegistry
    strategy = StrategyRegistry.get_instance("q23_neural_alpha")
"""

from q23.strategies.q23_neural_alpha.config import (
    NeuralAlphaConfig,
    neural_alpha_config,
    NEURAL_ALPHA_FACTORS,
    NEURAL_ALPHA_NOVEL_FACTORS,
    NEURAL_ALPHA_EXISTING_FACTORS,
)
from q23.strategies.q23_neural_alpha.engine import Q23NeuralAlphaStrategy
from q23.strategies.q23_neural_alpha.factors import NeuralAlphaFactorLibrary

__all__ = [
    "Q23NeuralAlphaStrategy",
    "NeuralAlphaConfig",
    "neural_alpha_config",
    "NeuralAlphaFactorLibrary",
    "NEURAL_ALPHA_FACTORS",
    "NEURAL_ALPHA_NOVEL_FACTORS",
    "NEURAL_ALPHA_EXISTING_FACTORS",
]
