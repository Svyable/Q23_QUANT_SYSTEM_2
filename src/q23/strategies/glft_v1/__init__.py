"""GLFTv1 - Guéant–Lehalle–Fernandez-Tapia Market Microstructure Strategy.

A systematic strategy based on the GLFT optimal market-making framework,
adapted for alpha generation from microstructure inefficiencies.

Key concepts from GLFT:
- γ (gamma): Inventory risk aversion coefficient
- σ: Volatility of mid-price
- k: Order arrival intensity
- Optimal spread: δ = γσ²τ + (2/γ)ln(1 + γ/k)

Alpha signals derived from:
1. Order Flow Imbalance (OFI): Buying vs selling pressure
2. VPIN: Probability of informed trading (toxicity)
3. Inventory Dynamics: Mean-reversion from accumulated positions
4. Price Impact Asymmetry: Information edge detection

The strategy:
- Goes long stocks with positive OFI and low toxicity
- Goes short stocks with negative OFI or high toxicity
- Uses spread-adjusted signals to filter noise

Usage:
    from q23.strategies.glft_v1 import GLFTv1Strategy
    
    strategy = GLFTv1Strategy()
    artifacts = strategy.run()
"""

from q23.strategies.glft_v1.config import (
    GLFTv1Config,
    glft_v1_config,
    GLFT_V1_FACTORS,
    GLFT_NOVEL_FACTORS,
    GLFT_COMPLEMENTARY_FACTORS,
)
from q23.strategies.glft_v1.engine import GLFTv1Strategy

__all__ = [
    "GLFTv1Strategy",
    "GLFTv1Config",
    "glft_v1_config",
    "GLFT_V1_FACTORS",
    "GLFT_NOVEL_FACTORS",
    "GLFT_COMPLEMENTARY_FACTORS",
]
