"""GLFTv3 - Advanced GLFT Microstructure Strategy.

V3 adds cutting-edge microstructure signals:
- Multi-timeframe OFI alignment
- Sector-relative flow signals
- Regime-adaptive toxicity
- Market flow momentum
- Execution quality scoring
- Adverse selection decomposition
- Optimal spread signal from GLFT formula
- Flow persistence factor

Target Profile: Sharpe > 3.5, MaxDD < -5%
"""

from q23.strategies.glft_v3.config import (
    GLFTv3Config,
    glft_v3_config,
    GLFT_V3_FACTORS,
    GLFT_V3_NOVEL_FACTORS,
    GLFT_V3_COMPLEMENTARY_FACTORS,
)
from q23.strategies.glft_v3.engine import GLFTv3Strategy

__all__ = [
    "GLFTv3Strategy",
    "GLFTv3Config",
    "glft_v3_config",
    "GLFT_V3_FACTORS",
    "GLFT_V3_NOVEL_FACTORS",
    "GLFT_V3_COMPLEMENTARY_FACTORS",
]
