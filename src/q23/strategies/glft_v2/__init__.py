"""GLFTv2 - Enhanced GLFT Microstructure Strategy.

V2 adds advanced microstructure signals:
- Kyle's Lambda for permanent price impact
- Volume-clock adjusted OFI
- Bulk Volume Classification
- Cross-sectional flow dispersion
- Avellaneda-Stoikov reservation price
- Amihud-GLFT hybrid liquidity
"""

from q23.strategies.glft_v2.config import (
    GLFTv2Config,
    glft_v2_config,
    GLFT_V2_FACTORS,
    GLFT_V2_NOVEL_FACTORS,
    GLFT_V2_COMPLEMENTARY_FACTORS,
)
from q23.strategies.glft_v2.engine import GLFTv2Strategy

__all__ = [
    "GLFTv2Strategy",
    "GLFTv2Config",
    "glft_v2_config",
    "GLFT_V2_FACTORS",
    "GLFT_V2_NOVEL_FACTORS",
    "GLFT_V2_COMPLEMENTARY_FACTORS",
]
