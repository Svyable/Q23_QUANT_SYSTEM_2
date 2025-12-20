from __future__ import annotations

from typing import Dict, List, Type, Optional

from q23.strategies.base import StrategyBase, StrategyConfig, StrategyInfo


class StrategyRegistry:
    _strategies: Dict[str, Type[StrategyBase]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, strategy_cls: Type[StrategyBase]) -> Type[StrategyBase]:
        strategy_id = strategy_cls.strategy_id()
        cls._strategies[strategy_id] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, strategy_id: str) -> Type[StrategyBase]:
        cls._ensure_initialized()
        if strategy_id not in cls._strategies:
            available = list(cls._strategies.keys())
            raise KeyError(
                f"Strategy '{strategy_id}' not found. Available: {available}"
            )
        return cls._strategies[strategy_id]

    @classmethod
    def get_instance(cls, strategy_id: str) -> StrategyBase:
        strategy_cls = cls.get(strategy_id)
        return strategy_cls()

    @classmethod
    def list_ids(cls) -> List[str]:
        cls._ensure_initialized()
        return list(cls._strategies.keys())

    @classmethod
    def list_configs(cls) -> Dict[str, StrategyConfig]:
        cls._ensure_initialized()
        return {
            sid: cls._strategies[sid]().config for sid in cls._strategies
        }

    @classmethod
    def list_info(cls) -> List[StrategyInfo]:
        cls._ensure_initialized()
        infos = []
        for sid, strategy_cls in cls._strategies.items():
            cfg = strategy_cls().config
            infos.append(
                StrategyInfo(
                    strategy_id=sid,
                    display_name=cfg.display_name,
                    version=cfg.version,
                    description=cfg.description,
                    min_date=cfg.min_date,
                    factors_count=len(cfg.factors),
                    long_only=cfg.long_only,
                )
            )
        return infos

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._auto_discover()
        cls._initialized = True

    @classmethod
    def _auto_discover(cls) -> None:
        try:
            from q23.strategies import nasnys_v4
        except ImportError:
            pass

        try:
            from q23.strategies import qs23_hybrid_alpha
        except ImportError:
            pass

        try:
            from q23.strategies import q23_composer_v1
        except ImportError:
            pass

        try:
            from q23.strategies import benchmarks  # Auto-registers benchmark strategies
        except ImportError:
            pass

        try:
            from q23.strategies import q23_neural_alpha  # Auto-registers Neural Alpha strategy
        except ImportError:
            pass

        try:
            from q23.strategies import q23_composer_v2  # Auto-registers Composer v2 strategy
        except ImportError:
            pass

        try:
            from q23.strategies import gtp51max  # Sharpe-optimized strategy
        except ImportError:
            pass

        try:
            from q23.strategies import q23_neural_alpha_v2  # Enhanced Neural Alpha v2
        except ImportError:
            pass

        try:
            from q23.strategies import ou_v1  # OU Mean-Reversion strategy
        except ImportError:
            pass

        try:
            from q23.strategies import glft_v1  # GLFT Microstructure strategy
        except ImportError:
            pass

        try:
            from q23.strategies import glft_v2  # Enhanced GLFT Microstructure v2
        except ImportError:
            pass

        try:
            from q23.strategies import glft_v3  # Advanced GLFT Microstructure v3
        except ImportError:
            pass

    @classmethod
    def reset(cls) -> None:
        cls._strategies.clear()
        cls._initialized = False
