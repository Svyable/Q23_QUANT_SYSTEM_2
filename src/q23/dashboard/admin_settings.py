"""
Admin defaults persistence.

We keep two concepts:
1) Platform admin defaults (UI flags, defaults) in `.streamlit/q23_admin_settings.json`
2) Strategy default selection in `src/q23/strategies/enabled_strategies.json` (default_on_startup)

This keeps the dashboard "platform-like" while staying single-admin and simple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from q23.dashboard.core import _project_root
from q23.shared.config import TransactionCostConfig, TransactionCostScheme


ADMIN_SETTINGS_REL = Path(".streamlit/q23_admin_settings.json")
ENABLED_STRATEGIES_REL = Path("src/q23/strategies/enabled_strategies.json")


def _admin_settings_path() -> Path:
    return _project_root() / ADMIN_SETTINGS_REL


def _enabled_strategies_path() -> Path:
    return _project_root() / ENABLED_STRATEGIES_REL


def load_admin_settings() -> Dict[str, Any]:
    path = _admin_settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_admin_settings(settings: Dict[str, Any]) -> bool:
    path = _admin_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2, sort_keys=True))
        return True
    except Exception:
        return False


def load_default_strategy_id() -> Optional[str]:
    """Load default strategy from enabled_strategies.json (default_on_startup)."""
    path = _enabled_strategies_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        val = data.get("default_on_startup")
        return str(val) if val else None
    except Exception:
        return None


def save_default_strategy_id(strategy_id: str) -> bool:
    """Persist default strategy into enabled_strategies.json (default_on_startup)."""
    path = _enabled_strategies_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        data["default_on_startup"] = strategy_id
        path.write_text(json.dumps(data, indent=2, sort_keys=False))
        return True
    except Exception:
        return False


def load_tc_config() -> TransactionCostConfig:
    """Load transaction cost configuration from admin settings.
    
    Returns:
        TransactionCostConfig with persisted or default values
    """
    persisted = load_admin_settings()
    tc_data = persisted.get("tc_config", {})
    
    if tc_data:
        return TransactionCostConfig.from_dict(tc_data)
    
    # Return default Quantiacs config
    return TransactionCostConfig()


def save_tc_config(tc_config: TransactionCostConfig) -> bool:
    """Save transaction cost configuration to admin settings.
    
    Args:
        tc_config: Transaction cost configuration to save
        
    Returns:
        True if saved successfully
    """
    persisted = load_admin_settings()
    persisted["tc_config"] = tc_config.to_dict()
    return save_admin_settings(persisted)


def get_session_tc_config() -> TransactionCostConfig:
    """Get the current session's TC config.
    
    Returns TC config from session state if overridden,
    otherwise returns admin default.
    """
    if "q23_tc_config" in st.session_state:
        return st.session_state.q23_tc_config
    
    return load_tc_config()


def set_session_tc_config(tc_config: TransactionCostConfig) -> None:
    """Set the session-level TC config override.
    
    Args:
        tc_config: Transaction cost configuration for this session
    """
    st.session_state.q23_tc_config = tc_config


def initialize_admin_defaults() -> None:
    """
    Load persisted defaults and seed st.session_state once per session.

    This must be called BEFORE widgets are created (i.e., before render_sidebar()).
    """
    if st.session_state.get("_q23_admin_defaults_initialized", False):
        return

    persisted = load_admin_settings()

    # Basic UI defaults
    st.session_state.setdefault("admin_show_elite_overlay", bool(persisted.get("admin_show_elite_overlay", True)))
    st.session_state.setdefault("admin_default_date_preset", str(persisted.get("admin_default_date_preset", "2025")))
    st.session_state.setdefault("admin_default_include_benchmarks", bool(persisted.get("admin_default_include_benchmarks", False)))

    # Default strategy
    default_strategy = persisted.get("admin_default_strategy_id") or load_default_strategy_id()
    if default_strategy:
        st.session_state.setdefault("admin_default_strategy_id", str(default_strategy))

    # Transaction cost defaults
    tc_config = load_tc_config()
    st.session_state.setdefault("admin_tc_scheme", tc_config.scheme.value)
    st.session_state.setdefault("admin_tc_atr_window", tc_config.atr_window)
    st.session_state.setdefault("admin_tc_atr_multiplier", tc_config.atr_multiplier)
    st.session_state.setdefault("admin_tc_flat_bps", tc_config.flat_bps)

    # Seed UI widget keys so first load matches admin defaults
    st.session_state.setdefault("q23_include_benchmarks", st.session_state["admin_default_include_benchmarks"])
    st.session_state.setdefault("q23_date_preset", st.session_state["admin_default_date_preset"])
    
    # Seed session TC config from admin defaults
    st.session_state.setdefault("q23_tc_config", tc_config)

    st.session_state["_q23_admin_defaults_initialized"] = True

