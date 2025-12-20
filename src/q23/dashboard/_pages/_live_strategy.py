"""
Live Strategy Lab

A comprehensive page for experimenting with and creating strategies:
- Select a base strategy from the registry or start fresh
- Full factor selection from 50+ available factors (grouped by category)
- Portfolio parameter tuning
- Ephemeral runs (cached, auto-deleted) vs Permanent saves (Python code generation)

This replaces the legacy StrategyEngine approach with the new strategy system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.shared.config import cfg, PortfolioMode
from q23.dashboard.core import (
    discover_available_strategies,
    get_strategy_display_name,
    get_strategy_configs,
)
from q23.strategy.factors import (
    FACTOR_REGISTRY,
    FactorCategory,
    get_factors_by_category,
    get_all_categories,
    V4_24_FACTORS,
    MTF_MOMENTUM_FACTORS,
    ALL_FACTORS,
)


# =============================================================================
# FACTOR CATEGORY STYLING
# =============================================================================

CATEGORY_STYLES = {
    FactorCategory.DEFENSIVE: {"icon": "🛡️", "name": "Defensive", "color": "#3498db"},
    FactorCategory.MOMENTUM: {"icon": "📈", "name": "Momentum", "color": "#2ecc71"},
    FactorCategory.REVERSAL: {"icon": "🔄", "name": "Reversal", "color": "#e67e22"},
    FactorCategory.TECHNICAL: {"icon": "📊", "name": "Technical", "color": "#9b59b6"},
    FactorCategory.QUALITY: {"icon": "✨", "name": "Quality", "color": "#f1c40f"},
    FactorCategory.VALUE: {"icon": "💰", "name": "Value", "color": "#1abc9c"},
    FactorCategory.LIQUIDITY: {"icon": "💧", "name": "Liquidity", "color": "#00bcd4"},
    FactorCategory.MICROSTRUCTURE: {"icon": "🔬", "name": "Microstructure", "color": "#607d8b"},
    FactorCategory.INTERACTION: {"icon": "🔗", "name": "Interaction", "color": "#673ab7"},
    FactorCategory.CROSS_SECTIONAL: {"icon": "📐", "name": "Cross-Sectional", "color": "#ff5722"},
    FactorCategory.MTF_MOMENTUM: {"icon": "⏱️", "name": "MTF Momentum", "color": "#4caf50"},
    FactorCategory.MTF_ALIGNMENT: {"icon": "🎯", "name": "MTF Alignment", "color": "#8bc34a"},
    FactorCategory.MTF_HLOC: {"icon": "📉", "name": "MTF HLOC", "color": "#cddc39"},
}


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def _init_session_state() -> None:
    """Initialize session state for live strategy parameters."""
    if "live_base_strategy" not in st.session_state:
        st.session_state.live_base_strategy = "(blank)"
    
    if "live_selected_factors" not in st.session_state:
        st.session_state.live_selected_factors = list(V4_24_FACTORS)
    
    if "live_params" not in st.session_state:
        st.session_state.live_params = {
            "max_pos": float(cfg.strategy.MAX_POS),
            "min_pos": float(cfg.strategy.MIN_POS),
            "topn_base": int(cfg.strategy.TOPN_BASE),
            "topn_vol": int(cfg.strategy.TOPN_VOLATILE),
            "softmax_alpha": float(cfg.strategy.SOFTMAX_TILT_ALPHA),
            "w_smooth": float(cfg.strategy.WEIGHT_SMOOTH_ALPHA),
            "long_seats": int(cfg.strategy.LONG_SEATS),
            "short_seats": int(cfg.strategy.SHORT_SEATS),
            "long_only": cfg.strategy.PORTFOLIO_MODE == PortfolioMode.LONG_ONLY,
            "target_vol": float(cfg.strategy.TARGET_VOL_BASE),
            "lev_cap": float(cfg.strategy.LEV_CAP),
            "lev_min": float(cfg.strategy.LEV_MIN),
            "pins": "",
            "blacklist": "",
        }
    
    if "live_save_mode" not in st.session_state:
        st.session_state.live_save_mode = "ephemeral"


def _load_strategy_config(strategy_id: str) -> Optional[Dict[str, Any]]:
    """Load configuration from a registered strategy."""
    try:
        from q23.strategies.registry import StrategyRegistry
        strategy = StrategyRegistry.get_instance(strategy_id)
        config = strategy.config
        
        return {
            "factors": list(config.factors),
            "max_pos": config.max_pos,
            "min_pos": config.min_pos,
            "topn_base": config.topn,
            "topn_vol": getattr(config, 'topn_volatile', config.topn),
            "softmax_alpha": config.softmax_tilt_alpha,
            "w_smooth": config.weight_smooth_alpha,
            "long_seats": config.long_seats,
            "short_seats": config.short_seats,
            "long_only": config.long_only,
            "target_vol": config.target_vol,
            "lev_cap": config.lev_cap,
            "lev_min": config.lev_min,
        }
    except Exception as e:
        st.warning(f"Could not load config for {strategy_id}: {e}")
        return None


def _populate_from_config(config: Dict[str, Any]) -> None:
    """Populate session state from a loaded config."""
    if config is None:
        return
    
    if "factors" in config:
        st.session_state.live_selected_factors = config["factors"]
    
    params = st.session_state.live_params
    for key in ["max_pos", "min_pos", "topn_base", "topn_vol", "softmax_alpha", 
                "w_smooth", "long_seats", "short_seats", "long_only", 
                "target_vol", "lev_cap", "lev_min"]:
        if key in config:
            params[key] = config[key]


# =============================================================================
# FACTOR SELECTION UI
# =============================================================================

def _render_factor_category(
    category: FactorCategory,
    current_selection: List[str],
) -> List[str]:
    """Render factor selection for a single category."""
    style = CATEGORY_STYLES.get(category, {"icon": "📋", "name": category.value, "color": "#666"})
    factors = get_factors_by_category(category)
    
    if not factors:
        return []
    
    selected_count = sum(1 for f in factors if f in current_selection)
    is_all_selected = selected_count == len(factors)
    
    col_btn, col_factors = st.columns([1, 4])
    
    with col_btn:
        btn_type = "primary" if is_all_selected else "secondary"
        btn_label = f"{style['icon']} {style['name']} ({selected_count}/{len(factors)})"
        
        if st.button(btn_label, key=f"live_cat_{category.value}", type=btn_type):
            if is_all_selected:
                # Deselect all in this category
                return [f for f in current_selection if f not in factors]
            else:
                # Select all in this category
                new_selection = list(current_selection)
                for f in factors:
                    if f not in new_selection:
                        new_selection.append(f)
                return new_selection
    
    with col_factors:
        selected = st.multiselect(
            style["name"],
            options=factors,
            default=[f for f in factors if f in current_selection],
            key=f"live_ms_{category.value}",
            label_visibility="collapsed",
        )
        return selected
    
    return [f for f in factors if f in current_selection]


def _render_factor_selection() -> List[str]:
    """Render the complete factor selection UI."""
    st.markdown("### 🎯 Factor Selection")
    st.caption("Select factors to include in the composite alpha score. Click category buttons to toggle all factors in that group.")
    
    current_factors = list(st.session_state.live_selected_factors)
    
    # Quick actions
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("✅ All Factors", key="live_select_all"):
            st.session_state.live_selected_factors = list(ALL_FACTORS)
            st.rerun()
    with qa2:
        if st.button("📋 V4 Set (24)", key="live_select_v4"):
            st.session_state.live_selected_factors = list(V4_24_FACTORS)
            st.rerun()
    with qa3:
        if st.button("⏱️ MTF Set", key="live_select_mtf"):
            st.session_state.live_selected_factors = list(MTF_MOMENTUM_FACTORS)
            st.rerun()
    with qa4:
        if st.button("❌ Clear All", key="live_clear_all"):
            st.session_state.live_selected_factors = []
            st.rerun()
    
    st.markdown("")
    
    # Render each category
    all_selected = []
    categories_to_render = [
        FactorCategory.DEFENSIVE,
        FactorCategory.MOMENTUM,
        FactorCategory.REVERSAL,
        FactorCategory.TECHNICAL,
        FactorCategory.QUALITY,
        FactorCategory.LIQUIDITY,
        FactorCategory.MICROSTRUCTURE,
        FactorCategory.INTERACTION,
        FactorCategory.CROSS_SECTIONAL,
        FactorCategory.MTF_MOMENTUM,
        FactorCategory.MTF_ALIGNMENT,
        FactorCategory.MTF_HLOC,
    ]
    
    for category in categories_to_render:
        factors = get_factors_by_category(category)
        if not factors:
            continue
        
        selected_count = sum(1 for f in factors if f in current_factors)
        is_all_selected = selected_count == len(factors)
        style = CATEGORY_STYLES.get(category, {"icon": "📋", "name": category.value})
        
        col_btn, col_factors = st.columns([1, 4])
        
        with col_btn:
            btn_type = "primary" if is_all_selected else "secondary"
            btn_label = f"{style['icon']} {style['name']} ({selected_count}/{len(factors)})"
            
            if st.button(btn_label, key=f"live_cat_{category.value}", type=btn_type):
                if is_all_selected:
                    st.session_state.live_selected_factors = [f for f in current_factors if f not in factors]
                else:
                    new_sel = list(current_factors)
                    for f in factors:
                        if f not in new_sel:
                            new_sel.append(f)
                    st.session_state.live_selected_factors = new_sel
                st.rerun()
        
        with col_factors:
            selected = st.multiselect(
                style["name"],
                options=factors,
                default=[f for f in factors if f in current_factors],
                key=f"live_ms_{category.value}",
                label_visibility="collapsed",
            )
            all_selected.extend(selected)
    
    # Update session state with combined selection
    combined = list(set(all_selected))
    if set(combined) != set(current_factors):
        st.session_state.live_selected_factors = combined
    
    # Summary
    total = len(st.session_state.live_selected_factors)
    if total == 0:
        st.warning("⚠️ No factors selected! Please select at least one factor.")
    else:
        st.success(f"📊 **{total} factors selected**")
    
    return st.session_state.live_selected_factors


# =============================================================================
# PORTFOLIO PARAMETERS UI
# =============================================================================

def _render_portfolio_params() -> Dict[str, Any]:
    """Render portfolio parameter controls."""
    st.markdown("### ⚙️ Portfolio Parameters")
    
    params = st.session_state.live_params
    
    # Position style
    col_style1, col_style2 = st.columns(2)
    with col_style1:
        long_only = st.checkbox(
            "Long Only",
            value=params["long_only"],
            key="live_long_only",
            help="If unchecked, uses long-short positioning"
        )
        params["long_only"] = long_only
    
    with col_style2:
        if not long_only:
            st.caption(f"Style: Long {params['long_seats']} / Short {params['short_seats']}")
    
    # Long/Short seats (only if not long-only)
    if not long_only:
        col_ls1, col_ls2 = st.columns(2)
        with col_ls1:
            params["long_seats"] = st.slider(
                "Long Seats",
                min_value=5, max_value=30, value=params["long_seats"],
                key="live_long_seats"
            )
        with col_ls2:
            params["short_seats"] = st.slider(
                "Short Seats",
                min_value=0, max_value=20, value=params["short_seats"],
                key="live_short_seats"
            )
    
    # Position sizing
    col1, col2 = st.columns(2)
    with col1:
        params["max_pos"] = st.slider(
            "Max Position Weight",
            min_value=0.01, max_value=0.25, value=params["max_pos"],
            step=0.005, key="live_max_pos",
            help="Maximum weight per position"
        )
        params["topn_base"] = st.slider(
            "Top N (Base Regime)",
            min_value=5, max_value=80, value=params["topn_base"],
            key="live_topn_base",
            help="Number of positions in normal volatility regime"
        )
        params["softmax_alpha"] = st.slider(
            "Softmax Tilt Alpha",
            min_value=0.0, max_value=3.0, value=params["softmax_alpha"],
            step=0.05, key="live_softmax",
            help="Score concentration (higher = more concentrated)"
        )
    
    with col2:
        params["min_pos"] = st.slider(
            "Min Position Weight",
            min_value=0.0, max_value=0.02, value=params["min_pos"],
            step=0.001, key="live_min_pos",
            help="Minimum weight per position"
        )
        params["topn_vol"] = st.slider(
            "Top N (Volatile Regime)",
            min_value=5, max_value=80, value=params["topn_vol"],
            key="live_topn_vol",
            help="Number of positions in volatile regime"
        )
        params["w_smooth"] = st.slider(
            "Weight Smoothing Alpha",
            min_value=0.0, max_value=0.9, value=params["w_smooth"],
            step=0.02, key="live_w_smooth",
            help="Weight smoothing (0 = none, higher = more)"
        )
    
    # Risk parameters
    st.markdown("#### Risk Parameters")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        params["target_vol"] = st.slider(
            "Target Volatility",
            min_value=0.05, max_value=0.30, value=params["target_vol"],
            step=0.01, key="live_target_vol"
        )
    with col_r2:
        params["lev_cap"] = st.slider(
            "Leverage Cap",
            min_value=0.5, max_value=3.0, value=params["lev_cap"],
            step=0.1, key="live_lev_cap"
        )
    with col_r3:
        params["lev_min"] = st.slider(
            "Leverage Floor",
            min_value=0.1, max_value=1.0, value=params["lev_min"],
            step=0.1, key="live_lev_min"
        )
    
    # Position constraints
    st.markdown("#### Position Constraints")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        params["pins"] = st.text_input(
            "Pinned Positions (comma-separated)",
            value=params["pins"],
            key="live_pins",
            help="Symbols to always include"
        )
    with col_p2:
        params["blacklist"] = st.text_input(
            "Blacklisted Positions (comma-separated)",
            value=params["blacklist"],
            key="live_blacklist",
            help="Symbols to exclude"
        )
    
    st.session_state.live_params = params
    return params


# =============================================================================
# RUN EXECUTION
# =============================================================================

def _generate_run_tag(name: Optional[str] = None, ephemeral: bool = False) -> str:
    """Generate a tag for the strategy run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "ephemeral" if ephemeral else "live"
    
    if name and name.strip():
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip())
        sanitized = sanitized.strip("_")
        if sanitized:
            return f"{prefix}_{sanitized}_{timestamp}"
    
    return f"{prefix}_{timestamp}"


def _run_strategy_with_params(
    factors: List[str],
    params: Dict[str, Any],
    base_strategy: Optional[str],
    tag: str,
    output_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Run the strategy with the given parameters.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        from q23.strategy.factors import FactorParams
        from q23.strategy.ic_weighting import ICWeightingParams
        from q23.strategy.portfolio import PortfolioParams
        from q23.strategy.outputs import OutputWriter
        
        # Use the appropriate strategy engine
        if base_strategy and base_strategy != "(blank)":
            # Run using registered strategy with overridden params
            from q23.strategies.registry import StrategyRegistry
            
            strategy = StrategyRegistry.get_instance(base_strategy)
            
            # For now, run the base strategy with the tag
            # TODO: Override params in the strategy run
            artifacts = strategy.run(tag=tag, write_outputs=True)
            return True, f"Strategy '{base_strategy}' executed with tag: {tag}"
        else:
            # Run using legacy StrategyEngine
            from q23.strategy.engine import StrategyEngine
            
            fparams = FactorParams(eps=float(cfg.strategy.EPS), robust_cs_z=True)
            iparams = ICWeightingParams(lam=0.95, eps=float(cfg.strategy.EPS))
            pparams = PortfolioParams(
                max_pos=float(params["max_pos"]),
                min_pos=float(params["min_pos"]),
                eps=float(cfg.strategy.EPS),
                topn_base=int(params["topn_base"]),
                topn_volatile=int(params["topn_vol"]),
                score_smooth_win=int(cfg.strategy.SCORE_SMOOTH_WIN),
                weight_smooth_alpha=float(params["w_smooth"]),
                softmax_tilt_alpha=float(params["softmax_alpha"]),
                use_equal_topk=False,
                long_seats=int(params["long_seats"]),
                short_seats=int(params["short_seats"]),
                fixed_long_frac=0.6,
                side_split_mode="fixed" if params["long_only"] else "prop_mass",
                budget_mode="vol_target",
                pm_gross_target=1.0,
                target_vol_base=float(params["target_vol"]),
                target_vol_volatile=float(params["target_vol"]) * 0.7,
                lev_cap=float(params["lev_cap"]),
                lev_min=float(params["lev_min"]),
            )
            
            pin_list = [x.strip() for x in params["pins"].split(",") if x.strip()]
            
            eng = StrategyEngine(
                factor_params=fparams,
                ic_params=iparams,
                portfolio_params=pparams,
                factors=factors if factors else list(V4_24_FACTORS),
            )
            
            eng.run(pinned=pin_list, exchanges=None, tag=tag)
            return True, f"Strategy executed with tag: {tag}"
            
    except Exception as e:
        return False, f"Strategy run failed: {str(e)}"


# =============================================================================
# SAVE OPTIONS UI
# =============================================================================

def _render_save_options() -> Tuple[str, str, bool]:
    """
    Render save options and return (mode, name, should_run).
    
    Returns:
        Tuple of (save_mode, strategy_name, should_run_now)
    """
    st.markdown("### 💾 Save & Run Options")
    
    save_mode = st.radio(
        "Run Type",
        options=["ephemeral", "permanent"],
        format_func=lambda x: {
            "ephemeral": "🔄 Ephemeral (test run, cached temporarily)",
            "permanent": "💾 Permanent (save as new strategy)",
        }.get(x, x),
        index=0 if st.session_state.live_save_mode == "ephemeral" else 1,
        key="live_save_mode_radio",
        horizontal=True,
    )
    st.session_state.live_save_mode = save_mode
    
    strategy_name = ""
    
    if save_mode == "ephemeral":
        st.caption(
            "Ephemeral runs are cached and auto-deleted when the dashboard closes. "
            "Great for quick experiments."
        )
        run_name = st.text_input(
            "Run Name (optional)",
            value="",
            key="live_ephemeral_name",
            placeholder="e.g., test_momentum_only",
        )
        strategy_name = run_name
        
    else:  # permanent
        st.caption(
            "Permanent saves create a new Python strategy in `src/q23/strategies/`. "
            "The strategy will be available in the Strategy Selection dropdown."
        )
        strategy_name = st.text_input(
            "Strategy Name (required)",
            value="",
            key="live_permanent_name",
            placeholder="e.g., neural_alpha_conservative",
            help="Use lowercase with underscores. This becomes the strategy ID.",
        )
        
        if strategy_name:
            # Validate name
            sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in strategy_name.lower())
            sanitized = sanitized.strip("_")
            
            if sanitized != strategy_name:
                st.warning(f"Strategy name will be sanitized to: `{sanitized}`")
                strategy_name = sanitized
            
            # Check if exists
            existing = discover_available_strategies()
            if strategy_name in existing:
                st.error(f"Strategy `{strategy_name}` already exists! Choose a different name.")
                strategy_name = ""
    
    st.divider()
    
    # Run button
    col_run, col_info = st.columns([1, 2])
    
    with col_run:
        can_run = True
        if save_mode == "permanent" and not strategy_name:
            can_run = False
        if len(st.session_state.live_selected_factors) == 0:
            can_run = False
        
        run_btn = st.button(
            "▶️ Run Strategy",
            type="primary",
            key="live_run_btn",
            disabled=not can_run,
        )
    
    with col_info:
        if save_mode == "ephemeral":
            st.caption("This will run a full backtest. Results will be cached temporarily.")
        else:
            st.caption(
                "This will generate Python code for the new strategy and run it. "
                "The strategy will be permanently available."
            )
    
    return save_mode, strategy_name, run_btn


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_live_strategy_page(preset_dir: Path) -> None:
    """
    Render the Live Strategy Lab page.
    
    Args:
        preset_dir: Directory for saving/loading presets (legacy, still supported)
    """
    st.subheader("🧪 Live Strategy Lab")
    
    st.markdown(
        """
        **Design, test, and save custom strategies.** Start from an existing strategy 
        or build from scratch with full access to 50+ factors.
        
        - **Ephemeral runs**: Quick experiments that are cached and auto-deleted
        - **Permanent saves**: Generate Python strategy code for production use
        """
    )
    
    st.divider()
    
    # Initialize session state
    _init_session_state()
    
    # =========================================================================
    # BASE STRATEGY SELECTION
    # =========================================================================
    st.markdown("### 📂 Base Strategy")
    
    available = discover_available_strategies()
    options = ["(blank)"] + available
    
    col_base, col_load = st.columns([3, 1])
    
    with col_base:
        base_strategy = st.selectbox(
            "Start from",
            options=options,
            index=options.index(st.session_state.live_base_strategy) if st.session_state.live_base_strategy in options else 0,
            format_func=lambda x: get_strategy_display_name(x) if x != "(blank)" else "🆕 New Strategy (blank slate)",
            key="live_base_strategy_select",
        )
        st.session_state.live_base_strategy = base_strategy
    
    with col_load:
        st.markdown("")  # Spacing
        st.markdown("")
        if base_strategy != "(blank)":
            if st.button("📥 Load Config", key="live_load_base"):
                config = _load_strategy_config(base_strategy)
                if config:
                    _populate_from_config(config)
                    st.success(f"Loaded config from {get_strategy_display_name(base_strategy)}")
                    st.rerun()
    
    if base_strategy != "(blank)":
        configs = get_strategy_configs()
        if base_strategy in configs:
            cfg_info = configs[base_strategy]
            st.caption(f"Base: v{cfg_info.version} | {len(cfg_info.factors)} factors | {'Long-only' if cfg_info.long_only else 'Long-Short'}")
    
    st.divider()
    
    # =========================================================================
    # FACTOR SELECTION
    # =========================================================================
    factors = _render_factor_selection()
    
    st.divider()
    
    # =========================================================================
    # PORTFOLIO PARAMETERS
    # =========================================================================
    params = _render_portfolio_params()
    
    st.divider()
    
    # =========================================================================
    # SAVE & RUN OPTIONS
    # =========================================================================
    save_mode, strategy_name, should_run = _render_save_options()
    
    # Execute run if button pressed
    if should_run:
        if save_mode == "permanent" and strategy_name:
            # Generate strategy code first
            with st.spinner("🔨 Generating strategy code..."):
                try:
                    from q23.strategies.strategy_generator import StrategyGenerator
                    
                    generator = StrategyGenerator()
                    strategy_dir = generator.generate(
                        variant_name=strategy_name,
                        base_strategy_id=base_strategy if base_strategy != "(blank)" else None,
                        factors=factors,
                        config_overrides=params,
                    )
                    st.success(f"✅ Strategy code generated at: `{strategy_dir}`")
                    
                except ImportError:
                    st.warning("Strategy generator not available. Running with legacy engine...")
                except Exception as e:
                    st.error(f"Failed to generate strategy: {e}")
                    st.stop()
        
        # Run the strategy
        with st.spinner("🔄 Running strategy backtest... This may take several minutes."):
            tag = _generate_run_tag(strategy_name, ephemeral=(save_mode == "ephemeral"))
            success, message = _run_strategy_with_params(
                factors=factors,
                params=params,
                base_strategy=base_strategy if base_strategy != "(blank)" else None,
                tag=tag,
            )
            
            if success:
                st.success(f"✅ {message}")
                st.info(f"📍 Tag: `{tag}` - Switch to Overview or refresh to see results.")
                st.balloons()
            else:
                st.error(f"❌ {message}")
    
    # =========================================================================
    # CONFIGURATION SUMMARY
    # =========================================================================
    with st.expander("📋 Current Configuration Summary", expanded=False):
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("**Factors**")
            st.text(f"Total: {len(factors)}")
            st.text(f"Categories: {len(set(FACTOR_REGISTRY[f].category for f in factors if f in FACTOR_REGISTRY))}")
        
        with col_s2:
            st.markdown("**Portfolio**")
            style_str = "Long-Only" if params['long_only'] else f"Long {params['long_seats']} / Short {params['short_seats']}"
            st.text(f"Style: {style_str}")
            st.text(f"Max Position: {params['max_pos']:.1%}")
            st.text(f"Target Vol: {params['target_vol']:.1%}")
        
        st.markdown("**Selected Factors:**")
        st.text(", ".join(factors[:10]) + ("..." if len(factors) > 10 else ""))
