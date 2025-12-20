"""
Live Strategy Re-Run Page

Provides a dedicated page for re-running the full strategy with modified parameters.
Key Features:
- Full factor selection (defensive, alpha, quality, interaction)
- Portfolio parameter controls (max/min positions, topN, softmax, smoothing)
- Preset save/load functionality
- Position constraints (pins, blacklist)
- Tags output with preset names for easy identification in Run Selection

This is distinct from What-If which only transforms existing weights.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.shared.config import cfg
from q23.strategy.engine import StrategyEngine
from q23.strategy.factors import FactorParams, V4_24_FACTORS
from q23.strategy.ic_weighting import ICWeightingParams
from q23.strategy.portfolio import PortfolioParams


# =============================================================================
# COLOR-CODED BUTTON CSS
# =============================================================================

FACTOR_BUTTON_CSS = """
<style>
/* Factor Category Quick-Select Buttons */
/* Row 1: All, Defensive, Alpha */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(1) > div:nth-child(1) button {
    background: linear-gradient(135deg, #636e72 0%, #7f8c8d 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(1) > div:nth-child(2) button {
    background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(1) > div:nth-child(3) button {
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Row 2: Quality, Combo, Reset */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(2) > div:nth-child(1) button {
    background: linear-gradient(135deg, #d4a516 0%, #f1c40f 100%) !important;
    color: #1a1a1a !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(2) > div:nth-child(2) button {
    background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]):nth-of-type(2) > div:nth-child(3) button {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Factor Category Expanders with color-coded left borders */
div[data-testid="stExpander"] details {
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}

/* Defensive - Blue */
div[data-testid="stExpander"]:has(summary:contains("Defensive")) details,
div[data-testid="stExpander"]:nth-of-type(1) details {
    border-left: 4px solid #3498db !important;
}

/* Alpha - Green */
div[data-testid="stExpander"]:has(summary:contains("Alpha")) details,
div[data-testid="stExpander"]:nth-of-type(2) details {
    border-left: 4px solid #2ecc71 !important;
}

/* Quality - Gold */
div[data-testid="stExpander"]:has(summary:contains("Quality")) details,
div[data-testid="stExpander"]:nth-of-type(3) details {
    border-left: 4px solid #f1c40f !important;
}

/* Interaction - Purple */
div[data-testid="stExpander"]:has(summary:contains("Interaction")) details,
div[data-testid="stExpander"]:nth-of-type(4) details {
    border-left: 4px solid #9b59b6 !important;
}

/* Run button styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #219a52 0%, #27ae60 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4);
}
</style>
"""


# Factor definitions
ALL_FACTORS = list(V4_24_FACTORS)

DEFENSIVE_FACTORS = [
    "inv_vol", "inv_idio", "inv_down", "low_corr",
    "low_beta", "liquidity", "amihud_inv",
]

ALPHA_FACTORS = [
    "resid_mom", "srev", "breakout", "slope", "calm_flow",
    "prox_52w_high", "resid_mom_mix", "vol_surprise",
    "vol_breakout", "ma_cloud",
]

QUALITY_FACTORS = [
    "idio_tail_risk", "idio_jump_freq", "beta_stability", "micro_noise",
]

INTERACTION_FACTORS = ["value_mom", "quality_defensive", "rel_sector_mom"]


def _init_session_state() -> None:
    """Initialize session state for live strategy parameters."""
    if "live_pm_max_pos" not in st.session_state:
        st.session_state.live_pm_max_pos = float(cfg.strategy.MAX_POS)
        st.session_state.live_pm_min_pos = float(cfg.strategy.MIN_POS)
        st.session_state.live_pm_topn_base = int(cfg.strategy.TOPN_BASE)
        st.session_state.live_pm_topn_vol = int(cfg.strategy.TOPN_VOLATILE)
        st.session_state.live_pm_softmax_alpha = float(cfg.strategy.SOFTMAX_TILT_ALPHA)
        st.session_state.live_pm_w_smooth = float(cfg.strategy.WEIGHT_SMOOTH_ALPHA)
        st.session_state.live_pm_pins = ""
        st.session_state.live_pm_blacklist = ""

    if "live_selected_factors" not in st.session_state:
        st.session_state.live_selected_factors = list(ALL_FACTORS)


def _generate_live_tag(preset_name: Optional[str] = None) -> str:
    """
    Generate a tag for live strategy runs.
    
    Format: live_{preset_name}_{YYYYMMDD_HHMMSS} if preset provided
            live_{YYYYMMDD_HHMMSS} otherwise
    
    This allows users to easily match Run Selection dropdown with their saved presets.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if preset_name and preset_name.strip():
        # Sanitize preset name: remove special chars, replace spaces with underscore
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in preset_name.strip())
        sanitized = sanitized.strip("_")
        if sanitized:
            return f"live_{sanitized}_{timestamp}"
    
    return f"live_{timestamp}"


def _save_preset(
    preset_dir: Path,
    preset_name: str,
    selected_factors: List[str],
    max_pos: float,
    min_pos: float,
    topn_base: int,
    topn_vol: int,
    softmax_alpha: float,
    w_smooth: float,
    pins: str,
    blacklist: str,
) -> bool:
    """Save preset to JSON file. Returns True on success."""
    if not preset_name or not preset_name.strip():
        return False
    
    preset_data = {
        "factors": selected_factors,
        "max_pos": float(max_pos),
        "min_pos": float(min_pos),
        "topn_base": int(topn_base),
        "topn_vol": int(topn_vol),
        "softmax_alpha": float(softmax_alpha),
        "w_smooth": float(w_smooth),
        "pins": pins,
        "blacklist": blacklist,
    }
    
    preset_path = preset_dir / f"preset_{preset_name.strip()}.json"
    preset_path.write_text(json.dumps(preset_data, indent=2))
    return True


def _load_preset(preset_dir: Path, preset_name: str) -> Optional[Dict]:
    """Load preset from JSON file. Returns preset data or None."""
    if not preset_name:
        return None
    
    preset_path = preset_dir / f"{preset_name}.json"
    if not preset_path.exists():
        return None
    
    return json.loads(preset_path.read_text())


def _run_live_strategy(
    selected_factors: List[str],
    max_pos: float,
    min_pos: float,
    topn_base: int,
    topn_vol: int,
    softmax_alpha: float,
    w_smooth: float,
    pins: str,
    blacklist: str,
    preset_name: str,
) -> str:
    """Execute the live strategy run. Returns the tag used."""
    pin_list = [x.strip() for x in pins.split(",") if x.strip()]
    blk_list = [x.strip() for x in blacklist.split(",") if x.strip()]

    fparams = FactorParams(eps=float(cfg.strategy.EPS), robust_cs_z=True)
    iparams = ICWeightingParams(lam=0.95, eps=float(cfg.strategy.EPS))
    pparams = PortfolioParams(
        max_pos=float(max_pos),
        min_pos=float(min_pos),
        eps=float(cfg.strategy.EPS),
        topn_base=int(topn_base),
        topn_volatile=int(topn_vol),
        score_smooth_win=int(cfg.strategy.SCORE_SMOOTH_WIN),
        weight_smooth_alpha=float(w_smooth),
        softmax_tilt_alpha=float(softmax_alpha),
        use_equal_topk=bool(cfg.strategy.USE_EQUAL_TOPK),
        long_seats=int(cfg.strategy.LONG_SEATS),
        short_seats=int(cfg.strategy.SHORT_SEATS),
        fixed_long_frac=float(cfg.strategy.FIXED_LONG_FRAC),
        side_split_mode=str(cfg.strategy.SIDE_SPLIT_MODE.value),
        budget_mode=str(cfg.strategy.PM_TARGET_MODE.value),
        pm_gross_target=float(cfg.strategy.PM_GROSS_TARGET),
        target_vol_base=float(cfg.strategy.TARGET_VOL_BASE),
        target_vol_volatile=float(cfg.strategy.TARGET_VOL_VOLATILE),
        lev_cap=float(cfg.strategy.LEV_CAP),
        lev_min=float(cfg.strategy.LEV_MIN),
    )

    # Generate tag with preset name if provided
    tag_live = _generate_live_tag(preset_name)

    eng = StrategyEngine(
        factor_params=fparams,
        ic_params=iparams,
        portfolio_params=pparams,
        factors=selected_factors,
    )

    eng.run(pinned=pin_list, exchanges=None, tag=tag_live)
    
    return tag_live


def render_live_strategy_page(preset_dir: Path) -> None:
    """
    Render the Live Strategy Re-Run page.
    
    Args:
        preset_dir: Directory for saving/loading presets
    """
    st.subheader("Live Strategy Re-Run")
    
    st.markdown(
        """
        **Re-run the full strategy backtest** with modified parameters. 
        This creates new output files with all weights, factors, and diagnostics recalculated.
        
        - **Configure factors**: Select which factors to include in the composite score
        - **Tune parameters**: Adjust position sizing, concentration, and smoothing
        - **Save presets**: Store configurations for reproducible runs
        - **Identify runs**: Tags include preset names for easy matching in Run Selection
        
        > **Note**: For quick portfolio adjustments without re-running the strategy, 
        > use the **What-If** page instead.
        """
    )
    
    st.divider()
    
    # Initialize session state
    _init_session_state()
    
    # =========================================================================
    # PRESET MANAGEMENT
    # =========================================================================
    st.markdown("### 💾 Preset Management")
    
    col_save, col_load = st.columns(2)
    
    with col_save:
        st.markdown("**Save Current Configuration**")
        preset_name_input = st.text_input(
            "Preset Name",
            value="",
            key="live_preset_name_input",
            help="Name for saving the current configuration. This will also tag the run output."
        )
        save_btn = st.button("💾 Save Preset", key="live_save_preset_btn")
    
    with col_load:
        st.markdown("**Load Existing Preset**")
        preset_files = sorted(
            [f.stem for f in preset_dir.glob("preset_*.json")], reverse=True
        )
        if preset_files:
            selected_preset = st.selectbox(
                "Select Preset",
                options=[""] + preset_files,
                key="live_load_preset_select"
            )
            load_btn = st.button("📂 Load Preset", key="live_load_preset_btn")
        else:
            st.caption("No presets saved yet")
            selected_preset = ""
            load_btn = False
    
    # Handle preset save
    if save_btn and preset_name_input:
        success = _save_preset(
            preset_dir=preset_dir,
            preset_name=preset_name_input,
            selected_factors=st.session_state.live_selected_factors,
            max_pos=st.session_state.live_pm_max_pos,
            min_pos=st.session_state.live_pm_min_pos,
            topn_base=st.session_state.live_pm_topn_base,
            topn_vol=st.session_state.live_pm_topn_vol,
            softmax_alpha=st.session_state.live_pm_softmax_alpha,
            w_smooth=st.session_state.live_pm_w_smooth,
            pins=st.session_state.live_pm_pins,
            blacklist=st.session_state.live_pm_blacklist,
        )
        if success:
            st.success(f"Preset '{preset_name_input}' saved!")
            st.rerun()
        else:
            st.error("Failed to save preset. Please provide a valid name.")
    
    # Handle preset load
    if load_btn and selected_preset:
        preset_data = _load_preset(preset_dir, selected_preset)
        if preset_data:
            st.session_state.live_selected_factors = preset_data.get("factors", ALL_FACTORS)
            st.session_state.live_pm_max_pos = preset_data.get("max_pos", float(cfg.strategy.MAX_POS))
            st.session_state.live_pm_min_pos = preset_data.get("min_pos", float(cfg.strategy.MIN_POS))
            st.session_state.live_pm_topn_base = preset_data.get("topn_base", int(cfg.strategy.TOPN_BASE))
            st.session_state.live_pm_topn_vol = preset_data.get("topn_vol", int(cfg.strategy.TOPN_VOLATILE))
            st.session_state.live_pm_softmax_alpha = preset_data.get("softmax_alpha", float(cfg.strategy.SOFTMAX_TILT_ALPHA))
            st.session_state.live_pm_w_smooth = preset_data.get("w_smooth", float(cfg.strategy.WEIGHT_SMOOTH_ALPHA))
            st.session_state.live_pm_pins = preset_data.get("pins", "")
            st.session_state.live_pm_blacklist = preset_data.get("blacklist", "")
            st.success(f"Preset '{selected_preset}' loaded!")
            st.rerun()
        else:
            st.error(f"Could not load preset '{selected_preset}'")
    
    st.divider()
    
    # =========================================================================
    # FACTOR SELECTION
    # =========================================================================
    st.markdown("### 🎯 Factor Selection")
    
    # Quick select buttons
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        if st.button("All Factors", key="live_select_all"):
            st.session_state.live_selected_factors = ALL_FACTORS
            st.rerun()
    with col_q2:
        if st.button("Defensive Only", key="live_select_defensive"):
            st.session_state.live_selected_factors = DEFENSIVE_FACTORS
            st.rerun()
    with col_q3:
        if st.button("Alpha Only", key="live_select_alpha"):
            st.session_state.live_selected_factors = ALPHA_FACTORS
            st.rerun()
    with col_q4:
        if st.button("Reset to Defaults", key="live_reset_factors"):
            st.session_state.live_selected_factors = ALL_FACTORS
            st.session_state.live_pm_max_pos = float(cfg.strategy.MAX_POS)
            st.session_state.live_pm_min_pos = float(cfg.strategy.MIN_POS)
            st.session_state.live_pm_topn_base = int(cfg.strategy.TOPN_BASE)
            st.session_state.live_pm_topn_vol = int(cfg.strategy.TOPN_VOLATILE)
            st.session_state.live_pm_softmax_alpha = float(cfg.strategy.SOFTMAX_TILT_ALPHA)
            st.session_state.live_pm_w_smooth = float(cfg.strategy.WEIGHT_SMOOTH_ALPHA)
            st.rerun()
    
    # Factor category expanders
    with st.expander("🛡️ Defensive / Low Risk Factors", expanded=False):
        defensive_selected = st.multiselect(
            "Defensive Factors",
            options=DEFENSIVE_FACTORS,
            default=[f for f in DEFENSIVE_FACTORS if f in st.session_state.live_selected_factors],
            key="live_defensive_ms",
        )
    
    with st.expander("📈 Alpha Factors", expanded=False):
        alpha_selected = st.multiselect(
            "Alpha Factors",
            options=ALPHA_FACTORS,
            default=[f for f in ALPHA_FACTORS if f in st.session_state.live_selected_factors],
            key="live_alpha_ms",
        )
    
    with st.expander("✨ Quality Factors", expanded=False):
        quality_selected = st.multiselect(
            "Quality Factors",
            options=QUALITY_FACTORS,
            default=[f for f in QUALITY_FACTORS if f in st.session_state.live_selected_factors],
            key="live_quality_ms",
        )
    
    with st.expander("🔗 Interaction Factors", expanded=False):
        interaction_selected = st.multiselect(
            "Interaction Factors",
            options=INTERACTION_FACTORS,
            default=[f for f in INTERACTION_FACTORS if f in st.session_state.live_selected_factors],
            key="live_interaction_ms",
        )
    
    # Combine selected factors
    selected_factors = defensive_selected + alpha_selected + quality_selected + interaction_selected
    
    if not selected_factors:
        st.warning("⚠️ No factors selected! Strategy will use all factors as fallback.")
        selected_factors = ALL_FACTORS
    
    st.session_state.live_selected_factors = selected_factors
    st.info(f"📊 **Selected: {len(selected_factors)}/{len(ALL_FACTORS)} factors**")
    
    st.divider()
    
    # =========================================================================
    # PORTFOLIO PARAMETERS
    # =========================================================================
    st.markdown("### ⚙️ Portfolio Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_pos = st.slider(
            "Max Position Weight",
            min_value=0.01,
            max_value=0.25,
            value=st.session_state.live_pm_max_pos,
            step=0.005,
            key="live_slider_max_pos",
            help="Maximum weight per position"
        )
        
        topn_base = st.slider(
            "Top N (Base Regime)",
            min_value=5,
            max_value=80,
            value=st.session_state.live_pm_topn_base,
            step=1,
            key="live_slider_topn_base",
            help="Number of positions in base (normal volatility) regime"
        )
        
        softmax_alpha = st.slider(
            "Softmax Tilt Alpha",
            min_value=0.0,
            max_value=3.0,
            value=st.session_state.live_pm_softmax_alpha,
            step=0.05,
            key="live_slider_softmax",
            help="Score concentration parameter (higher = more concentrated)"
        )
    
    with col2:
        min_pos = st.slider(
            "Min Position Weight",
            min_value=0.0,
            max_value=0.02,
            value=st.session_state.live_pm_min_pos,
            step=0.001,
            key="live_slider_min_pos",
            help="Minimum weight per position"
        )
        
        topn_vol = st.slider(
            "Top N (Volatile Regime)",
            min_value=5,
            max_value=80,
            value=st.session_state.live_pm_topn_vol,
            step=1,
            key="live_slider_topn_vol",
            help="Number of positions in volatile regime"
        )
        
        w_smooth = st.slider(
            "Weight Smoothing Alpha",
            min_value=0.0,
            max_value=0.9,
            value=st.session_state.live_pm_w_smooth,
            step=0.02,
            key="live_slider_w_smooth",
            help="Weight smoothing (0 = no smoothing, higher = more smoothing)"
        )
    
    # Update session state
    st.session_state.live_pm_max_pos = max_pos
    st.session_state.live_pm_min_pos = min_pos
    st.session_state.live_pm_topn_base = topn_base
    st.session_state.live_pm_topn_vol = topn_vol
    st.session_state.live_pm_softmax_alpha = softmax_alpha
    st.session_state.live_pm_w_smooth = w_smooth
    
    st.divider()
    
    # =========================================================================
    # POSITION CONSTRAINTS
    # =========================================================================
    st.markdown("### 📌 Position Constraints")
    
    col_pins, col_blacklist = st.columns(2)
    
    with col_pins:
        pins = st.text_input(
            "Pinned Positions (comma-separated)",
            value=st.session_state.live_pm_pins,
            key="live_input_pins",
            help="Symbols to always include in the portfolio"
        )
        st.session_state.live_pm_pins = pins
    
    with col_blacklist:
        blacklist = st.text_input(
            "Blacklisted Positions (comma-separated)",
            value=st.session_state.live_pm_blacklist,
            key="live_input_blacklist",
            help="Symbols to exclude from the portfolio"
        )
        st.session_state.live_pm_blacklist = blacklist
    
    st.divider()
    
    # =========================================================================
    # RUN CONTROLS
    # =========================================================================
    st.markdown("### 🚀 Run Strategy")
    
    run_preset_name = st.text_input(
        "Run Tag Name (optional)",
        value=preset_name_input if preset_name_input else "",
        key="live_run_tag_name",
        help="Name to include in the output tag. Leave empty for timestamp-only tags."
    )
    
    st.markdown(
        f"""
        **Output Tag Preview:** `{_generate_live_tag(run_preset_name)}`
        
        This tag will appear in the Run Selection dropdown, allowing you to easily 
        identify this run.
        """
    )
    
    col_run, col_info = st.columns([1, 2])
    
    with col_run:
        run_live = st.button(
            "▶️ Run Live Strategy",
            type="primary",
            key="live_run_btn",
            help="Execute the full strategy backtest with current parameters"
        )
    
    with col_info:
        st.caption(
            "⚠️ This will run the **full strategy backtest** which may take several minutes. "
            "All factors will be recalculated and new output files will be generated."
        )
    
    if run_live:
        with st.spinner("🔄 Running strategy with new parameters... This may take several minutes."):
            try:
                tag_used = _run_live_strategy(
                    selected_factors=selected_factors,
                    max_pos=max_pos,
                    min_pos=min_pos,
                    topn_base=topn_base,
                    topn_vol=topn_vol,
                    softmax_alpha=softmax_alpha,
                    w_smooth=w_smooth,
                    pins=pins,
                    blacklist=blacklist,
                    preset_name=run_preset_name,
                )
                st.success(f"✅ Live run complete! Tag: `{tag_used}`")
                st.info("📍 Switch to a different page and back, or refresh the Run Selection dropdown to see the new tag.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Strategy run failed: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # =========================================================================
    # CURRENT CONFIGURATION SUMMARY
    # =========================================================================
    with st.expander("📋 Current Configuration Summary", expanded=False):
        config_summary = {
            "Factors": f"{len(selected_factors)} selected",
            "Factor List": ", ".join(selected_factors[:5]) + ("..." if len(selected_factors) > 5 else ""),
            "Max Position": f"{max_pos:.2%}",
            "Min Position": f"{min_pos:.3%}",
            "Top N (Base)": topn_base,
            "Top N (Volatile)": topn_vol,
            "Softmax Alpha": softmax_alpha,
            "Weight Smoothing": w_smooth,
            "Pinned": pins if pins else "(none)",
            "Blacklisted": blacklist if blacklist else "(none)",
        }
        
        for key, value in config_summary.items():
            st.text(f"{key}: {value}")
