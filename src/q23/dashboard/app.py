from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime
from q23.strategy.engine import StrategyEngine
from q23.strategy.factors import FactorParams, V4_24_FACTORS
from q23.strategy.ic_weighting import ICWeightingParams
from q23.strategy.portfolio import PortfolioParams


import numpy as np
import pandas as pd
import streamlit as st

from q23.shared.config import cfg

TAG_RE = re.compile(r"_wide_weights_(.+?)\.csv$")  # captures {tag} in {base}_wide_weights_{tag}.csv


# -----------------------------
# Paths + discovery
# -----------------------------

def _project_root() -> Path:
    # .../Q23_QUANT_SYSTEM/src/q23/dashboard/app.py -> parents[3] = Q23_QUANT_SYSTEM
    return Path(__file__).resolve().parents[3]


def _base_output_dir() -> Path:
    """
    Output root is cfg.paths.OUTPUT_ROOT.
    We normalize relative paths against project root.
    """
    root = _project_root()
    p = Path(cfg.paths.OUTPUT_ROOT)
    if not p.is_absolute():
        p = root / p
    return p


def _discover_tags(base_dir: Path, base_name: str) -> List[str]:
    """
    Scan outputs/<BASE_NAME>/ for *_wide_weights_*.csv and extract tags.
    """
    if not base_dir.exists():
        return []

    tags: List[str] = []
    pat = f"{base_name}_wide_weights_*.csv"
    for f in sorted(base_dir.glob(pat)):
        m = TAG_RE.search(f.name)
        if m:
            tags.append(m.group(1))
    # newest-first if tags look like dates
    def _key(t: str):
        try:
            return pd.to_datetime(t)
        except Exception:
            return pd.Timestamp.min
    tags_sorted = sorted(tags, key=_key, reverse=True)
    return tags_sorted


def _artifact_paths(base_dir: Path, base_name: str, tag: str) -> Dict[str, Path]:
    """
    Map canonical tag-based artifact filenames to full paths.
    """
    return {
        "wide_weights": base_dir / f"{base_name}_wide_weights_{tag}.csv",
        "budget": base_dir / f"{base_name}_budget_{tag}.csv",
        "factor_weights": base_dir / f"{base_name}_factor_weights_{tag}.csv",
        "ic": base_dir / f"{base_name}_ic_{tag}.csv",
        "meta": base_dir / f"{base_name}_meta_{tag}.json",
        # enhanced artifacts:
        "portfolio_diag": base_dir / f"{base_name}_portfolio_diag_{tag}.csv",
        "factor_exposure": base_dir / f"{base_name}_factor_exposure_{tag}.csv",
        "factor_vectors": base_dir / f"{base_name}_factor_vectors_{tag}.csv",
    }


# -----------------------------
# Loading helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def _coerce_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect first column is 'time' or date-like.
    """
    d = df.copy()
    if len(d.columns) == 0:
        return d

    # typical writers: index saved, first column name is 'time'
    if d.columns[0].lower() in ("time", "date", "datetime"):
        c = d.columns[0]
        d[c] = pd.to_datetime(d[c], errors="coerce")
        d = d.set_index(c)
    else:
        # attempt parse anyway
        d.iloc[:, 0] = pd.to_datetime(d.iloc[:, 0], errors="coerce")
        if d.iloc[:, 0].notna().mean() > 0.7:
            d = d.set_index(d.columns[0])

    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    return d


def _sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        try:
            d[c] = pd.to_numeric(d[c])
        except Exception:
            # If conversion fails, leave column as-is
            continue
    return d.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _load_bundle(paths: Dict[str, Path]) -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {k: None for k in paths.keys()}

    # wide weights
    if paths["wide_weights"].exists():
        w = _read_csv(str(paths["wide_weights"]))
        w = _coerce_time_index(w)
        out["wide_weights"] = _sanitize_numeric(w)

    # budget
    if paths["budget"].exists():
        b = _read_csv(str(paths["budget"]))
        b = _coerce_time_index(b)
        out["budget"] = _sanitize_numeric(b)

    # factor weights (time x factor)
    if paths["factor_weights"].exists():
        fw = _read_csv(str(paths["factor_weights"]))
        fw = _coerce_time_index(fw)
        out["factor_weights"] = _sanitize_numeric(fw)

    # IC (MultiIndex columns when read by pandas)
    if paths["ic"].exists():
        ic = pd.read_csv(str(paths["ic"]), header=[0, 1])
        # first column should still be time
        if ic.columns[0][0].lower() in ("time", "date", "datetime"):
            ic[(ic.columns[0])] = pd.to_datetime(ic[(ic.columns[0])], errors="coerce")
            ic = ic.set_index(ic.columns[0])
        else:
            # fallback: parse first col
            ic.iloc[:, 0] = pd.to_datetime(ic.iloc[:, 0], errors="coerce")
            ic = ic.set_index(ic.columns[0])
        ic.index = pd.to_datetime(ic.index, errors="coerce")
        ic = ic[~ic.index.isna()].sort_index()
        ic = ic.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out["ic"] = ic

    # meta json
    if paths["meta"].exists():
        out["meta"] = _read_json(str(paths["meta"]))

    # enhanced artifacts
    if paths["portfolio_diag"].exists():
        d = _read_csv(str(paths["portfolio_diag"]))
        d = _coerce_time_index(d)
        out["portfolio_diag"] = _sanitize_numeric(d)

    if paths["factor_exposure"].exists():
        e = _read_csv(str(paths["factor_exposure"]))
        e = _coerce_time_index(e)
        out["factor_exposure"] = _sanitize_numeric(e)

    if paths["factor_vectors"].exists():
        fv = pd.read_csv(str(paths["factor_vectors"]), index_col=0)
        fv.index = fv.index.astype(str)
        out["factor_vectors"] = fv.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


# -----------------------------
# PM analytics helpers
# -----------------------------

def _summary_exposure(w_last: pd.Series) -> Dict[str, float]:
    gross = float(np.abs(w_last).sum())
    net = float(w_last.sum())
    long = float(w_last[w_last > 0].sum())
    short = float(w_last[w_last < 0].sum())
    return {
        "gross": gross,
        "net": net,
        "long": long,
        "short": short,
        "n_pos": int((w_last.abs() > 1e-12).sum()),
    }


def _top_positions(w_last: pd.Series, n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    longs = w_last[w_last > 0].sort_values(ascending=False).head(n)
    shorts = w_last[w_last < 0].sort_values(ascending=True).head(n)
    return longs.rename("weight").to_frame(), shorts.rename("weight").to_frame()


def _turnover_series(weights: pd.DataFrame) -> pd.Series:
    d = weights.fillna(0.0).diff().abs().sum(axis=1)
    if len(d) > 0:
        d.iloc[0] = 0.0
    d.name = "turnover"
    return d


def _select_return_series(diag: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Pick the best available portfolio return column from portfolio_diag.
    Preference order: net-of-TC, then gross, then active.
    """
    if diag is None or diag.empty:
        return None

    candidates = ["port_ret_net_tc", "port_ret", "active_ret_net_tc", "active_ret"]
    for c in candidates:
        if c in diag.columns:
            s = diag[c].astype(float).copy()
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[~s.index.isna()].sort_index()
            return s
    return None


def _period_return(s: pd.Series, start: pd.Timestamp) -> float:
    """Cumulative return from `start` (inclusive) to the end of the series."""
    s2 = s[s.index >= start]
    if s2.empty:
        return float("nan")
    cum = (1.0 + s2).cumprod() - 1.0
    return float(cum.iloc[-1])


def _period_windows(diag: Optional[pd.DataFrame], last_dt: pd.Timestamp) -> Dict[str, float]:
    """
    Compute WTD / MTD / YTD cumulative returns using the best available
    portfolio return series from portfolio_diag.
    """
    out: Dict[str, float] = {}
    s = _select_return_series(diag)
    if s is None or s.empty:
        return out

    last_dt = pd.Timestamp(last_dt)
    w_start = last_dt - pd.Timedelta(days=7)
    m_start = last_dt - pd.DateOffset(months=1)
    y_start = last_dt - pd.DateOffset(years=1)

    out["wtd"] = _period_return(s, w_start)
    out["mtd"] = _period_return(s, m_start)
    out["ytd"] = _period_return(s, y_start)
    return out


def _drawdown(ret_series: pd.Series) -> pd.Series:
    """Compute drawdown from a return series."""
    if ret_series is None or ret_series.empty:
        return pd.Series(dtype=float)
    perf = (1.0 + ret_series).cumprod()
    peak = perf.cummax()
    dd = perf / peak - 1.0
    dd.name = "drawdown"
    return dd


def _compute_trade_list(w: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, min_abs_delta: float = 0.002) -> pd.DataFrame:
    w0 = w.loc[t0].fillna(0.0)
    w1 = w.loc[t1].fillna(0.0)
    delta = (w0 - w1).astype(float)
    trades = pd.DataFrame({"prev_w": w1, "target_w": w0, "delta_w": delta})
    trades = trades[np.abs(trades["delta_w"]) >= float(min_abs_delta)].copy()
    trades["action"] = np.where(trades["delta_w"] > 0, "BUY", "SELL")
    trades["abs_delta"] = trades["delta_w"].abs()
    trades = trades.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])
    trades.index.name = "symbol"
    return trades


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Q23 PM Dashboard", layout="wide")
st.title("Q23 PM Dashboard — Elite Console (v4 modular + enhanced artifacts)")

proj = _project_root()
base_dir = _base_output_dir()
base_name = cfg.paths.BASE_NAME

with st.sidebar:
    st.header("Run selection (tags)")

    st.caption(f"Project: `{proj}`")
    st.caption(f"Outputs: `{base_dir}`")
    st.caption(f"BASE_NAME: `{base_name}`")

    tags = _discover_tags(base_dir, base_name)
    if not tags:
        st.error(
            "No tags found.\n\n"
            f"Expected files like `{base_name}_wide_weights_<tag>.csv` under:\n"
            f"`{base_dir}`\n\n"
            "Run strategies first."
        )
        st.stop()

    tag = st.selectbox("Tag", options=tags, index=0)
    paths = _artifact_paths(base_dir, base_name, tag)

    with st.expander("Diagnostics / Artifacts", expanded=False):
        st.caption("Underlying files for this tag (for debugging only).")
        for k, p in paths.items():
            st.write(f"- **{k}**: {'✅' if p.exists() else '—'}  `{p.name}`")

    st.divider()
    page = st.radio("Page", ["Overview", "Performance", "Weights", "Factors/IC", "Blotter", "Rebalance"], index=0)

    st.divider()
    st.header("PM Controls (live)")

    live_mode = st.toggle("Enable live re-run", value=False)

    # Preset management
    st.subheader("Presets")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        preset_name = st.text_input("Preset name", value="", key="preset_name")
        save_preset = st.button("💾 Save Preset", disabled=not live_mode)
    with preset_col2:
        preset_files = sorted([f.stem for f in _base_output_dir().glob("preset_*.json")], reverse=True)
        if preset_files:
            selected_preset = st.selectbox("Load preset", options=[""] + preset_files, key="load_preset")
            load_preset = st.button("📂 Load Preset", disabled=not live_mode or not selected_preset)
        else:
            selected_preset = ""
            load_preset = False
            st.caption("No presets found")

    # Factor selection
    st.subheader("Factor Selection")
    all_factors = list(V4_24_FACTORS)
    # Group factors for better UX
    defensive_factors = ["inv_vol", "inv_idio", "inv_down", "low_corr", "low_beta", "liquidity", "amihud_inv"]
    alpha_factors = ["resid_mom", "srev", "breakout", "slope", "calm_flow", "prox_52w_high", "resid_mom_mix", 
                     "vol_surprise", "vol_breakout", "ma_cloud"]
    quality_factors = ["idio_tail_risk", "idio_jump_freq", "beta_stability", "micro_noise"]
    interaction_factors = ["value_mom", "quality_defensive", "rel_sector_mom"]
    
    # Initialize session state for factors if not exists
    if "selected_factors" not in st.session_state:
        st.session_state.selected_factors = all_factors
    
    # Quick selectors
    col_quick1, col_quick2, col_quick3 = st.columns(3)
    with col_quick1:
        if st.button("Select All", key="select_all_factors"):
            st.session_state.selected_factors = all_factors
            st.rerun()
    with col_quick2:
        if st.button("Defensive Only", key="defensive_only"):
            st.session_state.selected_factors = defensive_factors
            st.rerun()
    with col_quick3:
        if st.button("Alpha Only", key="alpha_only"):
            st.session_state.selected_factors = alpha_factors
            st.rerun()
    
    # Factor checkboxes (grouped)
    with st.expander("Defensive / Low Risk", expanded=True):
        defensive_selected = st.multiselect(
            "Defensive factors",
            options=defensive_factors,
            default=[f for f in defensive_factors if f in st.session_state.selected_factors],
            key="defensive_ms"
        )
    
    with st.expander("Alpha", expanded=True):
        alpha_selected = st.multiselect(
            "Alpha factors",
            options=alpha_factors,
            default=[f for f in alpha_factors if f in st.session_state.selected_factors],
            key="alpha_ms"
        )
    
    with st.expander("Quality / Microstructure"):
        quality_selected = st.multiselect(
            "Quality factors",
            options=quality_factors,
            default=[f for f in quality_factors if f in st.session_state.selected_factors],
            key="quality_ms"
        )
    
    with st.expander("Interactions"):
        interaction_selected = st.multiselect(
            "Interaction factors",
            options=interaction_factors,
            default=[f for f in interaction_factors if f in st.session_state.selected_factors],
            key="interaction_ms"
        )
    
    # Combine all selected factors and update session state
    selected_factors = defensive_selected + alpha_selected + quality_selected + interaction_selected
    if not selected_factors:
        st.warning("⚠️ No factors selected! At least one factor is required.")
        selected_factors = all_factors  # fallback to all
    st.session_state.selected_factors = selected_factors  # Update session state
    st.caption(f"Selected: {len(selected_factors)}/{len(all_factors)} factors")

    # Initialize session state for parameters if not exists
    if "pm_max_pos" not in st.session_state:
        st.session_state.pm_max_pos = float(cfg.strategy.MAX_POS)
        st.session_state.pm_min_pos = float(cfg.strategy.MIN_POS)
        st.session_state.pm_topn_base = int(cfg.strategy.TOPN_BASE)
        st.session_state.pm_topn_vol = int(cfg.strategy.TOPN_VOLATILE)
        st.session_state.pm_softmax_alpha = float(cfg.strategy.SOFTMAX_TILT_ALPHA)
        st.session_state.pm_w_smooth = float(cfg.strategy.WEIGHT_SMOOTH_ALPHA)
        st.session_state.pm_pins = ""
        st.session_state.pm_blacklist = ""

    # core knobs
    st.subheader("Portfolio Parameters")
    max_pos = st.slider("MAX_POS", 0.01, 0.25, st.session_state.pm_max_pos, 0.005, key="slider_max_pos")
    min_pos = st.slider("MIN_POS", 0.0, 0.02, st.session_state.pm_min_pos, 0.001, key="slider_min_pos")
    topn_base = st.slider("TOPN_BASE", 5, 80, st.session_state.pm_topn_base, 1, key="slider_topn_base")
    topn_vol = st.slider("TOPN_VOLATILE", 5, 80, st.session_state.pm_topn_vol, 1, key="slider_topn_vol")
    softmax_alpha = st.slider("SOFTMAX_TILT_ALPHA", 0.0, 3.0, st.session_state.pm_softmax_alpha, 0.05, key="slider_softmax")
    w_smooth = st.slider("WEIGHT_SMOOTH_ALPHA", 0.0, 0.9, st.session_state.pm_w_smooth, 0.02, key="slider_w_smooth")

    st.caption("Pins/Blacklist are passed through as ids (comma-separated).")
    pins = st.text_input("PM_PIN_IDS", value=st.session_state.pm_pins, key="input_pins")
    blacklist = st.text_input("PM_BLACKLIST_IDS", value=st.session_state.pm_blacklist, key="input_blacklist")

    run_live = st.button("▶ Run Live (writes new tag)", disabled=not live_mode)

bundle = _load_bundle(paths)

# Preset save/load logic
preset_dir = _base_output_dir()

if save_preset and preset_name:
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
    preset_path = preset_dir / f"preset_{preset_name}.json"
    preset_path.write_text(json.dumps(preset_data, indent=2))
    st.success(f"Preset '{preset_name}' saved!")
    st.rerun()

if load_preset and selected_preset:
    preset_path = preset_dir / f"{selected_preset}.json"
    if preset_path.exists():
        preset_data = json.loads(preset_path.read_text())
        st.session_state.selected_factors = preset_data.get("factors", all_factors)
        # Update parameter values in session state
        st.session_state.pm_max_pos = preset_data.get("max_pos", float(cfg.strategy.MAX_POS))
        st.session_state.pm_min_pos = preset_data.get("min_pos", float(cfg.strategy.MIN_POS))
        st.session_state.pm_topn_base = preset_data.get("topn_base", int(cfg.strategy.TOPN_BASE))
        st.session_state.pm_topn_vol = preset_data.get("topn_vol", int(cfg.strategy.TOPN_VOLATILE))
        st.session_state.pm_softmax_alpha = preset_data.get("softmax_alpha", float(cfg.strategy.SOFTMAX_TILT_ALPHA))
        st.session_state.pm_w_smooth = preset_data.get("w_smooth", float(cfg.strategy.WEIGHT_SMOOTH_ALPHA))
        st.session_state.pm_pins = preset_data.get("pins", "")
        st.session_state.pm_blacklist = preset_data.get("blacklist", "")
        st.success(f"Preset '{selected_preset}' loaded!")
        st.rerun()

# Execute live run if requested
if run_live:
    pin_list = [x.strip() for x in pins.split(",") if x.strip()]
    blk_list = [x.strip() for x in blacklist.split(",") if x.strip()]

    # Build param overrides
    fparams = FactorParams(
        eps=float(cfg.strategy.EPS),
        robust_cs_z=True,
    )

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

    tag_live = datetime.now().strftime("live_%Y%m%d_%H%M%S")

    eng = StrategyEngine(
        factor_params=fparams,
        ic_params=iparams,
        portfolio_params=pparams,
        factors=selected_factors,  # Pass selected factors
    )

    # IMPORTANT: blacklist affects universe at load stage (data_loader). If your loader supports it,
    # pass blacklist in pinned/exchanges args or extend loader to accept it.
    eng.run(pinned=pin_list, exchanges=None, tag=tag_live)

    st.success(f"Live run complete → tag: {tag_live}")
    st.rerun()

weights: Optional[pd.DataFrame] = bundle.get("wide_weights")  # type: ignore
budget: Optional[pd.DataFrame] = bundle.get("budget")         # type: ignore
fw: Optional[pd.DataFrame] = bundle.get("factor_weights")     # type: ignore
ic: Optional[pd.DataFrame] = bundle.get("ic")                 # type: ignore
meta: Optional[Dict] = bundle.get("meta")                     # type: ignore

diag: Optional[pd.DataFrame] = bundle.get("portfolio_diag")   # type: ignore
exp: Optional[pd.DataFrame] = bundle.get("factor_exposure")   # type: ignore
fv: Optional[pd.DataFrame] = bundle.get("factor_vectors")     # type: ignore

if weights is None or weights.empty:
    st.error("Could not load wide weights for this tag.")
    st.stop()

# shared basics
last_dt = weights.index.max()
w_last = weights.loc[last_dt].astype(float)
expos = _summary_exposure(w_last)


# -----------------------------
# Overview
# -----------------------------

if page == "Overview":
    st.subheader("Overview")

    # First row: exposure snapshot
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("As of", str(last_dt.date()))
    c2.metric("Gross", f"{expos['gross']:.2f}")
    c3.metric("Net", f"{expos['net']:.2f}")
    c4.metric("Long", f"{expos['long']:.2f}")
    c5.metric("Short", f"{expos['short']:.2f}")

    # Second row: performance tiles (WTD / MTD / YTD)
    perf_windows = _period_windows(diag, last_dt)
    if perf_windows:
        c6, c7, c8 = st.columns(3)
        wtd = perf_windows.get("wtd", float("nan"))
        mtd = perf_windows.get("mtd", float("nan"))
        ytd = perf_windows.get("ytd", float("nan"))
        if np.isfinite(wtd):
            c6.metric("WTD", f"{wtd:.2%}")
        if np.isfinite(mtd):
            c7.metric("MTD", f"{mtd:.2%}")
        if np.isfinite(ytd):
            c8.metric("YTD", f"{ytd:.2%}")

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Key time-series")
        # Prefer portfolio_diag if present (richer)
        if diag is not None and not diag.empty:
            cols = [c for c in ["gross_exposure", "net_exposure", "turnover", "rolling_vol"] if c in diag.columns]
            if cols:
                st.line_chart(diag[cols], height=260)
            cols2 = [c for c in ["port_ret", "port_ret_net_tc", "active_ret", "active_ret_net_tc"] if c in diag.columns]
            if cols2:
                st.line_chart(diag[cols2], height=260)
            # Period performance focus: WTD / MTD / YTD cumulative curves
            ret_series = _select_return_series(diag)
            if ret_series is not None and not ret_series.empty:
                st.markdown("### Period performance (WTD / MTD / YTD)")
                tabs = st.tabs(["WTD", "MTD", "YTD"])
                last_dt_norm = pd.Timestamp(last_dt)
                w_start = last_dt_norm - pd.Timedelta(days=7)
                m_start = last_dt_norm - pd.DateOffset(months=1)
                y_start = last_dt_norm - pd.DateOffset(years=1)

                perf_full = (1.0 + ret_series).cumprod() - 1.0

                with tabs[0]:
                    s = perf_full[perf_full.index >= w_start]
                    if not s.empty:
                        st.area_chart(s, height=180)
                with tabs[1]:
                    s = perf_full[perf_full.index >= m_start]
                    if not s.empty:
                        st.area_chart(s, height=180)
                with tabs[2]:
                    s = perf_full[perf_full.index >= y_start]
                    if not s.empty:
                        st.area_chart(s, height=180)
        else:
            # fallback computed quick series
            gross_ts = weights.abs().sum(axis=1)
            net_ts = weights.sum(axis=1)
            turnover = _turnover_series(weights)
            st.line_chart(pd.DataFrame({"gross": gross_ts, "net": net_ts, "turnover": turnover}), height=320)

    with right:
        st.markdown("### Top positions (latest)")
        long_df, short_df = _top_positions(w_last, n=20)
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**Top Longs**")
            st.dataframe(long_df, width="stretch", height=360)
        with rc:
            st.markdown("**Top Shorts**")
            st.dataframe(short_df, width="stretch", height=360)

    st.divider()
    st.markdown("### Metadata")
    if meta:
        st.json(meta, expanded=False)
    else:
        st.info("No meta JSON found for this tag.")

# -----------------------------
# Performance
# -----------------------------

elif page == "Performance":
    st.subheader("Performance")

    ret_series = _select_return_series(diag)
    dd_series = _drawdown(ret_series) if ret_series is not None else None

    colA, colB, colC, colD = st.columns(4)
    if ret_series is not None and not ret_series.empty:
        latest_ret = float(ret_series.iloc[-1])
        colA.metric("Last daily ret", f"{latest_ret:.2%}")
    else:
        colA.metric("Last daily ret", "—")

    if dd_series is not None and not dd_series.empty:
        current_dd = float(dd_series.iloc[-1])
        max_dd = float(dd_series.min())
        colB.metric("Current DD", f"{current_dd:.2%}")
        colC.metric("Max DD", f"{max_dd:.2%}")
    else:
        colB.metric("Current DD", "—")
        colC.metric("Max DD", "—")

    if diag is not None and not diag.empty and "rolling_vol" in diag.columns:
        colD.metric("Rolling vol", f"{float(diag['rolling_vol'].iloc[-1]):.2%}")
    else:
        colD.metric("Rolling vol", "—")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Cumulative return")
        if ret_series is not None and not ret_series.empty:
            perf = (1.0 + ret_series).cumprod() - 1.0
            st.line_chart(perf, height=260)
        else:
            st.info("No return series available (need portfolio_diag).")

        st.markdown("### Drawdown")
        if dd_series is not None and not dd_series.empty:
            st.area_chart(dd_series, height=220)
        else:
            st.info("No drawdown available (need return series).")

    with c2:
        st.markdown("### Turnover distribution (recent)")
        if diag is not None and not diag.empty and "turnover" in diag.columns:
            turn = diag["turnover"].astype(float).tail(180)
        else:
            turn = _turnover_series(weights).tail(180)
        if not turn.empty:
            st.bar_chart(turn, height=260)
        else:
            st.info("No turnover data available.")

        st.markdown("### Seat utilization (latest)")
        long_seats = int(cfg.strategy.LONG_SEATS)
        short_seats = int(cfg.strategy.SHORT_SEATS)
        n_long = int((w_last > 0).sum())
        n_short = int((w_last < 0).sum())
        util_long = n_long / long_seats if long_seats > 0 else float("nan")
        util_short = n_short / short_seats if short_seats > 0 else float("nan")
        u1, u2 = st.columns(2)
        u1.metric("Long seats used", f"{n_long}/{long_seats}" if long_seats > 0 else "—", delta=f"{util_long:.0%}" if np.isfinite(util_long) else None)
        u2.metric("Short seats used", f"{n_short}/{short_seats}" if short_seats > 0 else "—", delta=f"{util_short:.0%}" if np.isfinite(util_short) else None)

# -----------------------------
# Weights
# -----------------------------

elif page == "Weights":
    st.subheader("Weights")

    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        n_days = st.number_input("Show last N days", min_value=10, max_value=4000, value=252, step=10)
    with c2:
        topk = st.number_input("Top K assets (abs, latest)", min_value=10, max_value=300, value=60, step=5)
    with c3:
        st.caption("Tip: keep Top K modest for speed — this is a wide matrix.")

    tail = weights.tail(int(n_days))
    latest = tail.iloc[-1]
    keep_cols = latest.abs().sort_values(ascending=False).head(int(topk)).index
    tail_small = tail[keep_cols]

    st.markdown("### Heatmap table (tail × topK)")
    st.dataframe(tail_small, width="stretch", height=420)

    st.markdown("### Latest snapshot (all assets, sorted)")
    snap = weights.loc[last_dt].sort_values()
    st.dataframe(snap.rename("weight").to_frame(), width="stretch", height=420)

    st.divider()
    st.markdown("### Turnover (computed)")
    st.line_chart(_turnover_series(weights).to_frame(), height=220)

# -----------------------------
# Factors / IC
# -----------------------------

elif page == "Factors/IC":
    st.subheader("Factors / IC")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Factor weights (time-series)")
        if fw is not None and not fw.empty:
            # choose subset for readability
            top = fw.abs().mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
            sel = st.multiselect("Factors", options=list(fw.columns), default=top)
            if sel:
                st.line_chart(fw[sel], height=320)
        else:
            st.info("No factor_weights CSV found for this tag.")

    with c2:
        st.markdown("### Factor IC (raw vs smooth)")
        if ic is not None and not ic.empty:
            # ic has MultiIndex columns: (factor, kind)
            factors = sorted({c[0] for c in ic.columns})
            fsel = st.multiselect("IC factors", options=factors, default=factors[: min(8, len(factors))])
            if fsel:
                view_cols = []
                for f in fsel:
                    if (f, "ic_raw") in ic.columns:
                        view_cols.append((f, "ic_raw"))
                    if (f, "ic_smooth") in ic.columns:
                        view_cols.append((f, "ic_smooth"))
                st.line_chart(ic[view_cols], height=320)
        else:
            st.info("No ic CSV found for this tag.")

    st.divider()
    st.markdown("### Portfolio factor exposure (from F × weights)")
    if exp is not None and not exp.empty:
        top = exp.abs().mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
        sel = st.multiselect("Exposure factors", options=list(exp.columns), default=top, key="exp_sel")
        if sel:
            st.line_chart(exp[sel], height=320)
        st.markdown("**Last day exposures**")
        last = exp.iloc[-1].sort_values(key=lambda s: s.abs(), ascending=False)
        st.dataframe(last.to_frame("exposure"), width="stretch", height=360)
    else:
        st.info("No factor_exposure CSV found for this tag (engine should write it after refactor).")

# -----------------------------
# Blotter (restored “stick tables”)
# -----------------------------

elif page == "Blotter":
    st.subheader("Portfolio Blotter")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("As of", str(last_dt.date()))
    colB.metric("Positions", expos["n_pos"])
    colC.metric("Gross", f"{expos['gross']:.2f}")
    colD.metric("Net", f"{expos['net']:.2f}")

    st.divider()

    blot = pd.DataFrame({"weight": w_last})
    blot = blot[blot["weight"].abs() > 1e-12].copy()
    blot["side"] = np.where(blot["weight"] > 0, "LONG", "SHORT")
    blot["abs_w"] = blot["weight"].abs()

    # Join factor_vectors summaries if present (this is how we restore the old “PM tables”)
    if fv is not None and not fv.empty:
        join_cols = [c for c in [
            "total_pnl_contrib", "pnl_per_day_held", "mean_score", "score_vol",
            "days_held", "avg_weight", "avg_weight_when_held"
        ] if c in fv.columns]
        if join_cols:
            blot = blot.join(fv[join_cols], how="left")

    blot = blot.sort_values(["side", "abs_w"], ascending=[True, False]).drop(columns=["abs_w"])
    st.markdown("### Positions (joined with factor_vectors summaries when available)")
    st.dataframe(blot, width="stretch", height=640)

    st.divider()
    st.markdown("### Factor fingerprint (single name)")
    if fv is None or fv.empty:
        st.info("factor_vectors not found — engine should write it after refactor.")
    else:
        symbols = sorted(set(blot.index).intersection(set(fv.index)))
        sym = st.selectbox("Symbol", options=symbols)
        row = fv.loc[sym].copy()

        # split exposures vs summary fields
        summary_cols = {"total_pnl_contrib", "pnl_per_day_held", "mean_score", "score_vol", "days_held", "avg_weight", "avg_weight_when_held"}
        fac_cols = [c for c in fv.columns if c not in summary_cols]

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Summary**")
            sview = row[[c for c in row.index if c in summary_cols]].to_frame("value")
            st.dataframe(sview, width="stretch", height=320)
        with right:
            st.markdown("**Factor exposures (top 20 by |exposure|)**")
            expos_vec = row[fac_cols].astype(float)
            expos_vec = expos_vec.reindex(expos_vec.abs().sort_values(ascending=False).head(20).index)
            st.dataframe(expos_vec.to_frame("exposure"), width="stretch", height=320)

# -----------------------------
# Rebalance
# -----------------------------

elif page == "Rebalance":
    st.subheader("Rebalance (trade list)")

    dates = list(weights.index)
    if len(dates) < 2:
        st.warning("Need at least 2 dates.")
        st.stop()

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        t0 = st.selectbox("T0 (target)", options=dates, index=len(dates) - 1, format_func=lambda x: x.strftime("%Y-%m-%d"))
    with c2:
        i0 = dates.index(t0)
        t1 = st.selectbox("T-1 (prev)", options=dates, index=max(0, i0 - 1), format_func=lambda x: x.strftime("%Y-%m-%d"))
    with c3:
        min_trade = st.slider("Min |Δw|", 0.0, 0.05, 0.002, step=0.0005)

    if t1 >= t0:
        st.info("Pick T-1 strictly earlier than T0.")
        st.stop()

    trades = _compute_trade_list(weights, pd.Timestamp(t0), pd.Timestamp(t1), min_abs_delta=float(min_trade))
    turnover = float(np.abs(trades["delta_w"]).sum())
    st.metric("Turnover Σ|Δw| (filtered)", f"{turnover:.2%}")

    st.dataframe(trades.reset_index(), width="stretch", height=560)

    csv_bytes = trades.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trades CSV",
        data=csv_bytes,
        file_name=f"{base_name}_rebalance_{pd.Timestamp(t0).strftime('%Y%m%d')}_vs_{pd.Timestamp(t1).strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Q23 Dashboard — tag-based, artifact-driven. Enhanced artifacts restore the PM console tables without rerunning strategy.")
