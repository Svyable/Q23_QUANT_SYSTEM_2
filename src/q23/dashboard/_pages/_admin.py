"""
Admin / Settings Panel

Single-admin MVP tools to make the dashboard feel like a platform:
- Settings / Feature flags (session-level)
- System status
- Ops quick actions (safe)
- Export tools
"""

from __future__ import annotations

import io
import json
import platform
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from importlib import import_module
from typing import Optional

import streamlit as st

from q23.dashboard.core import DashboardData, _project_root, _base_output_dir, _artifact_paths
from q23.dashboard._pages._sidebar import SidebarState
from q23.dashboard.run_cache import get_run_cache
from q23.dashboard.admin_settings import (
    load_admin_settings,
    save_admin_settings,
    save_default_strategy_id,
    load_tc_config,
    save_tc_config,
)
from q23.dashboard.core import discover_available_strategies, get_strategy_display_name
from q23.shared.config import TransactionCostConfig, TransactionCostScheme, cfg


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return json.dumps({"error": "Could not serialize object"}, indent=2)


def _df_to_csv_bytes(df) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def _export_bundle_bytes(
    strategy_label: str,
    tag: str,
    weights_csv: Optional[bytes],
    diag_csv: Optional[bytes],
    meta_json: Optional[bytes],
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        prefix = f"{strategy_label}_{tag}".replace("/", "_")
        zf.writestr(f"{prefix}__weights.csv", weights_csv or b"")
        zf.writestr(f"{prefix}__diagnostics.csv", diag_csv or b"")
        zf.writestr(f"{prefix}__meta.json", meta_json or b"{}")
        zf.writestr(f"{prefix}__exported_at.txt", _now_iso().encode("utf-8"))
    return buf.getvalue()


def _render_settings_tab() -> None:
    st.subheader("⚙️ Settings / Feature Flags")
    st.caption("Defaults persist across restarts (stored under `.streamlit/`).")

    strategies = discover_available_strategies()
    current_default_sid = st.session_state.get("admin_default_strategy_id") or (strategies[0] if strategies else None)

    col1, col2 = st.columns(2)
    with col1:
        if strategies:
            default_sid = st.selectbox(
                "Default strategy on load",
                options=strategies,
                format_func=get_strategy_display_name,
                index=strategies.index(current_default_sid) if current_default_sid in strategies else 0,
                key="admin_default_strategy_id",
                help="Used when the app starts. Your current selection can differ during a session.",
            )
        else:
            default_sid = None

        show_elite = st.toggle(
            "Show Elite Analytics overlay",
            value=bool(st.session_state.get("admin_show_elite_overlay", True)),
            help="If off, Elite overlay selection + rendering are disabled.",
        )
        st.session_state.admin_show_elite_overlay = bool(show_elite)

        # Build preset options dynamically (matching sidebar options)
        today = datetime.now()
        current_year = today.year
        
        preset_options_admin = []
        if current_year >= 2026:
            preset_options_admin.append("2026")
        preset_options_admin.extend(["2025", "2024", "2023", "2022", "2021", "2020"])
        preset_options_admin.extend(["All"])
        
        current_admin_preset = st.session_state.get("admin_default_date_preset", "2025")
        default_preset = st.selectbox(
            "Default date preset",
            options=preset_options_admin,
            index=preset_options_admin.index(current_admin_preset) if current_admin_preset in preset_options_admin else 0,
            help="Default date preset when the session first initializes. Full years show only that calendar year. YTD shows year-to-date up to today.",
        )
        st.session_state.admin_default_date_preset = default_preset

    with col2:
        include_bm = st.toggle(
            "Default: include benchmarks in Strategy Comparison",
            value=bool(st.session_state.get("admin_default_include_benchmarks", False)),
            help="Used as the initial value for the benchmark toggle on that page.",
        )
        st.session_state.admin_default_include_benchmarks = bool(include_bm)

        col_apply, col_save = st.columns(2)
        with col_apply:
            if st.button("Apply now (rerun)", type="primary"):
                # push defaults into widget keys (best-effort)
                st.session_state.q23_date_preset = st.session_state.admin_default_date_preset
                st.session_state.q23_include_benchmarks = st.session_state.admin_default_include_benchmarks
                # Also set active strategy to default (optional convenience)
                if st.session_state.get("admin_default_strategy_id"):
                    st.session_state.q23_active_strategy = st.session_state.admin_default_strategy_id
                st.rerun()
        with col_save:
            if st.button("Save defaults (sticky)"):
                persisted = load_admin_settings()
                persisted.update(
                    {
                        "admin_show_elite_overlay": bool(st.session_state.admin_show_elite_overlay),
                        "admin_default_date_preset": str(st.session_state.admin_default_date_preset),
                        "admin_default_include_benchmarks": bool(st.session_state.admin_default_include_benchmarks),
                        "admin_default_strategy_id": str(st.session_state.get("admin_default_strategy_id") or ""),
                        "last_saved": _now_iso(),
                    }
                )
                ok1 = save_admin_settings(persisted)
                ok2 = True
                if st.session_state.get("admin_default_strategy_id"):
                    ok2 = save_default_strategy_id(str(st.session_state.admin_default_strategy_id))
                if ok1 and ok2:
                    st.success("Defaults saved.")
                else:
                    st.error("Could not save one or more defaults.")

    st.divider()
    with st.expander("Current session_state (admin keys)", expanded=False):
        admin_keys = {k: v for k, v in st.session_state.items() if str(k).startswith("admin_") or str(k).startswith("q23_")}
        st.code(_safe_json(admin_keys), language="json")

    with st.expander("Persisted admin defaults (.streamlit/q23_admin_settings.json)", expanded=False):
        st.code(_safe_json(load_admin_settings()), language="json")


def _render_system_tab(data: DashboardData, sidebar_state: SidebarState) -> None:
    st.subheader("🧭 System Status")

    proj_root = _project_root()
    out_root = _base_output_dir()

    # Context / identity
    strategy_label = sidebar_state.selected_strategy or sidebar_state.base_name
    st.markdown("#### Current selection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy", strategy_label or "—")
    c2.metric("Tag", sidebar_state.tag or "—")
    c3.metric("Date range", f"{sidebar_state.date_range[0]} → {sidebar_state.date_range[1]}")
    try:
        c4.metric("Last data dt", str(getattr(data, "last_dt", "—")))
    except Exception:
        c4.metric("Last data dt", "—")

    st.markdown("#### Paths")
    st.code(
        "\n".join(
            [
                f"Project root: {proj_root}",
                f"Output root:  {out_root}",
                f"Python:       {sys.version.split()[0]}",
                f"Platform:     {platform.platform()}",
            ]
        )
    )

    st.markdown("#### Data presence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weights rows", 0 if data.weights is None else int(len(data.weights)))
    c2.metric("Weights cols", 0 if data.weights is None else int(len(data.weights.columns)))
    c3.metric("Diag rows", 0 if data.diag is None else int(len(data.diag)))
    c4.metric("Has meta", "yes" if bool(data.meta) else "no")


def _format_kv_line(label: str, value: str, width: int = 28) -> str:
    return f"{label:<{width}}: {value}"


def _env_var_status(name: str) -> str:
    val = os.environ.get(name)
    if val is None or val == "":
        return "UNSET"
    # Don’t leak secrets/keys; just show SET
    if any(s in name.upper() for s in ["KEY", "TOKEN", "SECRET", "PASS"]):
        return "SET"
    return val


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return True, str(ver)
    except Exception:
        return False, "error"


def _build_environment_report() -> str:
    lines: list[str] = []
    lines.append("========================================================================")
    lines.append("Q23 ENVIRONMENT DIAGNOSTIC")
    lines.append("========================================================================")
    lines.append("")

    # Python
    lines.append("========================================================================")
    lines.append("Python")
    lines.append("========================================================================")
    lines.append(_format_kv_line("sys.executable", sys.executable))
    lines.append(_format_kv_line("Python version", sys.version.replace("\n", " ")))
    lines.append(_format_kv_line("Platform", platform.platform()))
    lines.append("")

    # Conda / virtual env
    lines.append("========================================================================")
    lines.append("Conda / Virtual Env")
    lines.append("========================================================================")
    lines.append(_format_kv_line("CONDA_DEFAULT_ENV", os.environ.get("CONDA_DEFAULT_ENV", "UNSET")))
    lines.append(_format_kv_line("CONDA_PREFIX", os.environ.get("CONDA_PREFIX", "UNSET")))
    lines.append(_format_kv_line("VIRTUAL_ENV", os.environ.get("VIRTUAL_ENV", "UNSET")))
    lines.append("")

    # Key env vars
    lines.append("========================================================================")
    lines.append("Key Environment Variables")
    lines.append("========================================================================")
    key_vars = [
        "API_KEY",
        "Q23_PROJECT_DIR",
        "Q23_CONDA_ENV",
        "Q23_CONDA_SH",
        "PYTHONPATH",
    ]
    for k in key_vars:
        lines.append(_format_kv_line(k, _env_var_status(k)))
    lines.append("")

    # Executables
    lines.append("========================================================================")
    lines.append("Executable Availability")
    lines.append("========================================================================")
    for exe in ["python", "pip", "streamlit"]:
        lines.append(_format_kv_line(exe, shutil.which(exe) or "NOT FOUND"))
    lines.append("")

    # Imports
    lines.append("========================================================================")
    lines.append("Package Imports")
    lines.append("========================================================================")
    pkgs = ["streamlit", "qnt", "xarray", "pandas", "numpy"]
    import_ok = True
    versions: dict[str, str] = {}
    for p in pkgs:
        ok, ver = _try_import(p)
        import_ok = import_ok and ok
        versions[p] = ver
        lines.append(_format_kv_line(p, "OK" if ok else "ERROR"))
    lines.append("")

    # Versions
    lines.append("========================================================================")
    lines.append("Package Versions")
    lines.append("========================================================================")
    for p in pkgs:
        lines.append(_format_kv_line(f"{p}.__version__", versions.get(p, "unknown")))
    lines.append("")

    # Summary
    lines.append("========================================================================")
    lines.append("Summary")
    lines.append("========================================================================")
    if import_ok:
        lines.append("✅ Environment looks sane for Q23.")
    else:
        lines.append("⚠️ One or more key imports failed. See above.")

    return "\n".join(lines) + "\n"


def _render_marketstack_tab() -> None:
    """Render Marketstack API status and telemetry."""
    st.subheader("📡 Marketstack API Status")
    st.caption("Real-time data fetching status and telemetry.")
    
    try:
        from q23.strategy.marketstack_telemetry import get_telemetry_manager
        tel_mgr = get_telemetry_manager()
        tel = tel_mgr.get_telemetry()
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        status_color = {
            "enabled": "🟢",
            "disabled": "🔴",
            "error": "🔴",
            "rate_limited": "🟡",
        }
        status_icon = status_color.get(tel.status.value, "⚪")
        
        col1.metric("Status", f"{status_icon} {tel.status.value.upper()}")
        col2.metric("API Key", "✅ SET" if tel.api_key_set else "❌ NOT SET")
        col3.metric("Total Fetches", tel.total_fetches)
        col4.metric("Success Rate", f"{tel.get_success_rate():.1f}%")
        
        # Configuration
        st.markdown("#### Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.code(
                "\n".join([
                    f"Base URL: {tel.config.get('base_url', 'N/A')}",
                    f"Rate Limit Delay: {tel.config.get('rate_limit_delay', 'N/A')}s",
                    f"Default Lookback: {tel.config.get('default_lookback_days', 'N/A')} days",
                ]),
                language="text",
            )
        
        with config_col2:
            st.code(
                "\n".join([
                    f"Force Live Data: {tel.config.get('force_live_data', False)}",
                    f"Pre-Market Cutoff: {tel.config.get('pre_market_cutoff_hour', 'N/A')}:00 ET",
                    f"Post-Market Cutoff: {tel.config.get('post_market_cutoff_hour', 'N/A')}:00 ET",
                ]),
                language="text",
            )
        
        # Statistics
        st.markdown("#### Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Successful Fetches", tel.successful_fetches)
        stat_col2.metric("Failed Fetches", tel.failed_fetches)
        stat_col3.metric("Total Rows Fetched", f"{tel.total_rows_fetched:,}")
        
        # Last fetch details
        if tel.last_fetch:
            st.markdown("#### Last Fetch")
            last_fetch_time = datetime.fromtimestamp(tel.last_fetch.timestamp, tz=timezone.utc)
            last_fetch_col1, last_fetch_col2, last_fetch_col3 = st.columns(3)
            last_fetch_col1.metric("Time", last_fetch_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
            last_fetch_col2.metric("Date Fetched", tel.last_fetch.fetched_date or "N/A")
            last_fetch_col3.metric("Rows", tel.last_fetch.rows_fetched)
            
            result_icon = "✅" if tel.last_fetch.result.value == "success" else "❌"
            st.info(
                f"{result_icon} **Result**: {tel.last_fetch.result.value.upper()} | "
                f"**Symbols**: {tel.last_fetch.symbols_count} | "
                f"**Duration**: {tel.last_fetch.duration_ms:.0f}ms | "
                f"**Strategy**: {tel.last_fetch.strategy_id or 'N/A'}"
            )
            if tel.last_fetch.error_message:
                st.error(f"Error: {tel.last_fetch.error_message}")
        else:
            st.info("No fetch history yet.")
        
        # Recent activity
        if tel.fetch_history:
            st.markdown("#### Recent Activity (Last 10)")
            activity_data = []
            for record in tel.fetch_history[-10:]:
                fetch_time = datetime.fromtimestamp(record.timestamp, tz=timezone.utc)
                activity_data.append({
                    "Time": fetch_time.strftime("%H:%M:%S"),
                    "Date": record.fetched_date or f"{record.date_from}→{record.date_to}" or "N/A",
                    "Result": f"{'✅' if record.result.value == 'success' else '❌'} {record.result.value.upper()}",
                    "Symbols": record.symbols_count,
                    "Rows": record.rows_fetched,
                    "Duration (ms)": f"{record.duration_ms:.0f}",
                    "Strategy": record.strategy_id or "—",
                })
            st.dataframe(activity_data, use_container_width=True, hide_index=True)
        
        # Export telemetry
        st.divider()
        tel_json = json.dumps(tel_mgr.get_telemetry().to_dict(), indent=2, default=str)
        st.download_button(
            "Download Telemetry JSON",
            data=tel_json.encode("utf-8"),
            file_name=f"marketstack_telemetry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
        
    except Exception as e:
        st.error(f"Error loading Marketstack telemetry: {e}")
        st.code(str(e), language="text")


def _render_environment_diagnostics() -> None:
    st.subheader("🧪 Environment diagnostics")
    st.caption("High-signal runtime + dependency checks (safe: secrets are redacted).")

    report = _build_environment_report()

    # Quick glance cards
    py = sys.version.split()[0]
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or "—"
    streamlit_ok, streamlit_ver = _try_import("streamlit")
    st_ok, _ = _try_import("qnt")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Python", py)
    c2.metric("Conda env", conda_env)
    c3.metric("Streamlit", streamlit_ver if streamlit_ok else "ERROR")
    c4.metric("qnt import", "OK" if st_ok else "ERROR")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.download_button(
            "Download env report (txt)",
            data=report.encode("utf-8"),
            file_name="q23_environment_diagnostic.txt",
            mime="text/plain",
        )
    with col_b:
        if st.button("Copy-ready view"):
            st.info("Select the text below and copy.")

    st.code(report)


def _render_tc_settings_tab() -> None:
    """Render Transaction Cost configuration settings."""
    st.subheader("💰 Transaction Cost Settings")
    st.caption(
        "Configure the default transaction cost model. "
        "Default is **Quantiacs ATR**: TC = 5% × ATR(14) × |ΔPosition|"
    )
    
    # Load current TC config
    tc_config = load_tc_config()
    
    # Scheme selection
    scheme_options = [
        TransactionCostScheme.QUANTIACS_ATR,
        TransactionCostScheme.FLAT_BPS,
        TransactionCostScheme.PERCENTAGE,
        TransactionCostScheme.TIERED,
    ]
    scheme_labels = {
        TransactionCostScheme.QUANTIACS_ATR: "Quantiacs ATR (5% × ATR(14) × |ΔPos|)",
        TransactionCostScheme.FLAT_BPS: "Flat Basis Points",
        TransactionCostScheme.PERCENTAGE: "Fixed Percentage",
        TransactionCostScheme.TIERED: "Tiered by Trade Size",
    }
    
    current_scheme_idx = 0
    for i, s in enumerate(scheme_options):
        if s == tc_config.scheme:
            current_scheme_idx = i
            break
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_scheme = st.selectbox(
            "TC Model",
            options=scheme_options,
            format_func=lambda x: scheme_labels.get(x, x.value),
            index=current_scheme_idx,
            key="admin_tc_scheme_select",
            help="Choose the transaction cost model. Quantiacs ATR is recommended for contest alignment.",
        )
        
        # ATR-specific settings
        if selected_scheme == TransactionCostScheme.QUANTIACS_ATR:
            st.markdown("##### ATR Model Parameters")
            atr_window = st.slider(
                "ATR Window",
                min_value=5,
                max_value=30,
                value=tc_config.atr_window,
                step=1,
                key="admin_tc_atr_window_slider",
                help="Number of days for ATR calculation. Quantiacs uses 14.",
            )
            atr_multiplier = st.number_input(
                "ATR Multiplier",
                min_value=0.01,
                max_value=0.20,
                value=tc_config.atr_multiplier,
                step=0.01,
                format="%.2f",
                key="admin_tc_atr_mult_input",
                help="Multiplier for ATR. Quantiacs uses 0.05 (5%).",
            )
        else:
            atr_window = tc_config.atr_window
            atr_multiplier = tc_config.atr_multiplier
    
    with col2:
        # Flat BPS settings (used as fallback or for FLAT_BPS scheme)
        st.markdown("##### Flat BPS Settings")
        flat_bps = st.number_input(
            "Flat Basis Points",
            min_value=0.0,
            max_value=100.0,
            value=tc_config.flat_bps,
            step=1.0,
            format="%.1f",
            key="admin_tc_flat_bps_input",
            help="Used for FLAT_BPS scheme or as fallback when ATR unavailable.",
        )
        
        # Percentage settings
        if selected_scheme == TransactionCostScheme.PERCENTAGE:
            percentage = st.number_input(
                "Percentage",
                min_value=0.0001,
                max_value=0.01,
                value=tc_config.percentage,
                step=0.0001,
                format="%.4f",
                key="admin_tc_percentage_input",
                help="Fixed percentage per trade (e.g., 0.001 = 0.1%).",
            )
        else:
            percentage = tc_config.percentage
    
    # Display estimated impact
    st.divider()
    st.markdown("##### Estimated Impact")
    
    if selected_scheme == TransactionCostScheme.QUANTIACS_ATR:
        # Show estimate for typical ATR values
        typical_atr_pct = 0.015  # 1.5% typical ATR/price
        est_tc_per_trade = atr_multiplier * typical_atr_pct * 100
        st.info(
            f"**Estimated TC per trade**: ~{est_tc_per_trade:.1f} bps "
            f"(assuming typical ATR = 1.5% of price)\n\n"
            f"For a 100% daily turnover portfolio, this would be ~{est_tc_per_trade * 252 / 100:.1f}% annual drag."
        )
    elif selected_scheme == TransactionCostScheme.FLAT_BPS:
        st.info(
            f"**TC per trade**: {flat_bps:.1f} bps\n\n"
            f"For a 100% daily turnover portfolio, this would be ~{flat_bps * 252 / 10000 * 100:.1f}% annual drag."
        )
    elif selected_scheme == TransactionCostScheme.PERCENTAGE:
        st.info(
            f"**TC per trade**: {percentage * 100:.2f}%\n\n"
            f"For a 100% daily turnover portfolio, this would be ~{percentage * 252 * 100:.1f}% annual drag."
        )
    
    # Save buttons
    st.divider()
    col_save, col_reset = st.columns(2)
    
    with col_save:
        if st.button("💾 Save TC Settings", type="primary", key="admin_tc_save_btn"):
            new_tc_config = TransactionCostConfig(
                scheme=selected_scheme,
                atr_window=atr_window,
                atr_multiplier=atr_multiplier,
                flat_bps=flat_bps,
                percentage=percentage,
                min_cost_bps=tc_config.min_cost_bps,
                max_cost_bps=tc_config.max_cost_bps,
                tiered_thresholds=tc_config.tiered_thresholds,
            )
            
            try:
                new_tc_config.validate()
                if save_tc_config(new_tc_config):
                    # Also update session state
                    st.session_state.q23_tc_config = new_tc_config
                    st.success("Transaction cost settings saved!")
                else:
                    st.error("Failed to save TC settings.")
            except ValueError as e:
                st.error(f"Invalid TC config: {e}")
    
    with col_reset:
        if st.button("🔄 Reset to Quantiacs Default", key="admin_tc_reset_btn"):
            default_config = TransactionCostConfig()  # Quantiacs defaults
            if save_tc_config(default_config):
                st.session_state.q23_tc_config = default_config
                st.success("Reset to Quantiacs default (5% × ATR(14)).")
                st.rerun()
            else:
                st.error("Failed to reset TC settings.")
    
    # Show current config
    with st.expander("Current TC Configuration (JSON)", expanded=False):
        st.code(_safe_json(tc_config.to_dict()), language="json")


def _render_ops_tab() -> None:
    st.subheader("🛠 Ops")
    st.caption("Safe admin controls (no long-running subprocess execution inside Streamlit for MVP).")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh (clear cache + rerun)", type="primary"):
            st.cache_data.clear()
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

    with col2:
        if st.button("🧹 Clear Streamlit cache"):
            st.cache_data.clear()
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.success("Caches cleared. Use Refresh to rerun.")

    with col3:
        if st.button("🔁 Rerun"):
            st.rerun()

    st.divider()
    st.markdown("#### Run controls (CLI)")
    st.caption("Recommended commands to run outside the UI:")
    st.code("./Run_Q23.command", language="bash")
    st.code("python run_strategy.py --strategy <strategy_id>", language="bash")
    st.code("python run_dashboard.py", language="bash")

def _render_ephemeral_run_cache() -> None:
    st.subheader("🗄 Ephemeral Run Cache")
    st.caption("Temporary runs used by the Live Strategy Lab. Safe to clean up.")

    cache = get_run_cache()
    runs = cache.list_runs()

    c1, c2, c3 = st.columns(3)
    c1.metric("Cached runs", len(runs))
    c2.metric("Cache size (MB)", f"{cache.size_mb:.2f}")
    c3.metric("Cache dir", str(getattr(cache, "cache_dir", "")))

    if not runs:
        st.info("No ephemeral runs found.")
        return

    with st.expander("View / delete runs", expanded=False):
        selected = st.selectbox("Select run", options=runs, index=0, key="ops_ephemeral_selected")
        meta = cache.get_run_metadata(selected) if selected else None
        if meta:
            st.code(_safe_json(meta), language="json")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("🗑 Delete selected run", type="secondary"):
                ok = cache.delete_run(selected)
                if ok:
                    st.success("Deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete run.")
        with col_b:
            st.caption("Tip: use cleanup below to remove old runs in bulk.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        max_age = st.slider("Cleanup runs older than (hours)", 1, 168, 24, 1)
        if st.button("🧹 Cleanup old runs", type="primary"):
            n = cache.cleanup_old(max_age_hours=int(max_age))
            st.success(f"Deleted {n} old runs.")
            st.rerun()
    with col2:
        st.warning("This deletes ALL ephemeral runs in the cache directory.")
        if st.button("🔥 Delete ALL ephemeral runs"):
            n = cache.cleanup()
            st.success(f"Deleted {n} runs.")
            st.rerun()


def _render_artifact_health(sidebar_state: SidebarState) -> None:
    st.subheader("🧾 Artifact Health")
    st.caption("Sanity-check expected output files for the current selection.")

    try:
        paths = _artifact_paths(sidebar_state.base_dir, sidebar_state.base_name, sidebar_state.tag)
    except Exception:
        st.warning("Could not compute artifact paths for this selection.")
        return

    rows = []
    meta_data = None
    for key, p in paths.items():
        exists = p.exists()
        if exists and p.is_file():
            stat = p.stat()
            size_kb = round(stat.st_size / 1024, 1)
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            
            # Load meta.json to check Marketstack info
            if key == "meta" and p.suffix == ".json":
                try:
                    with open(p, 'r') as f:
                        meta_data = json.load(f)
                except Exception:
                    pass
        else:
            size_kb = "—"
            modified = "—"
        rows.append(
            {
                "artifact": key,
                "exists": exists,
                "size_kb": size_kb,
                "modified": modified,
                "path": str(p),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)
    
    # Show Marketstack info from meta.json if available
    if meta_data:
        marketstack_info = meta_data.get("data_source", {}).get("marketstack", {})
        if marketstack_info:
            st.markdown("#### 📡 Marketstack Data Status")
            col1, col2, col3 = st.columns(3)
            col1.metric("Enabled", "✅ YES" if marketstack_info.get("enabled") else "❌ NO")
            col2.metric("Used", "✅ YES" if marketstack_info.get("used") else "❌ NO")
            col3.metric("Fetched Date", marketstack_info.get("fetched_date") or "—")
            
            latest_date = meta_data.get("data_source", {}).get("latest_date")
            date_range = meta_data.get("date_range", [])
            if latest_date and date_range:
                st.info(
                    f"**Latest data date**: {latest_date} | "
                    f"**Date range in outputs**: {date_range[0]} → {date_range[1]}"
                )
                if latest_date != date_range[1]:
                    st.warning(
                        f"⚠️ Latest data date ({latest_date}) differs from end date in outputs ({date_range[1]}). "
                        f"This may indicate Marketstack data wasn't included in this run."
                    )


def _render_export_tab(data: DashboardData, sidebar_state: SidebarState) -> None:
    st.subheader("⬇️ Export")
    st.caption("Download the currently loaded selection (based on sidebar Strategy/Tag).")

    strategy_label = sidebar_state.selected_strategy or sidebar_state.base_name or "strategy"
    tag = sidebar_state.tag or "tag"

    weights_bytes: Optional[bytes] = None
    diag_bytes: Optional[bytes] = None
    meta_bytes: Optional[bytes] = None

    if data.weights is not None and not data.weights.empty:
        weights_bytes = _df_to_csv_bytes(data.weights)
    if data.diag is not None and not data.diag.empty:
        diag_bytes = _df_to_csv_bytes(data.diag)
    if data.meta is not None:
        meta_bytes = _safe_json(data.meta).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download weights CSV",
            data=weights_bytes or b"",
            file_name=f"{strategy_label}_weights_{tag}.csv",
            mime="text/csv",
            disabled=weights_bytes is None,
        )
    with c2:
        st.download_button(
            "Download diagnostics CSV",
            data=diag_bytes or b"",
            file_name=f"{strategy_label}_diagnostics_{tag}.csv",
            mime="text/csv",
            disabled=diag_bytes is None,
        )
    with c3:
        st.download_button(
            "Download meta JSON",
            data=meta_bytes or b"{}",
            file_name=f"{strategy_label}_meta_{tag}.json",
            mime="application/json",
            disabled=meta_bytes is None,
        )

    st.divider()
    bundle = _export_bundle_bytes(strategy_label=strategy_label, tag=tag, weights_csv=weights_bytes, diag_csv=diag_bytes, meta_json=meta_bytes)
    st.download_button(
        "Download full bundle (zip)",
        data=bundle,
        file_name=f"{strategy_label}_bundle_{tag}.zip",
        mime="application/zip",
    )


def render_admin_panel(
    data: DashboardData,
    sidebar_state: SidebarState,
    default_panel: str = "settings",
) -> None:
    st.header("Ops Console")
    st.caption("Single-admin control center. Expand sections as needed.")

    with st.expander("⬇️ Export (current selection)", expanded=True):
        _render_export_tab(data, sidebar_state)

    with st.expander("🛠 Quick actions", expanded=True):
        _render_ops_tab()

    with st.expander("💰 Transaction Costs", expanded=True):
        _render_tc_settings_tab()

    with st.expander("🧭 System status", expanded=True):
        _render_system_tab(data, sidebar_state)

    with st.expander("🧪 Environment diagnostics", expanded=True):
        _render_environment_diagnostics()

    with st.expander("🧾 Artifact health", expanded=True):
        _render_artifact_health(sidebar_state)

    with st.expander("🗄 Ephemeral run cache", expanded=True):
        _render_ephemeral_run_cache()

    with st.expander("⚙️ Settings / feature flags", expanded=True):
        _render_settings_tab()
    
    with st.expander("📡 Marketstack API", expanded=True):
        _render_marketstack_tab()

