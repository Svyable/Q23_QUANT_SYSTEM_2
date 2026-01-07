from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.shared.config import cfg

# Type imports (avoid circular imports)
try:
    from q23.strategies.base import StrategyArtifacts
except ImportError:
    StrategyArtifacts = None  # type: ignore

TAG_RE = re.compile(r"_wide_weights_(.+?)\.csv$")


def _parse_tag_datetime(tag: str) -> Optional[datetime]:
    """Parse date from tag, handling multiple formats.
    
    Supported formats:
    - YYYYMMDD_HHMMSS (e.g., 20251219_112716)
    - YYYYMMDD (e.g., 20251219)
    - YYYY-MM-DD_HHMMSS (e.g., 2025-12-19_112716)
    - YYYY-MM-DD (e.g., 2025-12-19)
    
    Args:
        tag: Tag string to parse
        
    Returns:
        datetime if successfully parsed, None otherwise
    """
    if not tag:
        return None
    
    formats = [
        ("%Y%m%d_%H%M%S", 15),   # 20251219_112716
        ("%Y%m%d", 8),           # 20251219
        ("%Y-%m-%d_%H%M%S", 17), # 2025-12-19_112716
        ("%Y-%m-%d", 10),        # 2025-12-19
    ]
    
    for fmt, length in formats:
        try:
            parse_str = tag[:length] if len(tag) >= length else tag
            return datetime.strptime(parse_str, fmt)
        except (ValueError, IndexError):
            continue
    
    # Fallback: try pandas as last resort
    try:
        return pd.to_datetime(tag).to_pydatetime()
    except Exception:
        pass
    
    return None


def get_default_date_range() -> Dict[str, str]:
    today = datetime.now()
    current_year = today.year
    return {
        "min_date": "2024-01-01",
        "default_start": f"{current_year}-01-01",
        "default_end": today.strftime("%Y-%m-%d"),
        "ytd_start": f"{current_year}-01-01",
    }


def get_date_preset_range(preset: str) -> Tuple[str, str]:
    """Convert preset name to (start_date, end_date) tuple."""
    today = datetime.now()

    presets = {
        "2025": ("2025-01-01", today.strftime("%Y-%m-%d")),
        "2024-25": ("2024-01-01", today.strftime("%Y-%m-%d")),
        "2024": ("2024-01-01", "2024-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2020": ("2020-01-01", "2020-12-31"),
        "All": ("2020-01-01", today.strftime("%Y-%m-%d")),
    }

    return presets.get(preset, ("2020-01-01", today.strftime("%Y-%m-%d")))


def get_strategy_output_dir(strategy_id: str) -> Path:
    """Get output directory for a strategy.
    
    Handles two cases:
    1. Legacy: OUTPUT_ROOT already contains strategy directory (e.g., "outputs/NASNYS_V4")
    2. New: OUTPUT_ROOT is base, append strategy_id (e.g., "outputs" -> "outputs/nasnys_v4")
    """
    root = _project_root()
    output_root = Path(cfg.paths.OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = root / output_root
    
    # Check if OUTPUT_ROOT already ends with a directory matching strategy_id (case-insensitive)
    output_root_name = output_root.name.upper()
    strategy_id_upper = strategy_id.upper()
    
    if output_root_name == strategy_id_upper:
        # Legacy case: OUTPUT_ROOT already contains the strategy directory
        return output_root
    else:
        # New case: append strategy_id
        return output_root / strategy_id


def discover_available_strategies(include_benchmarks: bool = False) -> List[str]:
    """Discover available strategies from the registry.
    
    Args:
        include_benchmarks: If False (default), excludes benchmark strategies
                           from the list. Benchmarks start with 'benchmark_'.
    
    Returns:
        List of strategy IDs
    """
    try:
        from q23.strategies.registry import StrategyRegistry
        all_ids = StrategyRegistry.list_ids()
        if include_benchmarks:
            return all_ids
        # Filter out benchmark strategies (they start with 'benchmark_')
        return [s for s in all_ids if not s.startswith("benchmark_")]
    except ImportError:
        return []


def get_enabled_strategies() -> Dict[str, bool]:
    """Load enabled strategies from enabled_strategies.json.
    
    Returns:
        Dictionary mapping strategy_id -> enabled (bool)
    """
    config_path = _project_root() / "src" / "q23" / "strategies" / "enabled_strategies.json"
    if config_path.exists():
        try:
            import json
            data = json.loads(config_path.read_text())
            return data.get("enabled", {})
        except Exception:
            pass
    return {}


def get_enabled_strategy_ids(include_benchmarks: bool = False) -> List[str]:
    """Get list of enabled strategy IDs.
    
    Args:
        include_benchmarks: If True, includes enabled benchmarks in the list.
    
    Returns:
        List of enabled strategy IDs
    """
    all_strategies = discover_available_strategies(include_benchmarks=include_benchmarks)
    enabled = get_enabled_strategies()
    return [sid for sid in all_strategies if enabled.get(sid, False)]


def get_available_benchmarks() -> List[str]:
    """Get list of available benchmark strategy IDs.
    
    Returns:
        List of benchmark strategy IDs that have saved runs
    """
    try:
        from q23.strategies.registry import StrategyRegistry
        all_ids = StrategyRegistry.list_ids()
        # Filter to only benchmark strategies that have saved runs
        benchmarks = [s for s in all_ids if s.startswith("benchmark_")]
        return [b for b in benchmarks if discover_strategy_tags(b)]
    except ImportError:
        return []


def discover_strategy_tags(strategy_id: str) -> List[str]:
    """Discover available tags for a strategy.
    
    Checks multiple locations and filename patterns:
    1. Strategy-specific directory with strategy_id base name
    2. Legacy location (OUTPUT_ROOT) with BASE_NAME base name
    3. Case-insensitive matching for filenames
    """
    strategy_dir = get_strategy_output_dir(strategy_id)
    root = _project_root()
    output_root = Path(cfg.paths.OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = root / output_root
    
    tags: List[str] = []
    seen_tags = set()
    
    strategy_id_upper = strategy_id.upper()
    
    # Check strategy-specific directory with strategy_id base name
    if strategy_dir.exists():
        # First try exact match with strategy_id
        pat = f"{strategy_id}_wide_weights_*.csv"
        for f in sorted(strategy_dir.glob(pat)):
            m = TAG_RE.search(f.name)
            if m:
                tag = m.group(1)
                if tag not in seen_tags:
                    tags.append(tag)
                    seen_tags.add(tag)
        
        # Also try case-insensitive matching - check if filename starts with strategy_id
        for f in sorted(strategy_dir.glob("*_wide_weights_*.csv")):
            fname_upper = f.stem.upper()
            # Match if filename starts with strategy_id (case-insensitive) or BASE_NAME
            if (fname_upper.startswith(f"{strategy_id_upper}_WIDE_WEIGHTS_") or
                fname_upper.startswith(f"{cfg.paths.BASE_NAME.upper()}_WIDE_WEIGHTS_")):
                m = TAG_RE.search(f.name)
                if m:
                    tag = m.group(1)
                    if tag not in seen_tags:
                        tags.append(tag)
                        seen_tags.add(tag)
    
    # Check legacy location (OUTPUT_ROOT directly) with BASE_NAME
    # This handles files created by legacy StrategyEngine when strategy_dir != output_root
    if output_root.exists() and output_root != strategy_dir:
        base_name = cfg.paths.BASE_NAME
        pat = f"{base_name}_wide_weights_*.csv"
        for f in sorted(output_root.glob(pat)):
            m = TAG_RE.search(f.name)
            if m:
                tag = m.group(1)
                if tag not in seen_tags:
                    tags.append(tag)
                    seen_tags.add(tag)
        
        # Also try case-insensitive matching for strategy_id
        for f in sorted(output_root.glob("*_wide_weights_*.csv")):
            # Check if filename starts with strategy_id (case-insensitive)
            fname_upper = f.stem.upper()
            if fname_upper.startswith(f"{strategy_id_upper}_WIDE_WEIGHTS_"):
                m = TAG_RE.search(f.name)
                if m:
                    tag = m.group(1)
                    if tag not in seen_tags:
                        tags.append(tag)
                        seen_tags.add(tag)

    def _key(t: str):
        dt = _parse_tag_datetime(t)
        if dt is not None:
            return dt
        return datetime.min

    return sorted(tags, key=_key, reverse=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _base_output_dir() -> Path:
    root = _project_root()
    p = Path(cfg.paths.OUTPUT_ROOT)
    if not p.is_absolute():
        p = root / p
    return p


def _discover_tags(base_dir: Path, base_name: str) -> List[str]:
    if not base_dir.exists():
        return []

    tags: List[str] = []
    pat = f"{base_name}_wide_weights_*.csv"
    for f in sorted(base_dir.glob(pat)):
        m = TAG_RE.search(f.name)
        if m:
            tags.append(m.group(1))

    def _key(t: str):
        dt = _parse_tag_datetime(t)
        if dt is not None:
            return dt
        return datetime.min

    tags_sorted = sorted(tags, key=_key, reverse=True)
    return tags_sorted


def _artifact_paths(base_dir: Path, base_name: str, tag: str) -> Dict[str, Path]:
    return {
        "wide_weights": base_dir / f"{base_name}_wide_weights_{tag}.csv",
        "budget": base_dir / f"{base_name}_budget_{tag}.csv",
        "factor_weights": base_dir / f"{base_name}_factor_weights_{tag}.csv",
        "ic": base_dir / f"{base_name}_ic_{tag}.csv",
        "meta": base_dir / f"{base_name}_meta_{tag}.json",
        "portfolio_diag": base_dir / f"{base_name}_portfolio_diag_{tag}.csv",
        "factor_exposure": base_dir / f"{base_name}_factor_exposure_{tag}.csv",
        "factor_vectors": base_dir / f"{base_name}_factor_vectors_{tag}.csv",
    }


@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    """Optimized CSV reading with better performance and memory usage.
    
    Uses optimized pandas parameters for faster loading and lower memory footprint.
    """
    return pd.read_csv(
        path,
        low_memory=False,  # Better performance when schema is known
        engine='c',  # C engine is faster
        na_values=['', 'NA', 'N/A', 'null', 'NULL'],  # Common NA representations
        keep_default_na=True,
    )


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def _coerce_time_index(df: pd.DataFrame, optimize: bool = True) -> pd.DataFrame:
    """Coerce time index with optimized date parsing.
    
    Args:
        df: DataFrame to process
        optimize: If True, optimize dtypes after date parsing
    """
    if len(df.columns) == 0:
        return df
    
    # Try to identify time column more efficiently
    time_col = None
    if df.columns[0].lower() in ("time", "date", "datetime"):
        time_col = df.columns[0]
    elif df.index.name and df.index.name.lower() in ("time", "date", "datetime"):
        # Already has time index
        if optimize:
            df.index = pd.to_datetime(df.index, format="mixed", errors="coerce")
            df = df[~df.index.isna()].sort_index()
        return df
    else:
        # Check first column
        first_col = df.columns[0]
        sample = df[first_col].head(100)
        if pd.api.types.is_datetime64_any_dtype(sample):
            time_col = first_col
        elif sample.dtype == 'object':
            # Try parsing a sample
            try:
                pd.to_datetime(sample.head(10), format="mixed", errors='raise')
                time_col = first_col
            except (ValueError, TypeError):
                pass
    
    if time_col:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], format="mixed", errors="coerce")
        df = df.set_index(time_col)
        df = df[~df.index.isna()].sort_index()
    else:
        # Fallback: try first column
        df = df.copy()
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], format="mixed", errors="coerce")
        if df[first_col].notna().mean() > 0.7:
            df = df.set_index(first_col)
            df = df[~df.index.isna()].sort_index()
    
    return df


def _sanitize_numeric(df: pd.DataFrame, optimize_dtypes: bool = True) -> pd.DataFrame:
    """Sanitize numeric columns with optional dtype optimization.
    
    Args:
        df: DataFrame to sanitize
        optimize_dtypes: If True, optimize dtypes to reduce memory
    """
    df = df.copy()
    
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            # Already numeric, just sanitize
            df[c] = df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        else:
            # Try to convert to numeric
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception:
                continue
    
    # Optimize dtypes if requested
    if optimize_dtypes:
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                # For dashboard display, float32 is usually sufficient
                # and reduces memory by 50%
                col_min = df[c].min()
                col_max = df[c].max()
                if (pd.notna(col_min) and pd.notna(col_max) and
                    abs(col_min) < 1e30 and abs(col_max) < 1e30):
                    df[c] = df[c].astype(np.float32, copy=False)
    
    return df


def _load_bundle(paths: Dict[str, Path]) -> Dict[str, Optional[object]]:
    """Load dashboard data bundle with optimized loading.
    
    Uses parallel-friendly structure and optimized data types.
    """
    out: Dict[str, Optional[object]] = {k: None for k in paths.keys()}

    # Load files in order of likely size (smaller first for faster initial feedback)
    if paths["meta"].exists():
        out["meta"] = _read_json(str(paths["meta"]))

    if paths["budget"].exists():
        b = _read_csv(str(paths["budget"]))
        b = _coerce_time_index(b, optimize=True)
        out["budget"] = _sanitize_numeric(b, optimize_dtypes=True)

    if paths["wide_weights"].exists():
        w = _read_csv(str(paths["wide_weights"]))
        w = _coerce_time_index(w, optimize=True)
        out["wide_weights"] = _sanitize_numeric(w, optimize_dtypes=True)

    if paths["factor_weights"].exists():
        fw = _read_csv(str(paths["factor_weights"]))
        fw = _coerce_time_index(fw, optimize=True)
        out["factor_weights"] = _sanitize_numeric(fw, optimize_dtypes=True)

    if paths["ic"].exists():
        ic = pd.read_csv(
            str(paths["ic"]), 
            header=[0, 1],
            low_memory=False,
            engine='c',
        )
        # Optimize IC loading
        if ic.columns[0][0].lower() in ("time", "date", "datetime"):
            time_col = ic.columns[0]
            ic[time_col] = pd.to_datetime(ic[time_col], format="mixed", errors="coerce")
            ic = ic.set_index(time_col)
        else:
            first_col = ic.columns[0]
            ic[first_col] = pd.to_datetime(ic[first_col], format="mixed", errors="coerce")
            if ic[first_col].notna().mean() > 0.7:
                ic = ic.set_index(first_col)
        
        ic.index = pd.to_datetime(ic.index, format="mixed", errors="coerce")
        ic = ic[~ic.index.isna()].sort_index()
        ic = ic.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out["ic"] = ic

    if paths["portfolio_diag"].exists():
        d = _read_csv(str(paths["portfolio_diag"]))
        d = _coerce_time_index(d, optimize=True)
        out["portfolio_diag"] = _sanitize_numeric(d, optimize_dtypes=True)

    if paths["factor_exposure"].exists():
        e = _read_csv(str(paths["factor_exposure"]))
        e = _coerce_time_index(e, optimize=True)
        out["factor_exposure"] = _sanitize_numeric(e, optimize_dtypes=True)

    if paths["factor_vectors"].exists():
        fv = pd.read_csv(
            str(paths["factor_vectors"]), 
            index_col=0,
            low_memory=False,
            engine='c',
        )
        fv.index = fv.index.astype(str)
        fv = fv.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Factor vectors are typically smaller, so dtype optimization less critical
        out["factor_vectors"] = fv

    return out


@dataclass
class DashboardData:
    weights: Optional[pd.DataFrame]
    budget: Optional[pd.DataFrame]
    factor_weights: Optional[pd.DataFrame]
    ic: Optional[pd.DataFrame]
    meta: Optional[Dict]
    diag: Optional[pd.DataFrame]
    exposure: Optional[pd.DataFrame]
    factor_vectors: Optional[pd.DataFrame]
    tag: str
    base_name: str

    @property
    def last_dt(self) -> pd.Timestamp:
        if self.weights is not None and not self.weights.empty:
            return self.weights.index.max()
        return pd.Timestamp.now()

    @property
    def w_last(self) -> pd.Series:
        if self.weights is not None and not self.weights.empty:
            return self.weights.loc[self.last_dt].astype(float)
        return pd.Series(dtype=float)


def load_dashboard_data(tag: str, base_dir: Path, base_name: str) -> DashboardData:
    paths = _artifact_paths(base_dir, base_name, tag)
    bundle = _load_bundle(paths)

    return DashboardData(
        weights=bundle.get("wide_weights"),
        budget=bundle.get("budget"),
        factor_weights=bundle.get("factor_weights"),
        ic=bundle.get("ic"),
        meta=bundle.get("meta"),
        diag=bundle.get("portfolio_diag"),
        exposure=bundle.get("factor_exposure"),
        factor_vectors=bundle.get("factor_vectors"),
        tag=tag,
        base_name=base_name,
    )


def artifacts_to_dashboard_data(
    artifacts: Any,  # StrategyArtifacts
    strategy_id: str,
) -> DashboardData:
    """Convert StrategyArtifacts to DashboardData for benchmarks.
    
    Benchmarks are computed on-demand and don't have output files,
    so we convert their StrategyArtifacts directly to DashboardData.
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None
    
    if xr is None:
        raise ImportError("xarray required for artifacts_to_dashboard_data")
    
    # Convert xarray DataArrays to pandas DataFrames
    weights_df = None
    if artifacts.weights is not None:
        weights_df = artifacts.weights.transpose("time", "asset").to_pandas()
        if not isinstance(weights_df, pd.DataFrame):
            weights_df = pd.DataFrame(weights_df)
        weights_df.index = pd.to_datetime(weights_df.index)
    
    budget_df = None
    if artifacts.budget is not None:
        budget_df = artifacts.budget.to_pandas()
        if isinstance(budget_df, pd.Series):
            budget_df = pd.DataFrame({"budget": budget_df})
        budget_df.index = pd.to_datetime(budget_df.index)
    
    factor_weights_df = None
    if artifacts.factor_weights is not None:
        factor_weights_df = artifacts.factor_weights.transpose("time", "factor").to_pandas()
        if not isinstance(factor_weights_df, pd.DataFrame):
            factor_weights_df = pd.DataFrame(factor_weights_df)
        factor_weights_df.index = pd.to_datetime(factor_weights_df.index)
    
    # Create IC DataFrame from ic_raw and ic_smooth
    ic_df = None
    if artifacts.ic_raw is not None and artifacts.ic_smooth is not None:
        ic_raw_df = artifacts.ic_raw.transpose("time", "factor").to_pandas()
        ic_smooth_df = artifacts.ic_smooth.transpose("time", "factor").to_pandas()
        if isinstance(ic_raw_df, pd.DataFrame) and isinstance(ic_smooth_df, pd.DataFrame):
            # Create multi-index columns: (raw/smooth, factor)
            ic_df = pd.DataFrame()
            for col in ic_raw_df.columns:
                ic_df[(col, "raw")] = ic_raw_df[col]
                ic_df[(col, "smooth")] = ic_smooth_df[col]
            ic_df.index = pd.to_datetime(ic_df.index)
    
    # Portfolio diagnostics should be computed in the benchmark's run method
    # and stored in meta, or we can compute them here if we have the data
    diag_df = None
    # For benchmarks, diag is computed in the run method and stored in meta
    # We'll extract it if available, otherwise create minimal diag
    if artifacts.meta and "_diag" in artifacts.meta:
        diag_df = artifacts.meta["_diag"]
        if diag_df is not None:
            # Ensure it's a DataFrame with proper index
            if not isinstance(diag_df, pd.DataFrame):
                diag_df = pd.DataFrame(diag_df)
            diag_df.index = pd.to_datetime(diag_df.index)
    elif weights_df is not None:
        # Create minimal diag from weights (portfolio returns computed from weights)
        # For benchmarks, portfolio return is the change in cumulative value
        # This is a simplified approach - proper diag should come from run method
        port_ret = pd.Series(0.0, index=weights_df.index)
        if len(weights_df) > 1:
            # Approximate: portfolio return is change in total weight
            port_ret = weights_df.sum(axis=1).pct_change().fillna(0.0)
        
        diag_df = pd.DataFrame({
            "port_ret": port_ret,
            "mkt_ret": port_ret,  # For benchmarks, market return = portfolio return
            "active_ret": 0.0,
            "gross_exposure": weights_df.abs().sum(axis=1),
            "net_exposure": weights_df.sum(axis=1),
            "n_positions": (weights_df.abs() > 1e-12).sum(axis=1),
            "herfindahl": (weights_df ** 2).sum(axis=1),
            "turnover": weights_df.diff().abs().sum(axis=1).fillna(0.0),
            "port_ret_net_tc": port_ret,
            "active_ret_net_tc": 0.0,
            "rolling_vol": port_ret.rolling(21).std().fillna(0.0) * np.sqrt(252),
        }, index=weights_df.index)
        diag_df.index.name = "time"
    
    return DashboardData(
        weights=weights_df,
        budget=budget_df,
        factor_weights=factor_weights_df,
        ic=ic_df,
        meta=artifacts.meta,
        diag=diag_df,
        exposure=None,  # Benchmarks don't have factor exposure
        factor_vectors=None,  # Benchmarks don't have factor vectors
        tag=artifacts.tag,
        base_name=strategy_id,
    )


def _filter_dashboard_data_by_date_range(
    data: DashboardData,
    date_range: Tuple[str, str]
) -> DashboardData:
    """Filter dashboard data by date range.

    Args:
        data: DashboardData to filter
        date_range: Tuple of (start_date, end_date) strings

    Returns:
        Filtered DashboardData
    """
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Filter weights (time x asset DataFrame)
    filtered_weights = None
    if data.weights is not None and not data.weights.empty:
        if isinstance(data.weights.index, pd.DatetimeIndex):
            mask = (data.weights.index >= start_date) & (data.weights.index <= end_date)
            filtered_weights = data.weights.loc[mask].copy() if mask.any() else None

    # Filter diagnostics (time series data)
    filtered_diag = None
    if data.diag is not None and not data.diag.empty:
        if isinstance(data.diag.index, pd.DatetimeIndex):
            mask = (data.diag.index >= start_date) & (data.diag.index <= end_date)
            filtered_diag = data.diag.loc[mask].copy() if mask.any() else None

    # Filter exposure data
    filtered_exposure = None
    if data.exposure is not None and not data.exposure.empty:
        if isinstance(data.exposure.index, pd.DatetimeIndex):
            mask = (data.exposure.index >= start_date) & (data.exposure.index <= end_date)
            filtered_exposure = data.exposure.loc[mask].copy() if mask.any() else None

    # Return filtered data (keep other fields unchanged)
    return DashboardData(
        weights=filtered_weights,
        budget=data.budget,  # Budget is usually static
        factor_weights=data.factor_weights,  # Factor weights are usually static
        ic=data.ic,  # IC data is usually static
        meta=data.meta,
        diag=filtered_diag,
        exposure=filtered_exposure,
        factor_vectors=data.factor_vectors,
        tag=data.tag,
        base_name=data.base_name,
    )


def load_strategy_dashboard_data(
    strategy_id: str,
    tag: str,
    date_range: Optional[Tuple[str, str]] = None,
) -> DashboardData:
    """Load dashboard data for a strategy.

    Tries multiple locations and base names to handle legacy and new file layouts.
    Applies date range filtering if specified.
    """
    strategy_dir = get_strategy_output_dir(strategy_id)
    root = _project_root()
    output_root = Path(cfg.paths.OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = root / output_root
    
    # Try strategy-specific directory first with strategy_id base name
    data = None
    try:
        data = load_dashboard_data(tag, strategy_dir, strategy_id)
        # Check if we actually got data (not all None)
        if data.weights is not None or data.budget is not None:
            pass  # Success, use this data
        else:
            data = None  # Try fallback
    except Exception:
        data = None
    
    # Fallback: try legacy location with BASE_NAME
    if data is None or (data.weights is None and data.budget is None):
        if output_root.exists() and output_root != strategy_dir:
            base_name = cfg.paths.BASE_NAME
            # Check if files exist with BASE_NAME pattern
            test_path = output_root / f"{base_name}_wide_weights_{tag}.csv"
            if test_path.exists():
                try:
                    data = load_dashboard_data(tag, output_root, base_name)
                except Exception:
                    pass
    
    # Final fallback: try case-insensitive matching in OUTPUT_ROOT
    if data is None or (data.weights is None and data.budget is None):
        if output_root.exists():
            strategy_id_upper = strategy_id.upper()
            # Look for any file matching the pattern (case-insensitive)
            for f in output_root.glob("*_wide_weights_*.csv"):
                fname_upper = f.stem.upper()
                if fname_upper.startswith(f"{strategy_id_upper}_WIDE_WEIGHTS_"):
                    m = TAG_RE.search(f.name)
                    if m and m.group(1) == tag:
                        # Extract base name from filename
                        base_name_from_file = f.name.split("_wide_weights_")[0]
                        try:
                            data = load_dashboard_data(tag, output_root, base_name_from_file)
                            break
                        except Exception:
                            continue
    
    # If still no data, create empty DashboardData
    if data is None:
        data = DashboardData(
            weights=None,
            budget=None,
            factor_weights=None,
            ic=None,
            meta=None,
            diag=None,
            exposure=None,
            factor_vectors=None,
            tag=tag,
            base_name=strategy_id,
        )

    # Apply date range filtering if specified
    if date_range is not None:
        data = _filter_dashboard_data_by_date_range(data, date_range)

    return data


def get_strategy_display_name(strategy_id: str) -> str:
    try:
        from q23.strategies.registry import StrategyRegistry
        strategy = StrategyRegistry.get_instance(strategy_id)
        return strategy.config.display_name
    except Exception:
        return strategy_id


def get_strategy_configs() -> Dict[str, "StrategyConfig"]:
    try:
        from q23.strategies.registry import StrategyRegistry
        return StrategyRegistry.list_configs()
    except ImportError:
        return {}


def load_benchmark_for_comparison(
    benchmark_id: str,
    date_range: Tuple[str, str],
) -> Optional[pd.Series]:
    """Load benchmark return series for comparison overlay.
    
    Returns the portfolio return series from the latest saved benchmark run,
    filtered to the specified date range.
    
    Args:
        benchmark_id: The benchmark strategy ID (e.g., 'benchmark_nys_ew')
        date_range: Tuple of (start_date, end_date) strings
    
    Returns:
        Portfolio return series, or None if benchmark data not available
    """
    from q23.dashboard.analytics import select_return_series
    
    tags = discover_strategy_tags(benchmark_id)
    if not tags:
        return None  # Benchmark hasn't been run
    
    # Load the latest tag (first in sorted list)
    data = load_strategy_dashboard_data(benchmark_id, tags[0], date_range)
    if data.diag is None or data.diag.empty:
        return None
    
    return select_return_series(data.diag)
