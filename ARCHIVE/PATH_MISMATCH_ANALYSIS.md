# Path Mismatch Analysis & Fix

## Problem Summary

The dashboard was showing an error:
```
No tags found. Expected files like nasnys_v4_wide_weights_<tag>.csv under:
/Users/svenbenson/Q23_QUANT_SYSTEM 2/outputs/NASNYS_V4/nasnys_v4
Run strategies first.
```

However, CSV files **do exist** in the repository, but in a different location and with different naming.

## Root Cause

There are **two different execution paths** in the codebase that write files differently:

### 1. Legacy Path (`run_strategy.py` → `StrategyEngine`)
- **Output Directory**: `outputs/NASNYS_V4/` (from `cfg.paths.OUTPUT_ROOT`)
- **Base Name**: `NASNYS_V4` (from `cfg.paths.BASE_NAME`)
- **File Pattern**: `NASNYS_V4_wide_weights_<tag>.csv`
- **Actual Files**: Located at `outputs/NASNYS_V4/NASNYS_V4_wide_weights_2025-12-11.csv`

### 2. New Strategy Path (`nasnys_v4` strategy → `StrategyBase`)
- **Output Directory**: `outputs/NASNYS_V4/nasnys_v4/` (from `get_output_dir()`)
- **Base Name**: `nasnys_v4` (from `get_base_name()`)
- **File Pattern**: `nasnys_v4_wide_weights_<tag>.csv`
- **Expected Files**: Would be at `outputs/NASNYS_V4/nasnys_v4/nasnys_v4_wide_weights_<tag>.csv`

### Dashboard Expectations
The dashboard was using the **new strategy path** logic:
- Looking in: `outputs/NASNYS_V4/nasnys_v4/`
- Looking for: `nasnys_v4_wide_weights_*.csv`

But files were created using the **legacy path**:
- Actually in: `outputs/NASNYS_V4/`
- Actually named: `NASNYS_V4_wide_weights_*.csv`

## Solution

Fixed three functions in `src/q23/dashboard/core.py`:

### 1. `get_strategy_output_dir(strategy_id)`
- **Before**: Always appended `strategy_id` to `OUTPUT_ROOT`
- **After**: Checks if `OUTPUT_ROOT` already ends with a directory matching `strategy_id` (case-insensitive)
  - If yes → returns `OUTPUT_ROOT` directly (legacy case)
  - If no → appends `strategy_id` (new case)

### 2. `discover_strategy_tags(strategy_id)`
- **Before**: Only checked strategy-specific directory with `strategy_id` base name
- **After**: Checks multiple locations:
  1. Strategy-specific directory (`outputs/NASNYS_V4/nasnys_v4/`) with `strategy_id` base name
  2. Legacy location (`outputs/NASNYS_V4/`) with `BASE_NAME` base name
  3. Case-insensitive matching for both locations

### 3. `load_strategy_dashboard_data(strategy_id, tag)`
- **Before**: Only tried strategy-specific directory with `strategy_id` base name
- **After**: Tries multiple locations and base names in order:
  1. Strategy-specific directory with `strategy_id` base name
  2. Legacy location (`OUTPUT_ROOT`) with `BASE_NAME` base name
  3. Case-insensitive matching in `OUTPUT_ROOT` to find files matching strategy

### 4. Error Message Enhancement (`app.py`)
- **Before**: Generic error message
- **After**: Shows what files were actually found in the directory for better debugging

## Testing

The fix handles these scenarios:
- ✅ Legacy files: `outputs/NASNYS_V4/NASNYS_V4_wide_weights_*.csv`
- ✅ New strategy files: `outputs/NASNYS_V4/nasnys_v4/nasnys_v4_wide_weights_*.csv`
- ✅ Case-insensitive matching
- ✅ Multiple strategies with different layouts

## Files Modified

1. `src/q23/dashboard/core.py` - Fixed path resolution and file discovery
2. `src/q23/dashboard/app.py` - Improved error messages

## Next Steps

The dashboard should now correctly discover and load files created by either execution path. If you run strategies using the new `nasnys_v4` strategy class, files will be written to the new location, and the dashboard will find them. If you run using the legacy `StrategyEngine`, files will be in the legacy location, and the dashboard will also find them.
