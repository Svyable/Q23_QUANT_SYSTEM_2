# Strategy Runner Refactoring Summary

## Changes Made

### 1. Dynamic Strategy Discovery

**Before**: Hardcoded strategy flags in `run_strategy.py`
- `Q23_RUN_V4` (legacy StrategyEngine - DEPRECATED and REMOVED)
- `Q23_RUN_NASNYS1010` (legacy stub)
- `Q23_RUN_Q23LSNEW3` (legacy stub)

**After**: Automatically discovers strategies from `StrategyRegistry`
- `Q23_RUN_<strategy_id>` for each registered strategy
- Legacy StrategyEngine (`Q23_RUN_V4`) has been fully removed

### 2. Environment Variable Naming Convention

Environment variables now match strategy IDs (uppercase):
- `nasnys_v4` → `Q23_RUN_NASNYS_V4`
- `qs23_hybrid_alpha` → `Q23_RUN_QS23_HYBRID_ALPHA`
- `q23_composer_v1` → `Q23_RUN_Q23_COMPOSER_V1`

### 3. Current Registered Strategies

The following strategies are automatically discovered:

1. **nasnys_v4** - NASNYS V4 (24 Factors)
   - Env var: `Q23_RUN_NASNYS_V4`
   - Display name: "NASNYS V4 (24 Factors)"

2. **qs23_hybrid_alpha** - QS23 Hybrid Alpha
   - Env var: `Q23_RUN_QS23_HYBRID_ALPHA`
   - Display name: (from config)

3. **q23_composer_v1** - Q23 Composer V1 (32 Factors)
   - Env var: `Q23_RUN_Q23_COMPOSER_V1`
   - Display name: "Q23 COMPOSER v1 (32 Factors)"

### 4. Backward Compatibility

- Legacy `StrategyEngine` mode (`Q23_RUN_V4`) has been fully deprecated and removed
- All strategy execution now goes through the `StrategyRegistry` system
- Old environment variables (`Q23_RUN_NASNYS1010`, `Q23_RUN_Q23LSNEW3`) are no longer used
- Script provides helpful warnings if no strategies are enabled

### 5. Files Modified

1. **`run_strategy.py`**
   - Refactored to dynamically discover strategies
   - Uses `StrategyRegistry.list_ids()` to get available strategies
   - Converts strategy IDs to environment variable names automatically
   - Improved error handling and user feedback

2. **`Run_Q23.command`**
   - Updated to remove hardcoded legacy strategy flags
   - Now relies on `.env` file for strategy configuration

3. **`ENV_CONFIGURATION.md`** (new)
   - Documentation for environment variable configuration
   - Example `.env` file structure
   - Migration guide from old format

## Migration Guide

### Update Your .env File

**Old format** (deprecated and removed):
```bash
Q23_RUN_V4=true
Q23_RUN_NASNYS1010=false
Q23_RUN_Q23LSNEW3=false
```

**New format**:
```bash
# Registered strategies (use enabled_strategies.json via Strategy Warehouse - RECOMMENDED)
# Legacy environment variables are deprecated but still work:
Q23_RUN_NASNYS_V4=true
Q23_RUN_QS23_HYBRID_ALPHA=false
Q23_RUN_Q23_COMPOSER_V1=false
```

**Note**: `Q23_RUN_V4` (legacy StrategyEngine) has been fully removed. All strategies must be registered in the StrategyRegistry.

### Adding New Strategies

When you add a new strategy to `StrategyRegistry`:
1. Register it with `@StrategyRegistry.register` decorator
2. It will automatically appear in the discovery
3. Use `Q23_RUN_<strategy_id>` (uppercase) in `.env`
4. No code changes needed in `run_strategy.py`!

## Benefits

1. **Maintainability**: No need to update `run_strategy.py` when adding strategies
2. **Consistency**: Environment variables match strategy IDs
3. **Discoverability**: Script shows available strategies if none are enabled
4. **Flexibility**: Easy to enable/disable strategies via `.env` file

## Testing

To verify everything works:

```bash
# See available strategies
python -c "from q23.strategies.registry import StrategyRegistry; print(StrategyRegistry.list_ids())"

# Run with no strategies enabled (shows help)
python run_strategy.py

# Run a specific strategy
export Q23_RUN_NASNYS_V4=true
python run_strategy.py
```

## Next Steps

1. Update your `.env` file with the new variable names
2. Remove old unused variables (`Q23_RUN_NASNYS1010`, `Q23_RUN_Q23LSNEW3`)
3. Test running strategies with the new configuration
4. Add new strategies to `StrategyRegistry` as needed - they'll be automatically discovered!
