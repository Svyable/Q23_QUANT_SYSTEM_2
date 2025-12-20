# Environment Configuration Guide

## Overview

The `run_strategy.py` script now dynamically discovers strategies from the `StrategyRegistry` and uses environment variables to control which strategies run.

## Environment Variables

### Required

- **`API_KEY`**: Your API key for data access
  ```
  API_KEY=your-api-key-here
  ```

### Optional

- **`Q23_TAG`**: Tag for output files (defaults to timestamp if not set)
  ```
  Q23_TAG=2025-12-17
  ```

### Strategy Execution Flags

Strategy flags follow the pattern: `Q23_RUN_<strategy_id>` (uppercase)

#### Legacy (Backward Compatibility)

- **`Q23_RUN_V4`**: Run legacy `StrategyEngine` (uses `cfg.paths.BASE_NAME` and `cfg.paths.OUTPUT_ROOT`)
  ```
  Q23_RUN_V4=false
  ```

#### Registered Strategies

Currently registered strategies (automatically discovered):

1. **`Q23_RUN_NASNYS_V4`**: NASNYS V4 Strategy (24 factors)
   ```
   Q23_RUN_NASNYS_V4=true
   ```

2. **`Q23_RUN_QS23_HYBRID_ALPHA`**: QS23 Hybrid Alpha Strategy
   ```
   Q23_RUN_QS23_HYBRID_ALPHA=false
   ```

3. **`Q23_RUN_Q23_COMPOSER_V1`**: Q23 Composer V1 Strategy (32 factors)
   ```
   Q23_RUN_Q23_COMPOSER_V1=false
   ```

## Example .env File

```bash
# Required: API key for data access
API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Optional: Tag for output files
Q23_TAG=2025-12-17

# Legacy StrategyEngine (backward compatibility)
Q23_RUN_V4=false

# Registered Strategies
Q23_RUN_NASNYS_V4=true
Q23_RUN_QS23_HYBRID_ALPHA=false
Q23_RUN_Q23_COMPOSER_V1=false
```

## Boolean Values

All strategy flags accept these values (case-insensitive):
- `true`, `1`, `yes`, `y`, `on` → Enable
- `false`, `0`, `no`, `n`, `off` → Disable

## Discovering Available Strategies

To see all available strategies programmatically:

```python
from q23.strategies.registry import StrategyRegistry
print(StrategyRegistry.list_ids())
```

Or run the script with no strategies enabled to see available options:

```bash
python run_strategy.py
```

## Migration from Old Format

### Old Format (deprecated)
```bash
Q23_RUN_V4=true
Q23_RUN_NASNYS1010=false
Q23_RUN_Q23LSNEW3=false
```

### New Format
```bash
Q23_RUN_V4=true                    # Legacy StrategyEngine (still supported)
Q23_RUN_NASNYS_V4=true             # New registered strategy
Q23_RUN_QS23_HYBRID_ALPHA=false    # New registered strategy
Q23_RUN_Q23_COMPOSER_V1=false      # New registered strategy
```

## Adding New Strategies

When you add a new strategy to the `StrategyRegistry`, it will automatically:
1. Be discovered by `run_strategy.py`
2. Use the environment variable `Q23_RUN_<strategy_id>` (uppercase)
3. Appear in the help output when no strategies are enabled

No code changes needed in `run_strategy.py` - it's fully dynamic!
