# Run Logic Refactor Plan

## Problem Analysis

### Current Issues

1. **Stale Run Detection**: GLFTv1 has a run from `20251219_112716` but wasn't re-run on `20251220`
   - Freshness check compares tag prefix (YYYYMMDD) with today's date
   - If tag is `20251219_112716`, prefix is `20251219` which doesn't match `20251220`
   - **Expected behavior**: Should detect as stale and re-run
   - **Actual behavior**: May be skipping due to other issues (enabled status, tag discovery, etc.)

2. **Benchmarks Not Auto-Run**: All benchmarks are disabled in `enabled_strategies.json`
   - New indices (SP500, NAS100) are disabled
   - Benchmarks are computed on-demand in dashboard, but not persisted
   - No mechanism to ensure benchmarks are fresh when strategies run

3. **Missing Dependency Management**: 
   - Strategies may depend on benchmark data for comparison
   - No automatic trigger to run benchmarks when strategies run
   - Benchmarks should be fresh before/after strategy runs

4. **Tag Format Inconsistencies**:
   - Tags may have different formats (YYYYMMDD_HHMMSS, YYYY-MM-DD, etc.)
   - Freshness check only looks at prefix, may miss edge cases
   - No validation of tag format

5. **Limited Debugging**:
   - Hard to understand why strategies are skipped
   - No detailed logging of freshness checks
   - No visibility into tag discovery process

## Solution Design

### 1. Enhanced Freshness Detection

**Current Logic**:
```python
is_fresh = latest_tag.startswith(today_prefix)
```

**Enhanced Logic**:
- Parse tag to extract date component (handle multiple formats)
- Compare dates (not just string prefix)
- Handle timezone issues
- Add debug logging

**Implementation**:
```python
def _parse_tag_date(tag: str) -> Optional[datetime]:
    """Parse date from tag, handling multiple formats."""
    formats = [
        "%Y%m%d_%H%M%S",  # 20251219_112716
        "%Y%m%d",         # 20251219
        "%Y-%m-%d_%H%M%S", # 2025-12-19_112716
        "%Y-%m-%d",       # 2025-12-19
    ]
    for fmt in formats:
        try:
            return datetime.strptime(tag[:len(fmt.replace("_%H%M%S", "").replace("-", ""))], fmt.replace("-", "").replace("_%H%M%S", ""))
        except:
            continue
    return None

def _is_fresh_run(tag: str, today: datetime) -> bool:
    """Check if tag represents a run from today."""
    tag_date = _parse_tag_date(tag)
    if tag_date is None:
        return False
    return tag_date.date() == today.date()
```

### 2. Auto-Run Benchmarks

**Strategy**:
- When strategies run, check if benchmarks are fresh
- If benchmarks are stale or missing, auto-enable and run them
- Ensure new indices (SP500, NAS100) are included

**Implementation**:
```python
def _ensure_benchmarks_fresh(
    today_prefix: str,
    required_benchmarks: Optional[List[str]] = None,
) -> None:
    """Ensure benchmark strategies are fresh.
    
    Args:
        today_prefix: Today's date as YYYYMMDD
        required_benchmarks: List of benchmark IDs to ensure (None = all)
    """
    from q23.strategies.registry import StrategyRegistry
    
    if required_benchmarks is None:
        # Get all benchmark strategies
        all_ids = StrategyRegistry.list_ids()
        required_benchmarks = [s for s in all_ids if s.startswith("benchmark_")]
    
    for bench_id in required_benchmarks:
        is_fresh, latest_tag = _has_fresh_run(bench_id, today_prefix)
        if not is_fresh:
            print(f">> Auto-running benchmark {bench_id} (stale/missing: {latest_tag})")
            _run_strategy(bench_id, tag=None)
```

**Integration Points**:
- After all regular strategies complete
- Before strategy comparison in dashboard
- As a separate phase in `run_strategy.py`

### 3. Benchmark Dependency Management

**New Function**:
```python
def _get_benchmark_dependencies(strategy_id: str) -> List[str]:
    """Get list of benchmark IDs that should be fresh for this strategy.
    
    Returns:
        List of benchmark strategy IDs (e.g., ['benchmark_sp500_ew', 'benchmark_nas100_mc'])
    """
    # Default: ensure all benchmarks are fresh
    # Can be customized per strategy in the future
    from q23.strategies.registry import StrategyRegistry
    all_ids = StrategyRegistry.list_ids()
    return [s for s in all_ids if s.startswith("benchmark_")]
```

**Run Order**:
1. Check which strategies need to run
2. For each strategy, check benchmark dependencies
3. Run stale/missing benchmarks first
4. Run regular strategies
5. Optionally re-run benchmarks after (to ensure latest data)

### 4. Tag Format Validation

**New Function**:
```python
def _normalize_tag(tag: str) -> str:
    """Normalize tag to standard format: YYYYMMDD_HHMMSS."""
    # Parse existing format
    tag_date = _parse_tag_date(tag)
    if tag_date is None:
        # Use current timestamp
        tag_date = datetime.now()
    
    # Format as YYYYMMDD_HHMMSS
    return tag_date.strftime("%Y%m%d_%H%M%S")
```

**Usage**:
- When creating new tags, use normalized format
- When comparing tags, parse and compare dates

### 5. Enhanced Logging

**Add Debug Mode**:
```python
DEBUG_RUNS = _str2bool(os.environ.get("Q23_DEBUG_RUNS", "false"))

def _log_freshness_check(strategy_id: str, is_fresh: bool, latest_tag: Optional[str], today_prefix: str):
    """Log freshness check details."""
    if DEBUG_RUNS:
        status = "FRESH" if is_fresh else "STALE"
        print(f"  [DEBUG] {strategy_id}: {status} (tag: {latest_tag}, today: {today_prefix})")
```

**Enhanced Status Output**:
```python
print(f"    {strategy_id}: {status_str}")
if DEBUG_RUNS and latest_tag:
    tag_date = _parse_tag_date(latest_tag)
    if tag_date:
        age_days = (datetime.now().date() - tag_date.date()).days
        print(f"      [DEBUG] Tag age: {age_days} days, Date: {tag_date.date()}")
```

## Implementation Plan

### Phase 1: Core Enhancements
1. ✅ Add `_parse_tag_date()` function
2. ✅ Enhance `_has_fresh_run()` to use date parsing
3. ✅ Add `_normalize_tag()` for consistent tag format
4. ✅ Add debug logging mode

### Phase 2: Benchmark Auto-Run
1. ✅ Add `_ensure_benchmarks_fresh()` function
2. ✅ Integrate into `run_strategy.py` main loop
3. ✅ Add configuration for which benchmarks to auto-run
4. ✅ Ensure new indices (SP500, NAS100) are included

### Phase 3: Dependency Management
1. ✅ Add `_get_benchmark_dependencies()` function
2. ✅ Update run order to run benchmarks first
3. ✅ Add option to skip benchmark auto-run (Q23_SKIP_BENCHMARKS)

### Phase 4: Testing & Validation
1. ✅ Test freshness detection with various tag formats
2. ✅ Test benchmark auto-run logic
3. ✅ Verify new indices are included
4. ✅ Test edge cases (missing tags, corrupted files, etc.)

## Configuration Options

### Environment Variables

- `Q23_DEBUG_RUNS=true`: Enable detailed logging of freshness checks
- `Q23_AUTO_RUN_BENCHMARKS=true`: Auto-run benchmarks (default: true)
- `Q23_SKIP_BENCHMARKS=false`: Skip benchmark auto-run (default: false)
- `Q23_BENCHMARK_INDICES=SP500,NAS100`: Comma-separated list of indices to ensure (default: all)

### enabled_strategies.json

Benchmarks can remain disabled in JSON config. The auto-run logic will temporarily enable them, run them, then restore their disabled status (or leave enabled if they were already enabled).

## Migration Notes

- Existing tags will continue to work (backward compatible)
- New tags will use normalized format (YYYYMMDD_HHMMSS)
- Freshness checks will be more robust (handle multiple formats)
- Benchmarks will auto-run unless explicitly disabled

## Expected Behavior After Refactor

1. **GLFTv1 from 20251219**: Will be detected as stale on 20251220 and re-run
2. **Benchmarks**: Will auto-run when strategies run, ensuring fresh data
3. **New Indices**: SP500 and NAS100 benchmarks will be included in auto-runs
4. **Logging**: Clear visibility into why strategies are skipped/run
5. **Robustness**: Handles edge cases (timezone, tag formats, missing files)
