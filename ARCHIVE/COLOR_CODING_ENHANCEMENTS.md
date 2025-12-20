# Color Coding Enhancement Plan

## ✅ Completed Enhancements

### 1. Regime Analysis Page (`_elite_analytics.py`)
- **Status**: ✅ Complete
- **Enhancements**:
  - Color-coded regime timeline with background areas (crisis=red, normal=green, trend=purple, calm=blue)
  - Performance by regime bar charts with coordinated colors
  - Time series visualization with regime overlays

### 2. Sector Analysis Page (`_risk_analytics.py`)
- **Status**: ✅ Complete + Enhanced
- **Enhancements**:
  - Fixed `AttributeError: 'DashboardData' object has no attribute 'returns'`
  - Created `SECTOR_COLORS` dictionary for coordinated sector colors
  - Added `_get_sector_color()` helper function with fallback logic
  - Created `_create_sector_bar_chart()` with Plotly and coordinated colors
  - Created `_create_sector_timeseries_chart()` with coordinated line colors
  - Sector allocation bar chart now uses coordinated colors matching time series
  - Refactored following SOLID principles with focused helper functions
  
- **NEW: Sector Health & Risk Analysis** 🎯
  - **`_compute_sector_health_metrics()`**: Computes comprehensive sector health metrics:
    - Exposure-weighted risk contribution
    - Sector concentration risk
    - Risk-adjusted exposure
    - Sector health score (0-100 composite metric)
    - Sector volatility estimates
  - **`_create_sector_health_chart()`**: Dual-panel chart showing:
    - Sector Health Score (left panel)
    - Exposure-Weighted Risk Contribution (right panel)
    - Coordinated colors matching sector palette
  - **`_create_sector_risk_heatmap()`**: Risk map visualization:
    - Scatter plot: Exposure % vs Risk Contribution %
    - Bubble size represents risk contribution
    - Color-coded by sector
    - Interactive hover with detailed metrics
  - **Key Insights Panel**: Displays:
    - Top 3 Risk Contributors
    - Healthiest Sectors
    - Best Risk-Adjusted Exposure
  - **Metrics Table**: Color-coded health scores:
    - Green (≥70): Healthy
    - Orange (50-70): Moderate
    - Red (<50): At Risk

### 3. Rank IC Quintile Analysis (`_elite_analytics.py`)
- **Status**: ✅ Complete
- **Enhancements**:
  - Quintile bar chart uses gradient colors (Q1=red → Q5=green)
  - Visual progression from low to high signal strength

## 🎯 Future Enhancement Opportunities

### High Priority

#### 1. Factor Weights Page (`_factors.py`)
- **Current**: Line charts for factor weights over time
- **Enhancement**: Add coordinated color palette for factors
- **Implementation**: Create `FACTOR_COLORS` mapping, use consistent colors across all factor visualizations
- **Files**: `src/q23/dashboard/_pages/_factors.py`

#### 2. Risk Attribution Page (`_risk_analytics.py`)
- **Current**: Uses `create_risk_contribution_chart()` which may have colors
- **Enhancement**: Ensure factor risk contributions use coordinated colors matching factor weights page
- **Files**: `src/q23/dashboard/_pages/_risk_analytics.py`

#### 3. Correlation Analysis Page (`_risk_analytics.py`)
- **Current**: Bar charts for current factor correlations
- **Enhancement**: Use coordinated factor colors for correlation bars
- **Files**: `src/q23/dashboard/_pages/_risk_analytics.py`

### Medium Priority

#### 4. Performance Page (`_performance.py`)
- **Current**: Various performance metrics
- **Enhancement**: If bar charts exist, add coordinated colors
- **Files**: `src/q23/dashboard/_pages/_performance.py`

#### 5. Weights Page (`_weights.py`)
- **Current**: Portfolio weights visualization
- **Enhancement**: If bar charts for top holdings exist, use coordinated colors
- **Files**: `src/q23/dashboard/_pages/_weights.py`

#### 6. Overview Page (`_overview.py`)
- **Current**: KPI cards and summary charts
- **Enhancement**: Ensure consistency with other pages if bar charts are added
- **Files**: `src/q23/dashboard/_pages/_overview.py`

### Low Priority

#### 7. Strategy Comparison Page (`_strategy_comparison.py`)
- **Current**: Multi-strategy comparison charts
- **Enhancement**: Consistent color scheme for strategies across all comparisons
- **Files**: `src/q23/dashboard/_pages/_strategy_comparison.py`

## 📋 Implementation Guidelines

### Color Palette Standards

1. **Sectors**: Use `SECTOR_COLORS` dictionary
   - Technology: Purple (#9b59b6)
   - Healthcare: Orange (#e67e22)
   - Financials: Red (#e74c3c)
   - Consumer: Blue (#3498db)
   - Industrials: Teal (#1abc9c)
   - Energy: Green (#2ecc71)

2. **Regimes**: Use `REGIME_COLORS` dictionary
   - Crisis: Red (#e74c3c)
   - Normal: Green (#2ecc71)
   - Trend: Purple (#9b59b6)
   - Calm: Blue (#3498db)

3. **Quintiles/Gradients**: Use gradient from red (low) to green (high)
   - Q1: Red (#e74c3c)
   - Q2: Orange (#f39c12)
   - Q3: Yellow (#f1c40f)
   - Q4: Green (#2ecc71)
   - Q5: Dark Green (#27ae60)

4. **Factors**: Create `FACTOR_COLORS` dictionary (to be defined)
   - Use distinct colors for each factor
   - Maintain consistency across pages

### Helper Function Pattern

```python
def _create_[entity]_bar_chart(
    data: pd.Series,
    title: str = "...",
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a bar chart with coordinated colors."""
    # Implementation using Plotly with coordinated colors
    pass
```

### SOLID Principles

- **Single Responsibility**: Each helper function does one thing
- **Open/Closed**: Extendable color mappings without modifying core functions
- **Dependency Inversion**: Functions depend on abstractions (DataFrames, Series)

## 🔍 Code Review Checklist

When adding color coding to a new page:

- [ ] Create color mapping dictionary at module level
- [ ] Create helper function for getting colors with fallback
- [ ] Create Plotly bar chart helper function
- [ ] Use consistent dark theme styling
- [ ] Add fallback to Streamlit charts if Plotly unavailable
- [ ] Ensure colors match across related visualizations
- [ ] Test with various data scenarios
- [ ] Document color choices

## 📝 Notes

- Always maintain backward compatibility
- Use Plotly when available, fallback to Streamlit charts
- Keep color palettes consistent across related pages
- Test with edge cases (empty data, single item, etc.)
