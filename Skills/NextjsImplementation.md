# Q23 Quant System - Next.js Implementation Guide

## 🧠 ProTips & Reminders for AI Implementation

> **Hey there, lovable AI builder!** This guide contains crucial insights for translating the Q23 Python/Streamlit quant system into a modern Next.js application. Pay special attention to these patterns - they're battle-tested from the original codebase.

## 🏗️ Architecture Translation

### Component Hierarchy Mapping
```
Python Streamlit Pages → Next.js App Router Structure
├── _pages/_overview.py → app/(dashboard)/overview/page.tsx
├── _pages/_strategy_comparison.py → app/(dashboard)/strategy-comparison/page.tsx
├── _pages/_risk_analytics.py → app/(dashboard)/risk-analytics/page.tsx
└── _pages/_factors.py → app/(dashboard)/factors/page.tsx
```

**ProTip**: Each Streamlit page becomes a Next.js route with its own loading states, error boundaries, and data fetching.

### State Management Strategy
```typescript
// Use Zustand for global trading state (not Context API)
interface TradingState {
  selectedStrategy: string | null;
  dateRange: { start: string; end: string };
  portfolioWeights: PortfolioWeights | null;
  factorExposures: FactorExposures | null;
  // ... other global state
}
```

**Reminder**: Trading dashboards need global state for strategy selection, date ranges, and portfolio data. Zustand scales better than Context for complex financial state.

## 📊 Data Handling & Real-time Updates

### Market Data Architecture
```typescript
// Data fetching pattern for financial time series
const useMarketData = (symbol: string, timeframe: Timeframe) => {
  return useSWR(
    `/api/market-data/${symbol}?timeframe=${timeframe}`,
    fetcher,
    {
      refreshInterval: 60000, // 1 minute for live data
      revalidateOnFocus: false,
      dedupingInterval: 30000, // 30 seconds
    }
  );
};
```

**ProTip**: Use SWR for market data with appropriate refresh intervals. Financial data changes frequently but doesn't need instant updates for most analytics.

### Caching Strategy
```typescript
// Implement multi-layer caching
const CACHE_CONFIG = {
  // Browser cache for static strategy metadata
  static: { ttl: 24 * 60 * 60 * 1000 }, // 24 hours

  // SW cache for frequently accessed data
  dynamic: { ttl: 15 * 60 * 1000 }, // 15 minutes

  // Memory cache for session data
  session: { ttl: 60 * 60 * 1000 }, // 1 hour
};
```

**Reminder**: Financial data caching is critical. Cache strategy outputs, factor computations, and historical data aggressively while ensuring real-time data freshness.

## 🎨 Component Patterns for Quant Analytics

### Portfolio Weight Visualization
```typescript
// TreeMap component for portfolio weights
const PortfolioTreeMap = ({ weights }: { weights: PortfolioWeights }) => {
  const processedData = useMemo(() => {
    return Object.entries(weights)
      .map(([symbol, weight]) => ({
        name: symbol,
        value: Math.abs(weight),
        color: weight > 0 ? 'green' : 'red',
      }))
      .sort((a, b) => b.value - a.value);
  }, [weights]);

  return (
    <ResponsiveContainer>
      <Treemap
        data={processedData}
        dataKey="value"
        aspectRatio={4/3}
        stroke="#fff"
        fill="#8884d8"
      >
        <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
      </Treemap>
    </ResponsiveContainer>
  );
};
```

**ProTip**: Use D3-based libraries (Recharts, Nivo) for quant visualizations. TreeMaps work great for portfolio composition, heatmaps for factor exposures.

### Factor Exposure Heatmap
```typescript
// Factor exposure matrix visualization
const FactorHeatmap = ({ exposures }: { exposures: FactorExposures }) => {
  return (
    <div className="overflow-x-auto">
      <table className="factor-heatmap">
        <thead>
          <tr>
            <th>Asset</th>
            {exposures.factors.map(factor => (
              <th key={factor}>{factor}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {exposures.assets.map(asset => (
            <tr key={asset}>
              <td className="font-medium">{asset}</td>
              {exposures.factors.map(factor => {
                const exposure = exposures.data[asset][factor];
                return (
                  <td
                    key={factor}
                    className={getHeatmapColor(exposure)}
                  >
                    {exposure.toFixed(3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

**Reminder**: Financial data tables need horizontal scrolling, custom formatting, and conditional styling based on values.

## 🔄 Real-time Data Management

### WebSocket Integration for Live Data
```typescript
// Live strategy execution status
const useStrategyExecutionStatus = (strategyId: string) => {
  const [status, setStatus] = useState<ExecutionStatus>('idle');

  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/strategy/${strategyId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data.status);
    };

    return () => ws.close();
  }, [strategyId]);

  return status;
};
```

**ProTip**: Use WebSockets for strategy execution monitoring, live portfolio updates, and real-time risk metrics. HTTP polling is insufficient for trading dashboards.

### Data Synchronization Strategy
```typescript
// Handle data synchronization across tabs
const useDataSync = () => {
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const syncData = useCallback(async () => {
    // Fetch latest data from all endpoints
    const [weights, returns, exposures] = await Promise.all([
      fetch('/api/portfolio/weights'),
      fetch('/api/portfolio/returns'),
      fetch('/api/factors/exposures'),
    ]);

    // Update global state
    setLastUpdate(new Date());
  }, []);

  return { lastUpdate, syncData };
};
```

**Reminder**: Multiple dashboard tabs need coordinated data updates. Use a global sync mechanism to prevent stale data across views.

## 🚨 Error Handling for Financial Data

### Financial Data Validation
```typescript
// Validate financial data integrity
const validateFinancialData = (data: unknown): data is FinancialData => {
  if (!data || typeof data !== 'object') return false;

  // Check required fields
  const required = ['weights', 'returns', 'dates'];
  if (!required.every(key => key in data)) return false;

  // Validate weight sums (should be ~1.0 for long-only)
  const weights = data.weights as number[];
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  if (Math.abs(totalWeight - 1.0) > 0.01) {
    console.warn(`Portfolio weights sum to ${totalWeight}, expected ~1.0`);
  }

  return true;
};
```

**ProTip**: Financial data requires strict validation. Check for NaN values, weight sums, date continuity, and statistical reasonableness.

### Graceful Degradation
```typescript
// Fallback UI for data loading failures
const DataWithFallback = ({ children, fallback }: DataFallbackProps) => {
  const { data, error, isLoading } = useFinancialData();

  if (error) {
    return (
      <ErrorFallback
        error={error}
        retry={() => window.location.reload()}
        fallback={fallback}
      />
    );
  }

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  return children(data);
};
```

**Reminder**: Financial dashboards must never show blank screens. Always provide meaningful fallbacks and loading states.

## ⚡ Performance Optimization

### Virtualization for Large Datasets
```typescript
// Virtualize large factor exposure tables
const VirtualizedFactorTable = ({ data }: { data: FactorData[] }) => {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: data.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 35,
    overscan: 5,
  });

  return (
    <div ref={parentRef} className="h-96 overflow-auto">
      <div
        style={{ height: `${rowVirtualizer.getTotalSize()}px` }}
        className="relative w-full"
      >
        {rowVirtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            className="absolute top-0 left-0 w-full"
            style={{ transform: `translateY(${virtualItem.start}px)` }}
          >
            <FactorRow data={data[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
};
```

**ProTip**: Financial datasets can have thousands of rows. Use virtualization (react-window, tanstack/virtual) for tables and charts.

### Memoization Strategy
```typescript
// Memoize expensive financial calculations
const usePortfolioAnalytics = (weights: Weights, returns: Returns) => {
  return useMemo(() => {
    const sharpe = calculateSharpeRatio(returns);
    const vol = calculateVolatility(returns);
    const maxDrawdown = calculateMaxDrawdown(returns);
    const factorBetas = calculateFactorBetas(weights, returns);

    return { sharpe, vol, maxDrawdown, factorBetas };
  }, [weights, returns]); // Only recalculate when data changes
};
```

**Reminder**: Financial calculations are expensive. Memoize everything possible and use structural equality checks for dependencies.

## 🔧 Development Workflow Reminders

### Type Safety First
```typescript
// Define strict types for financial data
interface PortfolioWeights {
  [symbol: string]: number; // Weight between -1 and 1
}

interface FactorExposure {
  factor: string;
  exposure: number; // Usually between -2 and 2
  tStat: number; // Statistical significance
}

interface StrategyArtifacts {
  weights: PortfolioWeights;
  factorWeights: FactorExposure[];
  compositeScore: number[];
  metadata: StrategyMetadata;
}
```

**ProTip**: Financial data types are complex. Use TypeScript interfaces extensively and validate data at runtime.

### Testing Strategy
```typescript
// Test financial calculations thoroughly
describe('Portfolio Analytics', () => {
  it('calculates Sharpe ratio correctly', () => {
    const returns = [0.01, 0.02, -0.01, 0.015];
    const sharpe = calculateSharpeRatio(returns);
    expect(sharpe).toBeCloseTo(1.23, 2);
  });

  it('handles edge cases', () => {
    const emptyReturns: number[] = [];
    expect(() => calculateSharpeRatio(emptyReturns)).toThrow();
  });
});
```

**Reminder**: Financial calculations must be tested extensively. Edge cases like zero returns, NaN values, and extreme outliers must be handled.

## 🎯 Key Reminders for Lovable AI

### 1. **Financial Data is Sacred**
- Never show placeholder data in production
- Always validate data integrity
- Provide clear error states for data failures
- Cache aggressively but invalidate correctly

### 2. **Performance is Critical**
- Financial dashboards load large datasets
- Users expect sub-second interactions
- Optimize bundle size - tree shake aggressively
- Use virtualization for large lists/tables

### 3. **Real-time is Expected**
- WebSockets for live updates
- Proper reconnection handling
- Background sync for data consistency
- Graceful degradation when offline

### 4. **Security Matters**
- API keys should never reach client
- Validate all financial calculations server-side
- Rate limit expensive operations
- Log all strategy executions for audit

### 5. **UX is Everything**
- Loading states for every async operation
- Clear error messages users can understand
- Progressive loading of complex charts
- Responsive design for trading desks

### 6. **State Management is Complex**
- Global state for strategy selection
- Local state for UI interactions
- Server state for financial data
- Proper state synchronization

Remember: You're building a tool that traders will use to make real financial decisions. Quality, reliability, and performance aren't nice-to-haves - they're requirements! 🚀📈