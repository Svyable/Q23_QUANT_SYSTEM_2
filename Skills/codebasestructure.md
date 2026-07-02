# Q23 Quantitative Trading System - Codebase Structure

## Overview

Q23 is a comprehensive quantitative trading platform built in Python, featuring a modular architecture with Streamlit dashboard, multiple trading strategies, and advanced analytics capabilities. The system supports automated strategy execution, real-time data integration, and sophisticated risk management.

## Directory Structure

```
Q23_QUANT_SYSTEM 2/
├── Skills/                          # Documentation and skills
│   ├── Agentic.md                  # Agentic AI integration guide
│   ├── Streamlit.md                # Streamlit dashboard documentation
│   ├── Plotly.md                   # Plotly visualization guide
│   ├── marketstack.md              # Marketstack API documentation
│   └── *.md                        # Other skill documentation
├── src/q23/                        # Main source code
│   ├── dashboard/                  # Streamlit web interface
│   ├── strategies/                 # Trading strategy implementations
│   ├── strategy/                   # Core strategy engine components
│   └── shared/                     # Shared utilities and configuration
├── run_dashboard.py                # Dashboard launcher
├── run_strategy.py                 # Strategy execution runner
├── Run_Q23.command                 # macOS launcher script
├── diagnose_env.py                 # Environment diagnostics
└── *.json                         # Configuration files
```

## Core Architecture Components

### 1. Dashboard Layer (`src/q23/dashboard/`)

**Purpose**: Web-based interface for strategy visualization, analytics, and management.

**Key Files**:
- `app.py` - Main Streamlit application entry point with page routing
- `core.py` - Dashboard data loading and management utilities
- `run_cache.py` - Caching system for dashboard performance

**Subdirectories**:
- `_pages/` - Individual dashboard pages (20+ pages for different views)
- `analytics/` - Data processing and analytics modules
- `components/` - Reusable UI components (charts, insights, styles)

**Architecture Pattern**: Page registry system with error boundaries and session state management.

### 2. Strategy Layer (`src/q23/strategies/`)

**Purpose**: Trading strategy implementations and management.

**Key Files**:
- `base.py` - Abstract base classes for strategies (`StrategyBase`, `StrategyConfig`)
- `registry.py` - Strategy registration and discovery system
- `enabled_strategies.json` - Runtime strategy enablement configuration

**Strategy Types**:
- `nasnys_v4/` - 24-factor IC-weighted strategy (main production strategy)
- `glft_v1/v2/v3/` - GLFT microstructure strategies
- `gpt52v4/` - GPT-based adaptive strategy
- `q23_neural_alpha/` - Neural network strategies
- `q23_composer_v1/v2/` - Multi-model composition strategies
- `benchmarks/` - Benchmark strategy implementations

**Architecture Pattern**: Plugin architecture with registry pattern for dynamic loading.

### 3. Strategy Engine (`src/q23/strategy/`)

**Purpose**: Core execution engine for running trading strategies.

**Key Components**:
- `engine.py` - Main strategy execution orchestrator
- `data_loader.py` - Market data loading with caching and API integration
- `factors.py` - Factor computation library (100+ technical indicators)
- `ic_weighting.py` - Information coefficient weighting system
- `portfolio.py` - Portfolio construction and optimization
- `outputs.py` - Result serialization and file writing

**Data Flow**:
```
Data Loading → Factor Computation → IC Weighting → Portfolio Construction → Output Writing
```

### 4. Shared Components (`src/q23/shared/`)

**Purpose**: Common utilities and configuration management.

**Key Files**:
- `config.py` - Centralized configuration system (`GlobalConfig`, `StrategyConfig`, etc.)
- `math_utils.py` - Mathematical utilities and helper functions

**Configuration Hierarchy**:
1. Environment variables (highest priority)
2. Strategy-specific configs
3. Global defaults (lowest priority)

## Data Architecture

### Data Sources
- **Primary**: QuantConnect/Quantiacs data via `qnt` library
- **Real-time**: Marketstack API for live data supplementation
- **Caching**: NetCDF4 files for performance optimization

### Data Flow Pipeline
```
Raw Market Data → Data Loader → Factor Library → IC Weighting → Portfolio Engine → Output Cache → Dashboard
```

### Key Data Structures
- **xarray.Dataset**: Multi-dimensional market data (time × asset × fields)
- **xarray.DataArray**: Single-dimensional data (time × asset)
- **pandas.DataFrame**: Tabular results and diagnostics

## Execution Patterns

### 1. Strategy Runner (`run_strategy.py`)
```bash
# Command-line execution
python run_strategy.py              # Run enabled strategies
Q23_FORCE_RERUN=true python run_strategy.py  # Force re-run
Q23_DEBUG_RUNS=true python run_strategy.py  # Verbose logging
```

**Features**:
- Freshness detection (skip if today's data exists)
- Parallel benchmark execution
- Comprehensive telemetry and error handling

### 2. Dashboard (`run_dashboard.py`)
```bash
python run_dashboard.py -- --server.port 8502
```

**Features**:
- Multi-page Streamlit application
- Real-time data visualization
- Interactive analytics and what-if scenarios

## Configuration System

### Global Configuration (`src/q23/shared/config.py`)
```python
@dataclass
class GlobalConfig:
    paths: PathConfig
    strategy: StrategyConfig
    dashboard: DashboardConfig
    benchmark: BenchmarkConfig
    tc: TransactionCostConfig
    marketstack: MarketstackConfig
```

### Strategy Configuration
Each strategy has its own config with parameters for:
- Universe selection (exchanges, dates)
- Risk constraints (position limits, leverage)
- Factor definitions and weighting
- Transaction cost models

## Caching and Performance

### Data Caching
- **Market Data**: NetCDF4 files in `data-cache/` directory
- **Strategy Outputs**: Timestamped directories in `outputs/`
- **Dashboard Cache**: Streamlit caching decorators

### Performance Optimizations
- Lazy loading of large datasets
- Incremental data updates via Marketstack
- Parallel strategy execution capabilities
- Memory-efficient xarray operations

## Error Handling and Resilience

### Error Boundaries
- Dashboard pages wrapped with error handling
- Strategy execution with comprehensive exception catching
- Graceful degradation when data sources unavailable

### Data Validation
- Schema validation for market data
- Sanity checks on strategy outputs
- Automatic retry mechanisms for API calls

## Testing and Validation

### Strategy Validation
- Output format validation
- Performance metric sanity checks
- Factor computation verification

### Data Integrity
- Checksum validation for cached data
- Gap detection and filling
- Cross-source data reconciliation

## Deployment and Operations

### Environment Setup
- Conda environment management
- API key configuration via `.env` files
- Path configuration for different machines

### Monitoring and Telemetry
- Marketstack API usage tracking
- Strategy execution performance metrics
- Error logging and alerting

## Integration Points

### External APIs
- **Marketstack**: Real-time data supplementation
- **QuantConnect**: Historical market data
- **Git**: Version control and deployment

### File System Integration
- **NetCDF4**: High-performance data storage
- **JSON**: Configuration management
- **CSV**: Result export and analysis

## Development Workflow

### Strategy Development
1. Implement strategy class inheriting from `StrategyBase`
2. Register with `StrategyRegistry`
3. Add configuration in strategy directory
4. Test via strategy runner
5. Enable in dashboard warehouse

### Dashboard Development
1. Add page module in `_pages/` directory
2. Register page in `app.py` routing
3. Implement with Streamlit components
4. Add to sidebar navigation

### Testing
1. Unit tests for individual components
2. Integration tests for strategy execution
3. Dashboard interaction tests
4. Performance benchmarking

## SOLID Principles Compliance Analysis

### Single Responsibility Principle (SRP) - ⭐⭐⭐⭐⭐ EXCELLENT
Each component has one clear, well-defined responsibility:

**Strategy Layer**:
- `StrategyBase`: Defines strategy execution interface only
- `StrategyRegistry`: Manages strategy discovery and instantiation only
- Individual Strategies: Implement specific trading logic only

**Engine Layer**:
- `StrategyEngine`: Orchestrates execution pipeline only
- `DataLoader`: Handles data loading and normalization only
- `FactorLibrary`: Computes technical indicators only
- `PortfolioConstructor`: Builds portfolios only

**Dashboard Layer**:
- Each page module focuses on one analytical view
- `DashboardCore`: Manages data loading and caching only
- Components handle specific UI concerns only

### Open-Closed Principle (OCP) - ⭐⭐⭐⭐⭐ EXCELLENT
System designed for extension without modification:

**Extensibility Patterns**:
- New strategies added via `@StrategyRegistry.register` decorator
- Dashboard pages added through `PAGE_REGISTRY` dictionary
- Configuration extended through dataclass inheritance
- Factor library supports new indicators without core changes

**Plugin Architecture Benefits**:
- Zero modification of existing code for new features
- Registry pattern enables runtime component discovery
- Configuration-driven behavior changes

### Liskov Substitution Principle (LSP) - ⭐⭐⭐⭐⭐ EXCELLENT
Abstract base classes ensure perfect substitutability:

**Interface Consistency**:
- All `StrategyBase` implementations provide identical interfaces
- `StrategyArtifacts` maintain consistent output structure
- Configuration classes preserve compatible method signatures
- Error handling follows uniform patterns

**Behavioral Guarantees**:
- Any strategy can replace any other strategy
- Dashboard pages are interchangeable in routing system
- Data loaders provide consistent `MarketDataBundle` format

### Interface Segregation Principle (ISP) - ⭐⭐⭐⭐⭐ EXCELLENT
Interfaces are focused, minimal, and client-specific:

**Clean Interfaces**:
- `StrategyBase`: Only essential execution methods (run, get_factors, config)
- Configuration classes expose only relevant parameters
- `MarketDataBundle`: Contains only necessary data fields
- Page render functions have minimal parameter requirements

**Client-Specific Design**:
- Dashboard depends only on data loading interfaces
- Strategies depend only on engine orchestration interfaces
- Analytics components have focused data requirements

### Dependency Inversion Principle (DIP) - ⭐⭐⭐⭐⭐ EXCELLENT
High-level modules depend on abstractions, not concretions:

**Abstraction Layers**:
- Strategies depend on `StrategyBase` abstraction
- Engine depends on configuration interfaces, not concrete configs
- Dashboard depends on data loading abstractions
- All components depend on `GlobalConfig` interface

**Dependency Injection**:
- Registry pattern provides runtime dependency resolution
- Configuration system enables parameter injection
- Factory methods create appropriate concrete implementations

## Architectural Congruence Analysis

### Overall Congruence Rating: ⭐⭐⭐⭐⭐ EXCELLENT

**Pattern Consistency**:
- Registry pattern used uniformly across strategies and dashboard pages
- Error handling standardized with `safe_render` wrapper pattern
- Configuration management consistent through dataclass hierarchy
- Naming conventions maintained throughout (`strategy_id`, `config`, etc.)

**Data Flow Congruence**:
- Consistent xarray/pandas data structure usage
- Standardized metadata propagation between components
- Uniform caching and output writing patterns
- Consistent parameter passing conventions

**Integration Congruence**:
- All components follow dependency injection patterns
- Consistent return value structures (`StrategyArtifacts`, `MarketDataBundle`)
- Standardized exception handling and logging approaches

### Minor Congruence Improvements Identified

1. **Error Message Formats**: Some inconsistency in error message formatting across components
2. **Caching Strategy Consolidation**: Multiple caching approaches could be unified
3. **Parameter Naming**: Some methods use different naming conventions (min_date vs start_date)

## Architecture Quality Assessment

### Strengths
- **Exceptional Modularity**: Clean separation enables independent development and testing
- **High Testability**: Dependency injection enables comprehensive mocking and unit testing
- **Maintainability**: Single responsibility principle enables focused code changes
- **Extensibility**: Plugin architecture supports seamless feature addition
- **Reliability**: Strong abstraction layers prevent cascading failures

### Quality Metrics
- **Cyclomatic Complexity**: Low due to focused, single-purpose functions
- **Coupling**: Loose coupling through abstraction layers and dependency injection
- **Cohesion**: High cohesion within each module and class
- **Abstraction Level**: Consistent and appropriate abstraction throughout

## Future Extensibility

### Strategy Plugins
- Easy addition of new strategies via registry pattern
- Configuration-driven strategy parameters
- Modular factor library for custom indicators

### Analytics Expansion
- Additional risk metrics and visualizations
- Machine learning integration for strategy optimization
- Real-time alerting and notification systems

### API Development
- REST API for external integrations
- WebSocket feeds for live data
- GraphQL interface for complex queries

This SOLID-compliant, highly congruent architecture enables rapid development of new strategies, comprehensive analytics, and scalable deployment while maintaining exceptional code quality and performance optimization.