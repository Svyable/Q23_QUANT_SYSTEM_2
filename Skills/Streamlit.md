# Streamlit Integration in Q23

Streamlit is a powerful open-source framework for building interactive web applications for machine learning and data science. In the Q23 system, Streamlit serves as the primary interface for the trading strategy dashboard and analytics platform.

## Core Concepts

### App Structure
The Q23 dashboard is built using Streamlit's multi-page application structure with custom page routing:

```python
import streamlit as st

# Main app entry point
def main():
    st.set_page_config(
        page_title="Q23 Quant System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Page routing
    pages = {
        "Overview": render_overview_page,
        "Performance": render_performance_page,
        "Risk Analytics": render_risk_page,
        # ... more pages
    }

    selected_page = st.sidebar.selectbox("Navigate", list(pages.keys()))
    pages[selected_page]()
```

### Session State Management
Streamlit's session state is used extensively for maintaining application state:

```python
# Initialize session state
if 'strategy_data' not in st.session_state:
    st.session_state.strategy_data = {}

# Store data across reruns
st.session_state.selected_strategy = st.selectbox(
    "Strategy",
    strategy_names,
    index=get_default_strategy_index()
)
```

## Key Components

### Data Visualization
The dashboard uses multiple visualization libraries through Streamlit:

#### Plotly Integration
```python
import plotly.express as px
import streamlit as st

def create_returns_chart(returns_data):
    fig = px.line(
        returns_data,
        x='date',
        y='returns',
        title='Strategy Returns'
    )

    # Use container width for responsive design
    st.plotly_chart(fig, use_container_width=True)
```

#### Matplotlib Integration
```python
import matplotlib.pyplot as plt

def create_equity_curve(equity_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_data.index, equity_data.values)
    ax.set_title('Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')

    st.pyplot(fig)
```

### Interactive Widgets

#### DataFrames and Tables
```python
import pandas as pd

def display_portfolio_weights(weights_df):
    # Interactive dataframe with sorting and filtering
    st.dataframe(
        weights_df,
        use_container_width=True,
        height=400
    )

    # Detailed view with formatting
    st.data_editor(
        weights_df.style.format("{:.2%}"),
        num_rows="dynamic"
    )
```

#### Metrics Display
```python
def display_performance_metrics(metrics):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.sharpe:.2f}",
            delta=f"{metrics.sharpe_change:.2f}"
        )

    with col2:
        st.metric(
            "Total Return",
            f"{metrics.total_return:.1%}",
            delta=f"{metrics.return_change:.1%}"
        )

    # ... more metrics
```

#### Charts and Visualizations
```python
def create_strategy_comparison_chart(strategies_data):
    # Create comparison chart
    fig = go.Figure()

    for strategy_name, data in strategies_data.items():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.equity,
            mode='lines',
            name=strategy_name
        ))

    fig.update_layout(
        title="Strategy Comparison",
        xaxis_title="Date",
        yaxis_title="Equity"
    )

    st.plotly_chart(fig, use_container_width=True)
```

## Layout and Styling

### Custom CSS
The dashboard uses custom CSS for enhanced styling:

```python
def inject_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #ff4b4b;
    }

    .sidebar-content {
        padding: 1rem 0;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .metric-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
```

### Responsive Layout
```python
def create_responsive_layout():
    # Use columns for responsive design
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Main Chart")
        # Large chart component

    with col2:
        st.subheader("Key Metrics")
        # Metrics sidebar
```

## Performance Optimization

### Caching
Streamlit's caching decorators are used extensively:

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_strategy_data(strategy_id):
    """Load strategy data with caching."""
    return expensive_data_loading_function(strategy_id)

@st.cache_resource
def get_database_connection():
    """Cache database connections."""
    return create_db_connection()
```

### Lazy Loading
```python
def lazy_load_strategy_data():
    """Only load data when needed."""
    if st.button("Load Strategy Data"):
        with st.spinner("Loading data..."):
            data = load_strategy_data()
            st.session_state.data_loaded = True
            st.rerun()
```

## Advanced Features

### Custom Components
```python
import streamlit.components.v1 as components

def custom_metric_component(label, value, delta=None):
    """Create a custom metric component."""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta">{delta}</div>' if delta else ''}
    </div>
    """

    components.html(html, height=100)
```

### File Upload/Download
```python
def handle_file_operations():
    # File upload
    uploaded_file = st.file_uploader("Upload strategy file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        process_uploaded_data(data)

    # File download
    csv_data = strategy_data.to_csv()
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="strategy_data.csv",
        mime="text/csv"
    )
```

## Error Handling

### User-Friendly Error Messages
```python
def safe_data_loading():
    try:
        data = load_data()
        return data
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("Please check your data source and try again.")
        return None
```

### Progress Indicators
```python
def show_progress_during_calculation():
    with st.spinner("Calculating performance metrics..."):
        # Long-running calculation
        metrics = calculate_metrics(data)

    st.success("Calculation complete!")
    display_metrics(metrics)
```

## Configuration and Settings

### Admin Settings
```python
def admin_settings_panel():
    st.sidebar.header("Admin Settings")

    # Theme selection
    theme = st.sidebar.selectbox(
        "Theme",
        ["Light", "Dark", "Auto"],
        index=0
    )

    # Performance settings
    cache_enabled = st.sidebar.checkbox("Enable caching", value=True)
    max_cache_age = st.sidebar.slider("Max cache age (hours)", 1, 24, 6)
```

## Best Practices

### Code Organization
```
dashboard/
├── app.py              # Main app entry point
├── _pages/             # Individual page modules
├── components/         # Reusable UI components
├── analytics/          # Data processing modules
├── core.py             # Shared utilities
└── admin_settings.py   # Configuration management
```

### Performance Tips
1. **Use caching** for expensive operations
2. **Lazy load** data only when needed
3. **Optimize images** and large datasets
4. **Use pagination** for large tables
5. **Implement proper error handling** to prevent app crashes

### UI/UX Guidelines
1. **Consistent styling** across pages
2. **Clear navigation** with sidebar
3. **Responsive design** for different screen sizes
4. **Loading states** for long operations
5. **Helpful error messages** and validation

## Migration Notes

### Width Parameter Changes (2024-2025)
As noted in the deprecation notice, `use_container_width` is being replaced:

```python
# Old (deprecated)
st.plotly_chart(fig, use_container_width=True)

# New (recommended)
st.plotly_chart(fig, use_container_width=True)  # Still works but deprecated

# Preferred approach
st.plotly_chart(fig, width="stretch")  # New parameter
```

### Future Updates
- Monitor Streamlit release notes for breaking changes
- Test dashboard functionality after major updates
- Update dependencies regularly for security and features

## Integration with Q23 Components

The Streamlit dashboard integrates with all Q23 components:

- **Strategy Engine**: Real-time strategy evaluation
- **Data Loading**: Interactive data exploration
- **Analytics**: Performance and risk analysis
- **Portfolio Management**: Position tracking and rebalancing

This integration provides a comprehensive web interface for quantitative trading strategy development and analysis.