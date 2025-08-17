import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="Stock Comparison Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    /* Mobile-friendly styles */
    @media (max-width: 640px) {
        .stApp {
            padding: 1rem 0.5rem;
        }
        
        /* Make charts more visible on mobile */
        .js-plotly-plot {
            min-height: 300px;
        }
        
        /* Improve button and input visibility */
        .stButton>button {
            width: 100%;
            margin: 0.5rem 0;
            min-height: 44px;
        }
        
        .stTextInput>div>div>input {
            min-height: 44px;
        }
        
        /* Better spacing for mobile */
        .row-widget {
            margin-bottom: 1rem;
        }
    }
    
    /* General improvements */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Custom button styles */
    .date-button {
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
    }
    
    /* Metrics styling */
    .metric-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Comparison")
st.markdown("""
Compare any two stocks or indices with advanced analysis tools. Enter symbols like:
- Stocks: AAPL, MSFT, GOOGL
- Indices: ^GSPC (S&P 500), ^DJI (Dow)
- ETFs: SPY, QQQ, IWM
""")

# Main input area
col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input("First Symbol", value="SPY").upper()
with col2:
    ticker2 = st.text_input("Second Symbol", value="AAPL").upper()

# Quick date range buttons
st.write("Quick Date Range:")
date_cols = st.columns(7)
today = datetime.now()
date_ranges = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "YTD": (today - datetime(today.year, 1, 1)).days,
    "1Y": 365,
    "3Y": 3*365,
    "5Y": 5*365
}

# Store the selected range in session state
if 'selected_range' not in st.session_state:
    st.session_state.selected_range = "1Y"

for i, (label, days) in enumerate(date_ranges.items()):
    with date_cols[i]:
        if st.button(label, key=f"date_range_{label}", 
                    help=f"Last {label}",
                    use_container_width=True):
            st.session_state.selected_range = label
            st.session_state.start_date = today - timedelta(days=days)
            st.session_state.end_date = today

# Custom date range
st.write("Or select custom date range:")
date_col1, date_col2 = st.columns(2)
with date_col1:
    start_date = st.date_input(
        "Start Date",
        value=getattr(st.session_state, 'start_date', today - timedelta(days=365)),
        max_value=today
    )
with date_col2:
    end_date = st.date_input(
        "End Date",
        value=getattr(st.session_state, 'end_date', today),
        max_value=today,
        min_value=start_date
    )

# Analysis options
with st.expander("Analysis Options", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_ma = st.checkbox("Show Moving Averages", value=True)
        if show_ma:
            ma_periods = st.multiselect(
                "MA Periods (days)",
                options=[20, 50, 100, 200],
                default=[50, 200]
            )
    
    with col2:
        show_volume = st.checkbox("Show Volume", value=True)
        show_rsi = st.checkbox("Show RSI", value=False)
    
    with col3:
        show_bollinger = st.checkbox("Show Bollinger Bands", value=False)
        if show_bollinger:
            bb_period = st.number_input("Bollinger Band Period", value=20, min_value=5, max_value=50)

# Time shift control
max_shift = min((end_date - start_date).days, 730)  # Max 2 years
shift_days = st.slider(
    "Shift Second Stock (days)",
    min_value=-max_shift,
    max_value=max_shift,
    value=0,
    help="Positive: stock lags comparison; Negative: stock leads comparison"
)

# Technical Analysis Functions
def calculate_ma(data, period):
    return data.rolling(window=period).mean()

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, period=20, std_dev=2):
    ma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, lower_band

# Function to validate ticker
@st.cache_data(ttl=3600)
def validate_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('regularMarketPrice') is not None
    except:
        return False

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, None
            
        # Handle both single-level and multi-level column indices
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close']
            volume_data = data[('Volume', ticker)] if ('Volume', ticker) in data.columns else data['Volume']
        else:
            close_data = data['Close']
            volume_data = data['Volume']
            
        return close_data, volume_data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

# Validate inputs
if not ticker1 or not ticker2:
    st.warning("Please enter both ticker symbols.")
    st.stop()

if ticker1 == ticker2:
    st.warning("Please enter different ticker symbols.")
    st.stop()

# Show loading message
with st.spinner("Validating tickers..."):
    valid1 = validate_ticker(ticker1)
    valid2 = validate_ticker(ticker2)

    if not valid1:
        st.error(f"Invalid ticker symbol: {ticker1}")
        st.stop()
    if not valid2:
        st.error(f"Invalid ticker symbol: {ticker2}")
        st.stop()

# Fetch data
with st.spinner("Fetching stock data..."):
    data1, volume1 = get_stock_data(ticker1, start_date, end_date)
    data2, volume2 = get_stock_data(ticker2, start_date, end_date)

    if data1 is None or data2 is None:
        st.error("Failed to fetch data for one or both stocks.")
        st.stop()

    # Normalize to start at 100
    data1_norm = (data1 / data1.iloc[0]) * 100
    data2_norm = (data2 / data2.iloc[0]) * 100

    # Create DataFrame
    df = pd.DataFrame({
        f"{ticker1}": data1_norm,
        f"{ticker2}": data2_norm,
        f"{ticker1}_raw": data1,
        f"{ticker2}_raw": data2,
        f"{ticker1}_volume": volume1,
        f"{ticker2}_volume": volume2
    }).dropna()

    if df.empty:
        st.error("No overlapping data found between the two stocks.")
        st.stop()

# Apply time shift if needed
if shift_days != 0:
    shifted_series2 = df[ticker2].shift(shift_days)
    df[ticker2] = shifted_series2
    df = df.dropna()

# Calculate technical indicators
if show_ma:
    for period in ma_periods:
        df[f"{ticker1}_MA{period}"] = calculate_ma(df[f"{ticker1}"], period)
        df[f"{ticker2}_MA{period}"] = calculate_ma(df[f"{ticker2}"], period)

if show_rsi:
    df[f"{ticker1}_RSI"] = calculate_rsi(df[f"{ticker1}_raw"])
    df[f"{ticker2}_RSI"] = calculate_rsi(df[f"{ticker2}_raw"])

if show_bollinger:
    df[f"{ticker1}_BB_upper"], df[f"{ticker1}_BB_lower"] = calculate_bollinger_bands(df[f"{ticker1}"], bb_period)
    df[f"{ticker2}_BB_upper"], df[f"{ticker2}_BB_lower"] = calculate_bollinger_bands(df[f"{ticker2}"], bb_period)

# Create subplots based on selected indicators
subplot_count = 1 + show_volume + show_rsi
heights = [0.5] + [0.25] * (subplot_count - 1)

fig = make_subplots(
    rows=subplot_count, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=heights
)

# Add main price comparison plot
fig.add_trace(
    go.Scatter(x=df.index, y=df[ticker1], name=ticker1, line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df[ticker2], name=ticker2, line=dict(color='red')),
    row=1, col=1
)

# Add moving averages
if show_ma:
    colors = ['rgba(0,0,255,0.3)', 'rgba(0,0,255,0.6)', 'rgba(255,0,0,0.3)', 'rgba(255,0,0,0.6)']
    for i, period in enumerate(ma_periods):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f"{ticker1}_MA{period}"],
                      name=f"{ticker1} MA{period}",
                      line=dict(color=colors[i % 2], dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f"{ticker2}_MA{period}"],
                      name=f"{ticker2} MA{period}",
                      line=dict(color=colors[2 + (i % 2)], dash='dash')),
            row=1, col=1
        )

# Add Bollinger Bands
if show_bollinger:
    for ticker in [ticker1, ticker2]:
        color = 'rgba(0,0,255,0.2)' if ticker == ticker1 else 'rgba(255,0,0,0.2)'
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f"{ticker}_BB_upper"],
                      name=f"{ticker} BB Upper",
                      line=dict(color=color, dash='dot'),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f"{ticker}_BB_lower"],
                      name=f"{ticker} BB Lower",
                      line=dict(color=color, dash='dot'),
                      fill='tonexty',
                      showlegend=False),
            row=1, col=1
        )

# Add volume subplot if selected
current_row = 2
if show_volume:
    fig.add_trace(
        go.Bar(x=df.index, y=df[f"{ticker1}_volume"], name=f"{ticker1} Volume",
               marker_color='rgba(0,0,255,0.3)', opacity=0.5),
        row=current_row, col=1
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df[f"{ticker2}_volume"], name=f"{ticker2} Volume",
               marker_color='rgba(255,0,0,0.3)', opacity=0.5),
        row=current_row, col=1
    )
    current_row += 1

# Add RSI subplot if selected
if show_rsi:
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f"{ticker1}_RSI"], name=f"{ticker1} RSI",
                   line=dict(color='blue')),
        row=current_row, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f"{ticker2}_RSI"], name=f"{ticker2} RSI",
                   line=dict(color='red')),
        row=current_row, col=1
    )
    # Add RSI reference lines
    for level in [30, 70]:
        fig.add_hline(y=level, line_dash="dash", line_color="gray",
                     annotation_text=f"RSI {level}", row=current_row)

# Update layout for mobile
fig.update_layout(
    height=200 * subplot_count + 400,  # Dynamic height based on subplots
    showlegend=True,
    hovermode='x unified',
    template='plotly_white',
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update y-axes labels
fig.update_yaxes(title_text="Normalized Price (Start = 100)", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="Volume", row=2, col=1)
if show_rsi:
    fig.update_yaxes(title_text="RSI", row=current_row, col=1)

# Show plot
st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': False,
    'scrollZoom': True
})

# Performance Metrics
st.subheader("Performance Metrics")
metrics_cols = st.columns(4)

# Calculate metrics
total_return1 = (df[f"{ticker1}_raw"].iloc[-1] / df[f"{ticker1}_raw"].iloc[0] - 1) * 100
total_return2 = (df[f"{ticker2}_raw"].iloc[-1] / df[f"{ticker2}_raw"].iloc[0] - 1) * 100
volatility1 = df[f"{ticker1}"].pct_change().std() * np.sqrt(252) * 100
volatility2 = df[f"{ticker2}"].pct_change().std() * np.sqrt(252) * 100
correlation = df[ticker1].corr(df[ticker2])
beta = df[ticker2].pct_change().cov(df[ticker1].pct_change()) / df[ticker1].pct_change().var()

with metrics_cols[0]:
    st.metric(f"{ticker1} Return", f"{total_return1:.1f}%")
    st.metric(f"{ticker1} Volatility", f"{volatility1:.1f}%")

with metrics_cols[1]:
    st.metric(f"{ticker2} Return", f"{total_return2:.1f}%")
    st.metric(f"{ticker2} Volatility", f"{volatility2:.1f}%")

with metrics_cols[2]:
    st.metric("Correlation", f"{correlation:.2f}")
    relative_perf = total_return1 - total_return2
    st.metric("Relative Performance", f"{relative_perf:.1f}%")

with metrics_cols[3]:
    st.metric("Beta", f"{beta:.2f}")
    sharpe1 = (total_return1 / volatility1) if volatility1 != 0 else 0
    st.metric(f"{ticker1} Sharpe Ratio", f"{sharpe1:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Data from Yahoo Finance • Prices adjusted for splits and dividends</small>
</div>
""", unsafe_allow_html=True)