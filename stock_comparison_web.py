import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Stock Comparison Tool",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Comparison Tool")
st.markdown("""
Compare the performance of two stocks or indices over time. The tool normalizes prices to start at 100
for easy comparison of relative performance.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # Ticker inputs with examples
    st.subheader("Enter Ticker Symbols")
    st.markdown("""
    Examples:
    - Stocks: AAPL, MSFT, GOOGL, TSLA
    - Indices: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (NASDAQ)
    - ETFs: SPY, QQQ, IWM
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input("First Ticker", value="SPY").upper()
    with col2:
        ticker2 = st.text_input("Second Ticker", value="AAPL").upper()

    # Date range selector
    st.subheader("Select Date Range")
    today = datetime.now()
    default_start = today - timedelta(days=3*365)  # 3 years ago
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=today
    )
    
    end_date = st.date_input(
        "End Date",
        value=today,
        max_value=today,
        min_value=start_date
    )
    
    # Shift range slider
    st.subheader("Time Shift")
    max_shift = min((end_date - start_date).days, 730)  # Max 2 years
    shift_days = st.slider(
        "Shift Second Stock (days)",
        min_value=-max_shift,
        max_value=max_shift,
        value=0,
        help="Positive: stock lags comparison; Negative: stock leads comparison"
    )

# Function to validate ticker
@st.cache_data(ttl=3600)  # Cache for 1 hour
def validate_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('regularMarketPrice') is not None
    except:
        return False

# Function to get stock data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
            
        # Handle both single-level and multi-level column indices
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns - select the 'Close' price for our ticker
            if ('Close', ticker) in data.columns:
                close_data = data[('Close', ticker)]
            else:
                raise ValueError(f"No closing price data available for {ticker}")
        else:
            # Single-level columns - just get 'Close'
            if 'Close' not in data.columns:
                raise ValueError(f"No closing price data available for {ticker}")
            close_data = data['Close']
            
        # Ensure we have a Series with the correct index
        if not isinstance(close_data, pd.Series):
            if isinstance(close_data, pd.DataFrame):
                if close_data.shape[1] == 1:
                    close_data = close_data.iloc[:, 0]
                else:
                    raise ValueError(f"Invalid data format for {ticker}")
                    
        return close_data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

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
    data1 = get_stock_data(ticker1, start_date, end_date)
    data2 = get_stock_data(ticker2, start_date, end_date)

    if data1 is None or data2 is None:
        st.error("Failed to fetch data for one or both stocks.")
        st.stop()

    # Normalize to start at 100
    data1 = (data1 / data1.iloc[0]) * 100
    data2 = (data2 / data2.iloc[0]) * 100

    # Create DataFrame
    df = pd.DataFrame({ticker1: data1, ticker2: data2}).dropna()

    if df.empty:
        st.error("No overlapping data found between the two stocks.")
        st.stop()

# Apply time shift if needed
if shift_days != 0:
    shifted_series2 = df[ticker2].shift(shift_days)
    df = pd.DataFrame({ticker1: df[ticker1], ticker2: shifted_series2}).dropna()

# Create plots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"Normalized Price Comparison: {ticker1} vs {ticker2}",
        "Price Difference"
    ),
    vertical_spacing=0.15,
    row_heights=[0.7, 0.3]
)

# Add price lines
fig.add_trace(
    go.Scatter(x=df.index, y=df[ticker1], name=ticker1, line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df[ticker2], name=ticker2, line=dict(color='red')),
    row=1, col=1
)

# Add difference plot
diff = df[ticker1] - df[ticker2]
fig.add_trace(
    go.Scatter(x=df.index, y=diff, name='Difference',
               line=dict(color='purple'), fill='tonexty'),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

# Update y-axes labels
fig.update_yaxes(title_text="Normalized Price (Start = 100)", row=1, col=1)
fig.update_yaxes(title_text="Difference", row=2, col=1)

# Show plot
st.plotly_chart(fig, use_container_width=True)

# Add summary statistics
st.subheader("Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        f"{ticker1} Total Return",
        f"{(df[ticker1].iloc[-1] - 100):.1f}%",
        delta=f"{(df[ticker1].iloc[-1] - df[ticker1].iloc[0]):.1f}"
    )

with col2:
    st.metric(
        f"{ticker2} Total Return",
        f"{(df[ticker2].iloc[-1] - 100):.1f}%",
        delta=f"{(df[ticker2].iloc[-1] - df[ticker2].iloc[0]):.1f}"
    )

with col3:
    relative_perf = df[ticker1].iloc[-1] - df[ticker2].iloc[-1]
    st.metric(
        "Relative Performance",
        f"{relative_perf:.1f}%",
        delta=f"{relative_perf:.1f}"
    )

# Add correlation analysis
st.subheader("Correlation Analysis")
correlation = df[ticker1].corr(df[ticker2])
st.write(f"Correlation coefficient between {ticker1} and {ticker2}: {correlation:.3f}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Data provided by Yahoo Finance. Prices are adjusted for splits and dividends.</p>
    </div>
    """, unsafe_allow_html=True)
