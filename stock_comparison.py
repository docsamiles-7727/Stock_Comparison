import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta

# Function to validate ticker symbol
def validate_ticker(ticker):
    try:
        # Try to get info about the ticker
        info = yf.Ticker(ticker).info
        if info.get('regularMarketPrice') is None:
            return False
        return True
    except:
        return False

# Get user input for tickers
while True:
    print("\nEnter ticker symbols to compare (e.g., AAPL, MSFT, GOOGL, SPY, etc.)")
    print("For indices, you can use: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (NASDAQ)")
    ticker1 = input("Enter first ticker symbol: ").strip().upper()
    ticker2 = input("Enter second ticker symbol: ").strip().upper()

    # Validate inputs
    if not ticker1 or not ticker2:
        print("Error: Ticker symbols cannot be empty")
        continue
    if ticker1 == ticker2:
        print("Error: Please enter different ticker symbols")
        continue

    # Validate ticker existence
    print(f"\nValidating {ticker1}...")
    if not validate_ticker(ticker1):
        print(f"Error: Could not find ticker symbol '{ticker1}'")
        continue

    print(f"Validating {ticker2}...")
    if not validate_ticker(ticker2):
        print(f"Error: Could not find ticker symbol '{ticker2}'")
        continue

    print(f"\nComparing {ticker1} vs {ticker2}")
    break
# Get current date for end_date

# Function to validate date format
def validate_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

# Get and validate date range
while True:
    print("\nEnter date range (format: YYYY-MM-DD)")
    print("Leave blank to use defaults: last 3 years to today")
    start_date_input = input("Start date: ").strip()
    end_date_input = input("End date: ").strip()
    
    # Use defaults if both empty
    if not start_date_input and not end_date_input:
        end = datetime.now()
        start = end - timedelta(days=3*365)
        break
    
    # Validate dates
    start = validate_date(start_date_input) if start_date_input else None
    end = validate_date(end_date_input) if end_date_input else datetime.now()
    
    if start_date_input and not start:
        print("Error: Invalid start date format. Please use YYYY-MM-DD")
        continue
        
    if end_date_input and not end:
        print("Error: Invalid end date format. Please use YYYY-MM-DD")
        continue
    
    # If only start date provided
    if start and not end_date_input:
        end = datetime.now()
    
    # If only end date provided
    if not start_date_input and end:
        start = end - timedelta(days=3*365)
    
    # Validate date range
    if start >= end:
        print("Error: Start date must be before end date")
        continue
    
    # Check if date range is too long (more than 10 years)
    if (end - start).days > 3650:
        print("Warning: Date range is very long (>10 years). This might affect performance.")
        if input("Continue? (y/n): ").lower() != 'y':
            continue
    
    break

# Convert dates to string format
start_date = start.strftime('%Y-%m-%d')
end_date = end.strftime('%Y-%m-%d')
print(f"\nUsing date range: {start_date} to {end_date}")

# Set max shift based on date range (but cap at 2 years)
max_shift = min((end - start).days, 730)  # Max 2 years shift

# Function to safely download and process stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        # Download data for a single ticker
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Debug information
        print(f"\nDebug info for {ticker}:")
        print(f"Data type: {type(data)}")
        print(f"Is empty: {data.empty}")
        if isinstance(data, pd.DataFrame):
            print(f"Columns: {data.columns}")
            print(f"Shape: {data.shape}")
            print(f"First few rows:\n{data.head()}")
        
        # Validate DataFrame
        if data.empty:
            raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}")
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Invalid data format received for {ticker}")
        
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
        
        # Ensure we have a Series
        if not isinstance(close_data, pd.Series):
            raise ValueError(f"Invalid closing price data format for {ticker}")
        
        # Ensure we have valid numeric data
        if not close_data.dtype.kind in 'fc':  # f=float, c=complex
            raise ValueError(f"Non-numeric data received for {ticker}")
        
        # Check for missing values
        if close_data.isna().any():
            print(f"Warning: Found {close_data.isna().sum()} missing values for {ticker}")
            close_data = close_data.fillna(method='ffill')  # Forward fill missing values
        
        return close_data
        
    except Exception as e:
        print(f"\nError processing {ticker}:")
        print(f"Original error: {str(e)}")
        raise

try:
    # Fetch historical closing prices
    print(f"Downloading data for {ticker1}...")
    data1 = get_stock_data(ticker1, start_date, end_date)
    print(f"Downloading data for {ticker2}...")
    data2 = get_stock_data(ticker2, start_date, end_date)

    # Verify we have enough data points
    if len(data1) < 2 or len(data2) < 2:
        raise ValueError("Insufficient data points for comparison")

    # Normalize to start at 100 (percentage change basis)
    data1 = (data1 / data1.iloc[0]) * 100
    data2 = (data2 / data2.iloc[0]) * 100

    # Combine into a DataFrame (aligns on dates automatically)
    df = pd.DataFrame({ticker1: data1, ticker2: data2}).dropna()  # Drop any mismatched dates

    if df.empty:
        raise ValueError("No overlapping dates found between the two stocks")

except Exception as e:
    print(f"Error: {str(e)}")
    print("Please check your ticker symbols and date range.")
    exit(1)

# Set up the figure with two subplots: original graphs and difference
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(bottom=0.25)

# Plot the normalized series
line1, = ax1.plot(df.index, df[ticker1], label=ticker1)
line2, = ax1.plot(df.index, df[ticker2], label=ticker2)
ax1.set_title(f'Normalized Price Comparison: {ticker1} vs {ticker2}')
ax1.set_ylabel('Normalized Price (Start = 100)')
ax1.legend()
ax1.grid(True)

# Plot the initial difference
initial_diff = df[ticker1] - df[ticker2]
diff_line, = ax2.plot(df.index, initial_diff, label='Difference', color='purple')
ax2.set_title('Difference (S&P 500 - Stock)')
ax2.set_ylabel('Difference')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)

# Add slider for time-shifting the second series
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Shift Stock (days)', -max_shift, max_shift, valinit=0, valstep=1)

# Update function for slider
def update(val):
    shift = int(slider.val)
    # Shift the second series (positive shift: stock lags S&P; negative: stock leads)
    shifted_series2 = df[ticker2].shift(shift)
    # Realign and drop NaNs for clean plotting
    shifted_df = pd.DataFrame({ticker1: df[ticker1], 'shifted': shifted_series2}).dropna()
    # Update plots
    line2.set_xdata(shifted_df.index)
    line2.set_ydata(shifted_df['shifted'])
    diff_line.set_xdata(shifted_df.index)
    diff_line.set_ydata(shifted_df[ticker1] - shifted_df['shifted'])
    # Rescale axes to fit new data
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
