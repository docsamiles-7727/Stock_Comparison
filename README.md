# Stock Comparison Tool

A web application built with Streamlit that allows users to compare the performance of any two stocks or market indices over time.

## Features

- Compare any two stocks or market indices
- Interactive date range selection
- Time-shift analysis to find leading/lagging relationships
- Normalized price comparison (starting at 100)
- Price difference analysis
- Summary statistics and correlation analysis
- Real-time data from Yahoo Finance

## Installation

1. Clone this repository:
```bash
git clone https://github.com/docsamiles-7727/Stock_Comparison.git
cd Stock_Comparison
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the web application:
```bash
streamlit run stock_comparison_web.py
```

Or run the command-line version:
```bash
python stock_comparison.py
```

## Input Examples

- Stock symbols: AAPL, MSFT, GOOGL, TSLA
- Market indices: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (NASDAQ)
- ETFs: SPY, QQQ, IWM

## Live Demo

You can try the live version of this tool at:
https://stock-comparison-docsamiles-7727.streamlit.app

## License

MIT License