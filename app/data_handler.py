# app/data_handler.py

import pandas as pd
import yfinance as yf

def load_portfolio(file):
    df = pd.read_csv(file)
    tickers = df['Ticker'].tolist()
    weights = df['Weight'].values
    return tickers, weights

def fetch_price_data(tickers, period='1y', interval='1d'):
    data = yf.download(tickers, period=period, interval=interval)
    if data.empty:
        raise ValueError("No price data returned. Check ticker symbols or network connection.")
    # Handle both single and multi-ticker cases
    if 'Adj Close' in data:
        price_data = data['Adj Close']
    elif 'Close' in data:
        price_data = data['Close']
    else:
        # If only one ticker, columns may not be multi-index
        if isinstance(data.columns, pd.MultiIndex):
            raise KeyError("'Adj Close' not found in downloaded data.")
        else:
            price_data = data
    return price_data

