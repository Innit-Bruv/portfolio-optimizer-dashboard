# utils/portfolio_metrics.py

import numpy as np
import pandas as pd

def calculate_daily_returns(prices_df):
    return prices_df.pct_change().dropna()

def calculate_cumulative_returns(daily_returns):
    return (1 + daily_returns).cumprod()

def annualized_return(daily_returns):
    return daily_returns.mean() * 252

def annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio(daily_returns):
    ann_return = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    return ann_return / ann_vol

