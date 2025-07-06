import streamlit as st
import pandas as pd
from data_handler import load_portfolio, fetch_price_data
from optimize_portfolio import optimize_portfolio, portfolio_performance
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")


# Upload Portfolio
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

# Time period selection
st.subheader("Select historical data period")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", key="start_date")
with col2:
    end_date = st.date_input("End date", key="end_date")

if uploaded_file:
    tickers, weights = load_portfolio(uploaded_file)
    st.success("Portfolio Loaded!")

    # Fetch Price Data
    st.subheader(f"Fetching historical price data from {start_date} to {end_date}...")
    # Convert dates to string format for yfinance (YYYY-MM-DD)
    price_data = fetch_price_data(tickers, period=None, interval='1d')
    # If both dates are selected, filter the data
    if start_date and end_date:
        price_data = price_data.loc[str(start_date):str(end_date)]
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252  # Annualize covariance matrix

    # Show Real Portfolio Performance
    real_ret, real_vol, real_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    st.metric("Real Sharpe Ratio", f"{real_sharpe:.2f}")
    st.metric("Real Return", f"{real_ret:.2%}")
    st.metric("Real Volatility", f"{real_vol:.2%}")

    # Optimize
    st.subheader("🧠 Optimize Portfolio for Max Sharpe")
    if st.button("Run Optimization"):
        opt_weights = optimize_portfolio(returns)
        opt_ret, opt_vol, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)
        
        st.metric("Optimized Sharpe Ratio", f"{opt_sharpe:.2f}")
        st.metric("Optimized Return", f"{opt_ret:.2%}")
        st.metric("Optimized Volatility", f"{opt_vol:.2%}")

        # Show weights
        st.subheader("🔍 Portfolio Allocation")
        opt_df = pd.DataFrame({
            'Ticker': returns.columns,
            'Original Weight': weights,
            'Optimized Weight': opt_weights
        })
        st.dataframe(opt_df)

        # Efficient Frontier
        st.subheader("📊 Efficient Frontier")
        def simulate_portfolios(num_portfolios=5000):
            results = np.zeros((3, num_portfolios))
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
                results[0,i] = ret
                results[1,i] = vol
                results[2,i] = sharpe
            return results

        results = simulate_portfolios()
        plt.figure(figsize=(10,6))
        plt.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.7)
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(opt_vol, opt_ret, c='red', s=50, label='Optimized')
        plt.legend()
        st.pyplot(plt)