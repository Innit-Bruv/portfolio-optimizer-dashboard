import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def optimize_portfolio(returns_df, method='max_sharpe', risk_free_rate=0.0):
    """
    Optimizes portfolio weights to either maximize Sharpe Ratio or minimize volatility.

    Parameters:
    - returns_df: DataFrame of daily returns (not annualized)
    - method: 'max_sharpe' or 'min_volatility'
    - risk_free_rate: annual risk-free rate (e.g., 0.03 for 3%)

    Returns:
    - opt_weights: array of optimal weights
    """
    # === Annualize inputs ===
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov()  * 252
    num_assets = len(mean_returns)

    # === Constraints: weights must sum to 1 ===
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # === Bounds: weights must be between 0 and 1 (no short selling) ===
    bounds = tuple((0, 1) for _ in range(num_assets))

    # === Starting guess: equally weighted ===
    initial_guess = np.array([1.0 / num_assets] * num_assets)

    # === Objective functions ===
    if method == 'max_sharpe':
        def negative_sharpe(w):
            return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]
        objective = negative_sharpe

    elif method == 'min_volatility':
        def volatility(w):
            return portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[1]
        objective = volatility

    else:
        raise ValueError("Method must be either 'max_sharpe' or 'min_volatility'")

    # === Optimization ===
    result = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    # === Check success ===
    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    return result.x  # Optimal weights

import numpy as np
from scipy.optimize import minimize

def calculate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    """
    Calculates the efficient frontier by finding the minimum volatility
    for a range of target returns.
    """
    results = np.zeros((3, num_portfolios))
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_portfolios)

    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1.0 / num_assets] * num_assets)

    for i, target_return in enumerate(target_returns):
        # Constraints: sum of weights is 1, and portfolio return equals the target return
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}
        )

        # Objective: minimize volatility
        def objective(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            portfolio_vol = result.fun
            results[0, i] = target_return
            results[1, i] = portfolio_vol
            # Sharpe for this point on the frontier (assuming risk-free rate = 0)
            results[2, i] = (target_return - 0.0) / portfolio_vol

    # Filter out any failed optimizations (vol = 0)
    return results[:, results[1] > 0]
