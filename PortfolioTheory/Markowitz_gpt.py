import numpy as np
import pandas as pd
import tiingo
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

# Set your Tiingo API key
config = {
    'api_key': '81f0783ae3d1756869af76d72b52a86f08e2ca15'
}
client = tiingo.TiingoClient(config)

# Define the assets and their tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Define the start and end dates for the historical data
start_date = '2017-05-05'
end_date = '2018-05-18'

# Retrieve the historical price data using Tiingo client
prices_df = client.get_dataframe(
    tickers,
    frequency='daily',
    startDate=start_date,
    endDate=end_date,
    metric_name='adjClose'
)

# Calculate the logarithmic returns from the price data
returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

# Calculate the mean returns and covariance matrix
mean_returns = returns_df.mean() * 100  # Convert to percentages
cov_matrix = returns_df.cov() * 100  # Convert to percentages

# Markowitz optimization
n_assets = len(tickers)
weights = cp.Variable(n_assets)
returns = mean_returns.values
risk = cp.quad_form(weights, cov_matrix.values)

# Define the optimization problem
problem = cp.Problem(cp.Maximize(returns @ weights),
                     [cp.sum(weights) == 1,
                      weights >= 0])

# Solve the optimization problem
problem.solve()

# Get the optimal portfolio weights
optimal_weights = weights.value

# Calculate the portfolio statistics
portfolio_returns = np.dot(returns, optimal_weights)
portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))

# Print the optimal weights and portfolio statistics
print("Optimal Weights:")
for i in range(len(tickers)):
    print(tickers[i], optimal_weights[i])

print("\nPortfolio Statistics:")
print("Expected Return:", portfolio_returns)
print("Volatility:", portfolio_volatility)

# Markowitz efficient frontier
n_points = 20
target_returns = np.linspace(min(mean_returns), max(mean_returns), n_points)
efficient_frontier = np.zeros((n_points, 2))

for i, target_return in enumerate(target_returns):
    weights = cp.Variable(n_assets)
    risk = cp.quad_form(weights, cov_matrix.values)

    # Define the optimization problem for a target return
    problem = cp.Problem(cp.Minimize(risk),
                         [cp.sum(weights) == 1,
                          returns @ weights == target_return,
                          weights >= 0])

    # Solve the optimization problem
    problem.solve()

    efficient_frontier[i, 0] = target_return
    efficient_frontier[i, 1] = np.sqrt(risk.value)

# Print the efficient frontier
print("\nEfficient Frontier:")
for point in efficient_frontier:
    print("Return:", point[0], "Volatility:", point[1])

# Define the colormap
cmap = cm.coolwarm_r

# Normalize the values for colormap
norm = mcolors.Normalize(
    vmin=min(efficient_frontier[:, 0] / efficient_frontier[:, 1]),
    vmax=max(efficient_frontier[:, 0] / efficient_frontier[:, 1])
)

# Calculate the points with maximum Sharpe ratio and minimum variance
max_sharpe_ratio = efficient_frontier[np.argmax(efficient_frontier[:, 0] / efficient_frontier[:, 1])]
min_variance = efficient_frontier[np.argmin(efficient_frontier[:, 1])]

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    efficient_frontier[:, 1],
    efficient_frontier[:, 0],
    c=efficient_frontier[:, 0] / efficient_frontier[:, 1],
    cmap=cmap,
    norm=norm,
    marker='o'
)

# Add markers for maximum Sharpe ratio and minimum variance
plt.scatter(max_sharpe_ratio[1], max_sharpe_ratio[0], color='red', marker='*', s=200, label="Max Sharpe Ratio")
plt.scatter(min_variance[1], min_variance[0], color='green', marker='*', s=200, label="Min Variance")

for i in range(len(efficient_frontier) - 1):
    x = [efficient_frontier[i, 1], efficient_frontier[i + 1, 1]]
    y = [efficient_frontier[i, 0], efficient_frontier[i + 1, 0]]
    color = cmap(norm(efficient_frontier[i, 0] / efficient_frontier[i, 1]))
    plt.plot(x, y, c=color, linewidth=1.5)

plt.xlabel('Volatility (%)')
plt.ylabel('Return (%)')
plt.title('Markowitz Efficient Frontier')

# Add colorbar on the right side
cbar = plt.colorbar(scatter, aspect=40, label='Sharpe Ratio')
cbar.ax.set_ylabel('Sharpe Ratio')

plt.grid(True)
plt.legend()
plt.show()
