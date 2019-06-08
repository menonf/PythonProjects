import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tiingo import TiingoClient

# Tiingo API call to fetch data
config = {}
config['session'] = True
config['api_key'] = "81f0783ae3d1756869af76d72b52a86f08e2ca15"

client = TiingoClient(config)
stocks = ['MSFT','AAPL', 'AMZN', 'GOOGL', 'FB']
ticker_history = client.get_dataframe(stocks, frequency='daily',
                                      metric_name='adjClose', startDate='2017-05-05', endDate='2019-05-18')
# calculate daily returns
returns = ticker_history.pct_change()

# calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# set number of runs of random portfolio weights
num_portfolios = 25000

# set up an array to hold results
# I have increased the size of the array to hold the weight values for each stock
results = np.zeros((4 + len(stocks) - 1, num_portfolios))

for i in range(num_portfolios):
    # select random weights for portfolio holdings
    weights = np.array(np.random.random(len(stocks)))
    weights = weights / np.sum(weights)

    # calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # store results in results array
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev

    # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2, i] = results[0, i] / results[1, i]

    # iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j + 3, i] = weights[j]

# convert results array to Pandas DataFrame
col1 = ['Returns', 'stdev', 'sharpe'] + stocks
results_frame = pd.DataFrame(results.T, columns=col1)

# locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
# locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

# create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.Returns, c=results_frame.sharpe, cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()

# plot red star to highlight position of portfolio with
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker= '*', color='r', label="Max Sharpe Ratio", s=200)
# plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1], min_vol_port[0], marker='*', color='g', s=200, label="Min Variance")

plt.title('Efficient Portfolio')
plt.legend(numpoints=1)
plt.show()

print(max_sharpe_port)
print(max_sharpe_port[1])
print(max_sharpe_port[0])
