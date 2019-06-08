import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get adjusted closing prices of 5 selected companies with Quandl
quandl.ApiConfig.api_key = 'puxYR3syR5S-DyzWszgu'
stocks = ['AAPL', 'AMZN', 'GOOGL', 'FB']
data = quandl.get_table('WIKI/PRICES', ticker=stocks, qopts={'columns': ['date', 'ticker', 'adj_close']},
                        date={'gte': '2016-1-1', 'lte': '2018-07-26'}, paginate=True)

new = data.set_index('date')
table = new.pivot(columns='ticker')
returns = table.pct_change()

# calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# set number of runs of random portfolio weights
num_portfolios = 25000

# set up array to hold results
# We have increased the size of the array to hold the weight values for each stock
results = np.zeros((4 + len(stocks) - 1, num_portfolios))

for i in range(num_portfolios):
    # select random weights for portfolio holdings
    weights = np.array(np.random.random(4))
    # rebalance weights to sum to 1
    weights /= np.sum(weights)

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
results_frame = pd.DataFrame(results.T, columns=['Returns', 'stdev', 'sharpe', stocks[0], stocks[1], stocks[2], stocks[3]])

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