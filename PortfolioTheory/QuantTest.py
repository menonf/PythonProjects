import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

webData = pd.DataFrame.from_csv("Quant.csv", header=0)
returns = webData
stocks  = ['FundA', 'FundB', 'FundC', 'FundD', 'FundE']

mean_daily_returns = returns.mean()                     # calculate mean daily return and covariance of daily returns
cov_matrix = returns.cov()
num_portfolios = 25000                                  # set number of runs of random portfolio weights

# set up array to hold results, increased the size of the array to hold the weight values for each stock
results = np.zeros((5 + len(stocks) - 1, num_portfolios))

for i in range(num_portfolios):
    weights = np.array(np.random.random(5))              # select random weights for portfolio holdings
    weights /= np.sum(weights)                           # rebalance weights to sum to 1

    # calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 12
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)

    # store results in results array
    results[1, i] = portfolio_return
    results[2, i] = portfolio_std_dev

    # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[3, i] = results[1, i] / results[2, i]

    # iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j + 4, i] = weights[j]

# convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T, columns=['Date', 'Returns', 'stdev', 'sharpe', stocks[0], stocks[1], stocks[2], stocks[3],stocks[4]])
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
plt.scatter(max_sharpe_port[2], max_sharpe_port[1], marker='*', color='r', label="Max Sharpe Ratio", s=200)
# plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[2], min_vol_port[1], marker='*', color='g', s=200, label="Min Variance")
plt.title('Efficient Portfolio')
plt.legend(numpoints=1)
plt.show()

print(min_vol_port)
print(max_sharpe_port)