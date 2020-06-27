import numpy as np
import scipy.stats
import pandas as pd
import yfinance as yf
from itertools import chain
from pandas_datareader import data as pdr
from ValueAtRisk import StochasticProcesses

yf.pdr_override()
# prices downloaded from MSCI's RiskMetrics System
# webData = pd.read_csv("MSCIRiskMetricsPrices.csv", header=0)
# dailyPrices = webData.iloc[:, 1:6]

dailyPrices = pdr.get_data_yahoo("GOOGL AMZN AAPL FB MSFT", start="2019-06-26", end="2020-06-24")['Close']
quantity = np.array([9, 5, 18, 7, 3])
weights = quantity / quantity.sum()

logReturns = np.log(dailyPrices / dailyPrices.shift(1)).dropna()
previousDayPrice = dailyPrices.tail(1).values
previousDayPortfolio = list(chain.from_iterable(previousDayPrice * quantity))

# Historic Simulation
historicSimulatedPrice = logReturns.mul(previousDayPortfolio, axis=1)
historicSimulatedPortfolio = historicSimulatedPrice.sum(axis=1)
historicSimulationVaR95 = historicSimulatedPortfolio.quantile(0.05)
historicSimulationVaR99 = historicSimulatedPortfolio.quantile(0.01)

# Delta Normal Parametric
portfolioMean = (logReturns.mean() * weights).sum()
portfolioVariance = np.dot(weights.T, np.dot(logReturns.cov(), weights))
portfolioStdDeviation = np.sqrt(portfolioVariance)
parametricVaR95 = scipy.stats.norm.ppf(0.05, portfolioMean, portfolioStdDeviation) * np.sum(previousDayPortfolio)
parametricVaR99 = scipy.stats.norm.ppf(0.01, portfolioMean, portfolioStdDeviation) * np.sum(previousDayPortfolio)

# Monte Carlo Simulation using Geometric Brownian Motion path of Stock Prices
ndays = dailyPrices.shape[0] - 1
timeLine = np.linspace(0, ndays, ndays)
simulation_df = pd.DataFrame()
for run in range(1000):
    randVariables = np.random.normal(0, 1, int(ndays))
    price = (previousDayPrice * quantity).sum() * \
        StochasticProcesses.geometricbrownianWithDrft(portfolioMean, portfolioStdDeviation, ndays, randVariables)[0]
    prices = pd.Series(price)
    simulation_df = simulation_df.append(prices, ignore_index=True)
price_array = simulation_df.iloc[-1, :]
monteCarloVaR95 = - ((previousDayPrice * quantity).sum() - price_array.quantile(0.05))
monteCarloVaR99 = - ((previousDayPrice * quantity).sum() - price_array.quantile(0.01))

print(['1 Day 95% Historical VaR =', historicSimulationVaR95], ['1 Day 99% Historical VaR = ', historicSimulationVaR99])
print(['1 Day 95% Parametric VaR = ', parametricVaR95], ['1 Day 99% Parametric VaR ', parametricVaR99])
print(['1 Day 95% Monte Carlo VaR = ', monteCarloVaR95], ['1 Day 99% Monte Carlo VaR = ', monteCarloVaR99])
