import numpy
from itertools import chain
from matplotlib import pyplot
import scipy.stats
import numpy as np

from tiingo import TiingoClient
import datetime

# Tiingo API call to fetch data
config = {}
config['session'] = True
config['api_key'] = "81f0783ae3d1756869af76d72b52a86f08e2ca15"

client = TiingoClient(config)
stocks = ['MSFT','AAPL', 'AMZN', 'GOOGL', 'FB']
dailyPrices = client.get_dataframe(stocks,
                                      frequency='daily',
                                      metric_name='open',
                                      startDate=str(datetime.datetime.now().date() - datetime.timedelta(days=366)),
                                      endDate=str(datetime.datetime.now().date()))

shares = np.array([4, 2, 2, 4, 25])
weights =  shares/shares.sum()

logReturns = numpy.log(dailyPrices/dailyPrices.shift(1)).dropna()
portfolioLogReturns = logReturns.sum(axis=1)

previousDayPrice = dailyPrices.tail(1).values
previousDayPortfolio = list(chain.from_iterable(previousDayPrice * shares))

portfolioMean = (logReturns.mean() * weights).sum()
portfolioVariance = numpy.dot(weights.T, numpy.dot(logReturns.cov(), weights))
portfolioStdDeviation = numpy.sqrt(portfolioVariance)

print('1 day VaR @ 99% confidence interval =',scipy.stats.norm.ppf(0.01, portfolioMean, portfolioStdDeviation) * numpy.sum(previousDayPortfolio))
print('1 day VaR @ 95% confidence interval =',scipy.stats.norm.ppf(0.05, portfolioMean, portfolioStdDeviation) * numpy.sum(previousDayPortfolio))
print('1 day VaR @ 90% confidence interval =',scipy.stats.norm.ppf(0.10, portfolioMean, portfolioStdDeviation) * numpy.sum(previousDayPortfolio))

pyplot.hist(portfolioLogReturns, bins=25)
xmin, xmax = pyplot.xlim()
x = numpy.linspace(xmin, xmax, 1000)
p =scipy.stats.norm.pdf(x, portfolioMean.mean(), portfolioStdDeviation)
pyplot.plot(x, p, 'k', linewidth=2)
title = "Plot mu = %.2f,  std = %.2f" % (portfolioMean.mean(), portfolioStdDeviation)
pyplot.title(title)
pyplot.show()
