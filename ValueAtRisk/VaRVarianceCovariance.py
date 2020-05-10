import numpy
import pandas
from itertools import chain
from matplotlib import pyplot
import scipy.stats

webData = pandas.DataFrame.from_csv("Portfolio.csv", header=0)
dailyPrices = webData.iloc[:, 0:3]
weights = numpy.array([0.375, 0.5, 0.125])
shares = numpy.array([30, 40, 10])

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
