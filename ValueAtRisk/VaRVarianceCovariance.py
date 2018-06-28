import numpy
import pandas
from itertools import chain
from matplotlib import pyplot
import scipy.stats

webData = pandas.DataFrame.from_csv("Portfolio.csv", header=0)
dailyPrices = webData.iloc[:, 0:3]
weights = numpy.array([30, 40, 10])

logReturns = numpy.log(dailyPrices/dailyPrices.shift(1)).dropna()
portfolioLogReturns = logReturns.sum(axis=1)

previousDayPrice = dailyPrices.tail(1).values
previousDayPortfolio = list(chain.from_iterable(previousDayPrice * weights))

PortfolioMean = (logReturns.mean() * previousDayPortfolio).sum()
portfolioVariance = numpy.matmul(numpy.matmul(logReturns.cov(), previousDayPortfolio), previousDayPortfolio)
portfolioStdDeviation = numpy.sqrt(portfolioVariance)

tdf, tmean, tsigma = scipy.stats.t.fit(portfolioLogReturns.as_matrix())
support = numpy.linspace(portfolioLogReturns.min(), portfolioLogReturns.max(), 100)
portfolioLogReturns.hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5);
pyplot.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
pyplot.title(u"Daily change in Portfolio(%)", weight='bold');
pyplot.show()

#print(['90%', portfolioStdDeviation*1.28], ['95%', portfolioStdDeviation*1.65], ["99%", portfolioStdDeviation*2.33])
print('VaR @ 99% confidence interval =',scipy.stats.norm.ppf(0.01, PortfolioMean, portfolioStdDeviation))
print('VaR @ 95% confidence interval =',scipy.stats.norm.ppf(0.05, PortfolioMean, portfolioStdDeviation))
print('VaR @ 90% confidence interval =',scipy.stats.norm.ppf(0.10, PortfolioMean, portfolioStdDeviation))

