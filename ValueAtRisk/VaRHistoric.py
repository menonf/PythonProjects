import numpy
import pandas
from matplotlib import pyplot
from itertools import chain

webData = pandas.read_csv("Sp500history.csv", header=0)
dailyPrices = webData.iloc[:, 1:3]
weights = numpy.array([2000, 500])

logReturns = numpy.log(dailyPrices/dailyPrices.shift(1)).dropna()  # same as Portfolio.pct_change() #Percentage change
previousDayPrice = dailyPrices.tail(1).values

previousDayPortfolio = list(chain.from_iterable(previousDayPrice * weights))
possibleOutcome = logReturns.mul(previousDayPortfolio, axis=1)
possiblePortfolio = possibleOutcome.sum(axis=1)

possiblePortfolio.hist()

VaR_90 = possiblePortfolio.quantile(0.1)
VaR_95 = possiblePortfolio.quantile(0.05)
VaR_99 = possiblePortfolio.quantile(0.01)

print(['90%', VaR_90], ['95%', VaR_95], ["99%", VaR_99])
pyplot.show()
