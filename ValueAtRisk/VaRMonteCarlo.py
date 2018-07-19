import numpy
import pandas
from itertools import chain
from matplotlib import pyplot
from matplotlib import mlab
from ValueAtRisk import StochasticProcesses as SP
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

print(portfolioMean)
print(portfolioStdDeviation)
print((previousDayPrice * shares).sum())

ndays = dailyPrices.shape[0] - 1
timeLine = numpy.linspace(0, ndays, ndays)

simulation_df = pandas.DataFrame()
for run in range(1000):
    randVariables = numpy.random.normal(0, 1, int(ndays))
    price = (previousDayPrice * shares).sum() * SP.geometricbrownianWithDrft(portfolioMean, portfolioStdDeviation, ndays, randVariables)[0]
    prices = pandas.Series(price)
    pyplot.plot(timeLine, price)
    simulation_df = simulation_df.append(prices, ignore_index=True)

pyplot.title('Geometric Brownian Motion with Drift')
pyplot.xlabel("Time")
pyplot.ylabel("Price")
pyplot.show()

price_array = simulation_df.iloc[-1, :]
x = price_array
mu = price_array.mean()
sigma = price_array.std()

var = numpy.percentile(price_array, 10)
var1 = numpy.percentile(price_array, 5)
var2 = numpy.percentile(price_array, 1)

q = numpy.percentile(price_array , 1)
num_bins = 20
n, bins, patches = pyplot.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins, mu, sigma)
pyplot.plot(bins, y, 'r--')
pyplot.xlabel('Price')
pyplot.ylabel('Probability')
pyplot.title(r'Histogram of Speculated Stock Prices', fontsize=18, fontweight='bold')
pyplot.subplots_adjust(left=0.15)
pyplot.axvline(x=q, linewidth=4, color='r')
pyplot.show()

print("VaR at 90% Confidence: " + '${:,.2f}'.format((previousDayPrice * shares).sum()- var))
print("VaR at 95% Confidence: " + '${:,.2f}'.format((previousDayPrice * shares).sum() - var1))
print("VaR at 99% Confidence: " + '${:,.2f}'.format((previousDayPrice * shares).sum() - var2))

