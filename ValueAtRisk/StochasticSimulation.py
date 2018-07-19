import numpy
from ValueAtRisk import StochasticProcesses as SP
from matplotlib import pyplot

n = 1000
randVariables = numpy.random.normal(0, 1, int(n))
timeLine = numpy.linspace(0, n, n)                                          # Arranges the number line in linear space

w, b = SP.brownian(n, randVariables)                                        # brownian motion
bd, drift, bdi = SP.brownianWithDrft(0.3, 1.5, n, randVariables)            # brownian motion with drift
gbd, gdf, gbi = SP.geometricbrownianWithDrft(0.3, 1.5, n, randVariables)    # GBM with drift

rw = pyplot.plot(timeLine, b, label='Random Walk')
bm = pyplot.plot(timeLine, w, label='Brownian Motion')
bmd = pyplot.plot(timeLine, bd, label='Brownian Motion with Drift')
gbm = pyplot.plot(timeLine, gbd, label='GBM with Drift')
dft = pyplot.plot(timeLine, drift, label='Drift')

pyplot.title('Stochastic Process Paths')
pyplot.xlabel("Time")
pyplot.ylabel("Motion")
pyplot.legend()
pyplot.show()

for run1 in range(1000):
    randVariables = numpy.random.normal(0, 1, int(n))
    brownian = SP.brownian(n, randVariables)[1]
    pyplot.plot(timeLine, brownian)

pyplot.title('Brownian Motion')
pyplot.xlabel("Time")
pyplot.ylabel("Price")
pyplot.show()

for run2 in range(1000):
    randVariables = numpy.random.normal(0, 1, int(n))
    brownianDrift = SP.brownianWithDrft(0.3, 1.5, n, randVariables)[0]
    pyplot.plot(timeLine, brownianDrift)

pyplot.title('Brownian Motion with Drift')
pyplot.xlabel("Time")
pyplot.ylabel("Price")
pyplot.show()

for run3 in range(1000):
    randVariables = numpy.random.normal(0, 1, int(n))
    GBMDrift = SP.geometricbrownianWithDrft(0.3, 1.5, n, randVariables)[0]
    pyplot.plot(timeLine, GBMDrift)

pyplot.title('Geometric Brownian Motion with Drift')
pyplot.xlabel("Time")
pyplot.ylabel("Price")
pyplot.show()
