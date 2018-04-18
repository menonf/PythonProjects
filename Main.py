import os                                           # OS module
import platform                                     # OS platform module
import sys                                          # system module
import warnings
import pandas
import numpy as np
import PyModules
import RModules
from matplotlib import pyplot
from arch import arch_model

warnings.filterwarnings("ignore")
print(os.name)
print(platform.system())
print(platform.release())
print(sys.version)

webData = pandas.DataFrame.from_csv("GSPC.csv", header=0)
adjustedClose = webData.iloc[:, 0:1]
PyModules.Graphs.tsplot(y=adjustedClose, lags=20)
adjustedClose.plot(kind='kde')
pyplot.show()
print('\n\nResults of Dickey-Fuller Test adjustedClose:')
print(PyModules.functions.adftest(adjustedClose.iloc[:, 0].values))
lrets=np.log(adjustedClose).diff(periods=1).dropna()
#lrets = np.log(adjustedClose/adjustedClose.shift(1)).dropna()  # log returns

PyModules.Graphs.SeasonalPatern(lrets).plot()
pyplot.show()
PyModules.Graphs.tsplot(y=lrets, lags=20)
lrets.plot(kind='kde')
pyplot.show()
print(lrets.describe())
print('\n\nResults of Dickey-Fuller Test LogReturns:')
print(PyModules.functions.adftest(lrets.iloc[:, 0].values))

p1_values = [4]
d1_values = [0]
q1_values = [3]
bst_mdl = PyModules.functions.evaluate_models(lrets.values, p1_values, d1_values, q1_values)  # constant variance

n_steps = 100
ts = lrets.iloc[-2400:].copy()
idx = pandas.date_range(lrets.index[-1], periods=n_steps, freq='D')
forecast = PyModules.functions.forecast(bst_mdl, n_steps, idx).head()
PyModules.Graphs.fcstPlot(forecast, ts, n_steps)
print(forecast)

PyModules.Graphs.tsplot(bst_mdl.resid, lags=20)  # residual
PyModules.Graphs.tsplot(np.square(bst_mdl.resid), lags=20)
garch11 = arch_model(10*lrets, p=1, o=0, q=1)  # Python GARCH - constant mean with normal distribution
res = garch11.fit(update_freq=5, disp='off')
print(res.summary())
GARCHResiduals = RModules.rfunctions.RGarch(lrets) # R GARCH used for graphing constant mean with normal distribution
PyModules.Graphs.tsplot(GARCHResiduals, lags=20)
PyModules.Graphs.tsplot(np.square(GARCHResiduals), lags=20)