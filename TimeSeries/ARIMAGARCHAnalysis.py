import pandas as pd
import numpy as np
import warnings
from TimeSeries import PyModules
from matplotlib import pyplot
from arch import arch_model
from TimeSeries.PyModules import Graphs

warnings.filterwarnings("ignore")
WebData = pd.read_csv("SP500Historical.csv", header=0, index_col="Date")
DailyClose = WebData['AdjClose']
TSPlot = pd.DataFrame(DailyClose).plot()

plt = Graphs()
plt.plot_axis(ts_ax=TSPlot)

plt.SeasonalPattern(DailyClose)
print('\n\nResults of Dickey-Fuller Test on AdjustedClose:')
print(PyModules.Functions.adftest(DailyClose.values))
plt.tsplot(y=DailyClose, lags=20)

LogReturns = np.log(DailyClose).diff(periods=1).dropna()
logrets = LogReturns.plot(kind='kde')
plt.plot_axis(logrets)
pyplot.show()

plt.tsplot(y=LogReturns, lags=20)
print('\n\nResults of Dickey-Fuller Test LogReturns:')
print(PyModules.Functions.adftest(LogReturns.values))

PList = [4]   # PList = [0, 1, 2, 3, 4, 5]  Lag order of the symmetric innovation
DList = [0]   # Difference or Lag order of the asymmetric innovation
QList = [3]   # QList = [0, 1, 2, 3, 4, 5] Lag order of lagged volatility

BestARIMAModel = PyModules.Functions.evaluate_models(LogReturns, PList, DList, QList)  # constant variance
OrderPDQ = BestARIMAModel[1]
print(OrderPDQ)
ARIMAModelOutput = BestARIMAModel[2]
print(ARIMAModelOutput.summary())
plt.tsplot(ARIMAModelOutput.resid, lags=20)  # residual
plt.tsplot(np.square(ARIMAModelOutput.resid), lags=20)

GARCHModel = arch_model(ARIMAModelOutput.resid, p=OrderPDQ[0], o=OrderPDQ[1], q=OrderPDQ[2], dist='StudentsT')
GARCHModelOutput = GARCHModel.fit()
print(GARCHModelOutput.summary())
plt.tsplot(GARCHModelOutput.std_resid, lags=20)
plt.tsplot(np.square(GARCHModelOutput.std_resid), lags=20)
