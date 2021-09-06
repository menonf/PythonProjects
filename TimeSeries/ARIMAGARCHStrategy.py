import pandas as pd
from TimeSeries import PyModules
from matplotlib import pyplot
from arch import arch_model
from pylab import *
from TimeSeries.PyModules import Graphs

WebData = pd.read_csv("SP500LastThreeYears.csv", header=0, index_col="Date")
DailyClose = WebData['AdjClose']
LogReturns = np.log(DailyClose).diff(periods=1).dropna()        # Differenced TimeSeries

PList = [0, 1, 2, 3, 4, 5]             # Lag order of the symmetric innovation
DList = [0]                            # Difference or Lag order of the asymmetric innovation
QList = [0, 1, 2, 3, 4, 5]             # Lag order of lagged volatility

RollingWindowLength = 25
ForecastLength = len(LogReturns) - RollingWindowLength
Signal = 0 * LogReturns[-ForecastLength:]

for i in range(ForecastLength):
    TimeSeries = LogReturns[(1 + i):(RollingWindowLength + i)]
    BestARIMAModel = PyModules.Functions.evaluate_models(TimeSeries, PList, DList, QList)
    OrderPDQ = BestARIMAModel[1]
    ModelOutput = BestARIMAModel[2]

    GARCHModel = arch_model(ModelOutput.resid, p=1, o=OrderPDQ[1], q=1)
    Result = GARCHModel.fit(update_freq=10, disp='off')
    ForecastOutput = Result.forecast(horizon=1, start=None, align='origin')
    Signal.iloc[i] = np.sign(ForecastOutput.mean['h.1'].iloc[-1])

Returns = pd.DataFrame(index=Signal.index, columns=['Buy and Hold', 'Strategy'])
Returns['Buy and Hold'] = LogReturns[-ForecastLength:]
Returns['Signal'] = Signal
Returns['Strategy'] = Returns['Signal'] * Returns['Buy and Hold']

CumulativeReturns = pd.DataFrame(index=Signal.index, columns=['Buy and Hold', 'ARIMA-GARCH Strategy'])
CumulativeReturns['Buy and Hold'] = Returns['Buy and Hold'].cumsum() + 1
CumulativeReturns['ARIMA-GARCH Strategy'] = Returns['Strategy'].cumsum() + 1
CumulativeReturns['ARIMA-GARCH Strategy'].plot(figsize=(16, 8), color='crimson')
fig = CumulativeReturns['Buy and Hold'].plot()

# Plotting the Strategy
ax = gca()
plt = Graphs()
plt.plot_axis(ts_ax=fig)
pyplot.xlabel('Date', fontsize=12, fontweight='bold')
pyplot.ylabel('Returns', fontsize=12, fontweight='bold')
leg = pyplot.legend()
for line in leg.get_lines():
    line.set_linewidth(2)
for text in leg.get_texts():
    text.set_fontsize('large')
    text.set_fontweight('bold')
pyplot.show()
