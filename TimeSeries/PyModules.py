import pandas
import scipy.stats as scs
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import *

class Graphs:

    def SeasonalPattern(self,data):
        result = seasonal_decompose(data, model='additive', period=1)
        with pyplot.style.context(style='bmh'):
            pyplot.figure(figsize=(16, 14))
            layout = (3, 1)
            trend = pyplot.subplot2grid(layout, (0, 0), colspan=2)
            ses = pyplot.subplot2grid(layout, (1, 0))
            res = pyplot.subplot2grid(layout, (2, 0))
            result.trend.plot(ax=trend).set_title('Trend')
            result.resid.plot(ax=res).set_title('Residuals')
            result.seasonal.plot(ax=ses).set_title('Seasonality')
            self.plot_axis(trend)
            self.plot_axis(ses)
            self.plot_axis(res)
            pyplot.show()
        return None

    def tsplot(self, y, lags=None, figsize=(16, 12), style='bmh'):
        if not isinstance(y, pandas.DataFrame):
            y = pandas.DataFrame(y)
        with pyplot.style.context(style):
            fig = pyplot.figure(figsize=figsize)
            layout = (3, 2)
            ts_ax = pyplot.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = pyplot.subplot2grid(layout, (1, 0))
            pacf_ax = pyplot.subplot2grid(layout, (1, 1))
            qq_ax = pyplot.subplot2grid(layout, (2, 0))
            pp_ax = pyplot.subplot2grid(layout, (2, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            tsaplots.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
            tsaplots.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
            sm.qqplot(y.iloc[:, 0].values, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(y.iloc[:, 0].values, sparams=(y.mean(), y.std()), plot=pp_ax)
            pyplot.tight_layout()
            self.plot_axis(ts_ax)
            self.plot_axis(acf_ax)
            self.plot_axis(pacf_ax)
            self.plot_axis(pp_ax)
            self.plot_axis(qq_ax)
            pyplot.show()
        return None

    def plot_axis(self, ts_ax):
        for axis in ['top', 'bottom', 'left', 'right']:
            ts_ax.spines[axis].set_linewidth(2)
        for tick in ts_ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ts_ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        return None

class Functions:
    def adftest(p_series):  # Perform Dickey-Fuller test to test the stationarity: pg 98 jason brown lee
        dftest = adfuller(p_series)
        dfoutput = pandas.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        return dfoutput

    def evaluate_models(dataset, p_values, d_values, q_values):  # returns ARIMAResults class
        best_aic = float("inf")
        best_order = None
        best_model = None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        model = ARIMA(dataset, order=order)
                        results = model.fit(method='mle', trend='nc')
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = order
                            best_model = results
                    except:
                        continue
        print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
        return best_aic, best_order, best_model
