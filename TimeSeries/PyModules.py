import pandas
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from matplotlib import pyplot
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
import scipy.stats as scs

class Graphs:
    
    def SeasonalPatern(data):  # time series decompositions jbl pg 108
        result = seasonal_decompose(data, model='additive', freq=1)
        return result

    def tsplot(y, lags=None, figsize=(16,12), style='bmh'):
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
            pyplot.show()
        return None

    def fcstPlot(forecast, tsdata, n_steps): # Plot 21 day forecast for SPY returns
        pyplot.style.use('bmh')
        fig = pyplot.figure(figsize=(9, 7))
        ax = pyplot.gca()
        tsdata.plot(ax=ax, label='S&P500 Returns')
        styles = ['b-', '0.2', '0.75', '0.2', '0.75']
        forecast.plot(ax=ax, style=styles)
        pyplot.fill_between(forecast.index, forecast.lower_ci_95, forecast.upper_ci_95, color='gray', alpha=0.7)
        pyplot.fill_between(forecast.index, forecast.lower_ci_99, forecast.upper_ci_99, color='gray', alpha=0.2)
        pyplot.title('{} Day S&P500 Return Forecast\nARIMA{}'.format(n_steps, (4, 0, 3)))
        pyplot.legend(loc='best', fontsize=10)
        pyplot.show()
        return None


class functions:
    def adftest(p_series):# Perform Dickey-Fuller test to test the stationarity: pg 98 jason brown lee
        dftest = adfuller(p_series)
        dfoutput = pandas.Series(dftest[0:4],index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        return dfoutput

    def evaluate_models(dataset, p_values, d_values, q_values):# returns ARIMAResults class
        best_aic = float("inf")
        best_order = None
        best_model = None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        model = ARIMA(dataset, order=order)
                        results = model.fit(method='mle', disp=0)
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = order
                            best_model = results
                    except: continue
        print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
        return best_model

    def forecast(dataset, n_steps, idx): # Create a n day forecast of SPY returns with 95%, 99% CI
        f, err95, ci95 = dataset.forecast(steps=n_steps)  # 95% CI
        _, err99, ci99 = dataset.forecast(steps=n_steps, alpha=0.01)  # 99% CI
        fc_95 = pandas.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
        fc_99 = pandas.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
        fc_all = fc_95.combine_first(fc_99)
        return fc_all