import statsmodels.api as sm

def worker(data, pdq_list):
    results = []
    results_aic = []
    for pdq in pdq_list:
        p, d, q = pdq
        try:
            model = sm.tsa.arima.ARIMA(data, order=(p, d, q))
            arima_results = model.fit()
            results.append([arima_results.aic, pdq, arima_results])
        except:
            pass
    return results
