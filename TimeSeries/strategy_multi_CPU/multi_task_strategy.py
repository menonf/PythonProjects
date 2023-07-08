import numpy as np
import statsmodels.api as sm
import multiprocessing
import my_arima_worker  # Import the worker function from another module

import pandas as pd
from matplotlib import pyplot
from arch import arch_model
from pylab import *
from PyModules import Graphs
import PyModules
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    WebData = pd.read_csv("E:\Repositories\PythonProjects\TimeSeries\SP500LastThreeYears.csv", header=0, index_col="Date")
    DailyClose = WebData['AdjClose']
    LogReturns = np.log(DailyClose).diff(periods=1).dropna()        # Differenced TimeSeries


    # Define the list of ARIMA orders to try
    p_values = [0,1,2,3,4,5]
    d_values = [0,1,2,3]
    q_values = [0,1,2,3,4,5]

    # Define the number of CPU cores to use
    num_cpus = multiprocessing.cpu_count()

    # Divide the list of ARIMA orders into num_cpus sublists
    pdq_sublists = [(p, d, q) for p in p_values for d in d_values for q in q_values]
    pdq_sublists = [pdq_sublists[i::num_cpus] for i in range(num_cpus)]

    # Create a pool of worker processes and map each sublist of ARIMA orders to a worker process
    with multiprocessing.Pool(num_cpus) as pool:
        results = pool.starmap(my_arima_worker.worker, [(LogReturns, pdq_sublist) for pdq_sublist in pdq_sublists])


    RollingWindowLength = 100
    ForecastLength = len(LogReturns) - RollingWindowLength
    Signal = 0 * LogReturns[-ForecastLength:]

    df = pd.DataFrame(columns=['BestOrder'])

    for i in range(ForecastLength):
        TimeSeries = LogReturns[(1 + i):(RollingWindowLength + i)]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.starmap(my_arima_worker.worker, [(TimeSeries, pdq_sublist) for pdq_sublist in pdq_sublists])

        # Combine the results from each worker process into a single list
        arima_params = []
        for result in results:
            arima_params.extend(result)

        min_value = min(row[0] for row in arima_params)
        BestARIMAModel = [row for row in arima_params if row[0] == min_value]
        #print([row[2].resid for row in BestARIMAModel])
        #print([row[1] for row in BestARIMAModel])
        GARCHModel = arch_model([row[2].resid for row in BestARIMAModel], p=1, o=0, q=1)
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