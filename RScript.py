import warnings
import pandas
import numpy as np
import rpy2.robjects.packages as rpackages

#------INSTALL R PACKAGES REQUIRED
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
#packnames = ('quantmod','timeSeries','rugarch')
#utils.install_packages(StrVector(packnames))
#-------------------------------------------

# ------LOAD R PACKAGES REQUIRED
utils = rpackages.importr('quantmod')
utils = rpackages.importr('lattice')
utils = rpackages.importr('timeSeries')
utils = rpackages.importr('rugarch')

warnings.filterwarnings("ignore")
webData = pandas.DataFrame.from_csv("GSPC.csv", header=0)
adjustedClose = webData.iloc[:, 0:1]
p_dataframe  = np.log(adjustedClose/adjustedClose.shift(1)).dropna()

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
p_dataframe.index = p_dataframe.index.strftime('%m-%d-%Y')
r_dataframe = pandas2ri.py2ri(p_dataframe)

RScript= r( '''
            func1 <- function(r) 
            {
                data = r
                dates <- as.Date(as.character(row.names(data)), format="%m-%d-%Y")
                lgReturns <- data$AdjClose
                windowLength = 100
                foreLength = length(lgReturns) - windowLength
                forecasts <- vector(mode="numeric", length=foreLength) 
                forecasts.ts <- xts(lgReturns, dates[0:length(lgReturns)])
                print(forecasts.ts)
            }
            ''')

r_f = r['func1']
result = r_f(r_dataframe)