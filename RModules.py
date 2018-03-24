import warnings
import pandas
import numpy as np
import rpy2.robjects.packages as rpackages

#------INSTALL R PACKAGES REQUIRED
#utils = rpackages.importr('utils')
#utils.chooseCRANmirror(ind=1)
#packnames = ('quantmod','timeSeries','rugarch','tseries')
#from rpy2.robjects.vectors import StrVector
#utils.install_packages(StrVector(packnames))
#-------------------------------------------

# ------LOAD R PACKAGES REQUIRED
utils = rpackages.importr('quantmod')
utils = rpackages.importr('lattice')
utils = rpackages.importr('timeSeries')
utils = rpackages.importr('rugarch')
utils = rpackages.importr('tseries')
warnings.filterwarnings("ignore")
#
#
class rfunctions:
    def RGarch(p_dataframe):
        from rpy2.robjects import r, pandas2ri
        pandas2ri.activate()
        p_dataframe.index = p_dataframe.index.strftime('%m/%d/%Y')
        r_dataframe = pandas2ri.py2ri(p_dataframe)
        dfoutput = pandas.Series()
        RScript = r('''
                    func1 <- function(r)
                    {
                        data = r
                        dates <- as.Date(as.character(row.names(data)), format="%m/%d/%Y")
                        lgReturns <- data$AdjClose
                        ft.garch <- garch(lgReturns, trace=F)
                        ft.res <- ft.garch$res[-1]
                    }
                    ''')

        r_f = r['func1']
        dfoutput= r_f(r_dataframe)
        return np.array(dfoutput)

    def ARIMAGARCH(p_dataframe):
        from rpy2.robjects import r, pandas2ri
        pandas2ri.activate()
        p_dataframe.index = p_dataframe.index.strftime('%m/%d/%Y')
        r_dataframe = pandas2ri.py2ri(p_dataframe)
        dfoutput = pandas.Series()
        RScript = r('''
                    func1 <- function(r) 
                    {
                        data = r
                        dates <- as.Date(as.character(row.names(data)), format="%m/%d/%Y")
                        lgReturns <- data$AdjClose
                        windowLength = 100
                        foreLength = length(lgReturns) - windowLength 
                        forecasts <- vector(mode="numeric", length=foreLength) 
                        directions <- vector(mode="numeric", length=foreLength) 
                        p.val <- vector(mode="numeric", length=foreLength) 
                        
                        for (i in 0:foreLength) 
                        {
                          lgReturnsOffset = lgReturns[(1 + i):(windowLength + i)]
                          final.order <- c(4, 0, 3)
                          
                          spec = ugarchspec(mean.model = list(armaOrder = c(final.order[1], final.order[3]),include.mean = T),
                                            variance.model = list(garchOrder = c(1, 1)), distribution.model = "sged")
                          
                          fit = tryCatch(ugarchfit(spec, lgReturnsOffset, solver = 'hybrid'),error = function(e)e, warning = function(w)w)
                          
                          if (is(fit, "warning")) 
                          {
                            forecasts[i+1] <- 0 
                            p.val[i+1] <- 0
                          }
                          else 
                          {
                            next.day.fore = ugarchforecast(fit, n.ahead = 1)
                            x = next.day.fore@forecast$seriesFor
                            directions[i+1] <- ifelse(x[1] > 0, 1, -1) # directional prediction only
                            forecasts[i+1] <- x[1] # actual value of forecast
                          }
                        }
                        
                        forecasts.ts <- xts(forecasts, dates[(windowLength):length(lgReturns)])
                        print(forecasts.ts)
                    }
                    ''')

        r_f = r['func1']
        dfoutput= r_f(r_dataframe)
        return np.array(dfoutput)