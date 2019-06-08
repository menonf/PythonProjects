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