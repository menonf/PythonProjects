import statsmodels.api as stat
import statsmodels.tsa.stattools as ts
import quandl

#fetching financial data for two securities from Quandl
data1 = quandl.get("CHRIS/MCX_AL1", start_date="2016-11-01", api_key='U_PJwA55r5u8Lz_uFJ6L')
data2 = quandl.get("CHRIS/MCX_PB1", start_date="2016-11-01", api_key='U_PJwA55r5u8Lz_uFJ6L')


#performing ADF test on the closing prices of the fetched data
result = stat.OLS(data1['Close'], data2['Close']).fit()
c_t = ts.adfuller(result.resid)

#checking whether the pair of securities is co-integrated and printing the result
print(c_t)

if c_t[0] <= c_t[4]['10%'] and c_t[1] <= 0.1:
    print ("Pair of securities is co-integrated")
else:
    print ("Pair of securities is not co-integrated")
