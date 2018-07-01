import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression


df = pd.read_csv('insurance.csv', names= ['LifeEx', 'LogIncome'])
results1 = smf.ols(formula='LogIncome ~ LifeEx', data=df).fit()
print ('\nStatsmodel.Formula.Api Method')
print (results1.params)


X = df.loc[:,['LifeEx']]
Y = df.loc[:,['LogIncome']]
X = sm.add_constant(X)
results2= sm.OLS(Y,X)
results2 = results2.fit()
print ('\nStatsmodel.Api Method')
print (results2.params)


results3 = LinearRegression(fit_intercept='false')
results3.fit(X, Y)
print ("\nSci-Kit Learn Method")

print (results3.coef_)
print (results3.intercept_)