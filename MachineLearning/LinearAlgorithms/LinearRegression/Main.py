import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import LeastSquares
import GradientDescent

########################################################################
df = pd.read_csv('MachineLearning\LinearAlgorithms\LinearRegression\LifeExpectancy.csv', names=['LifeEx', 'LogIncome'])
X = df.loc[:, ['LifeEx']]
Y = df.loc[:, ['LogIncome']]

X = sm.add_constant(X)
results = sm.OLS(Y, X)
results = results.fit()
print('\nUsing Statsmodels')
print(results.summary())
print(results.params)

########################################################################
print("\nUsing Sci-Kit Learn")
Model = LinearRegression()
Model.fit(X, Y)
print(Model.coef_)
print(Model.intercept_)

#########################################################################
print('\nSimple Linear Regression(OLS) Method of Estimation from scratch')
filename = 'MachineLearning\LinearAlgorithms\LinearRegression\LifeExpectancy.csv'
dataset = LeastSquares.load_csv(filename)
for i in range(len(dataset[0])):
    LeastSquares.str_column_to_float(dataset, i)
b0, b1 = LeastSquares.coefficients_leastSquares(dataset)
print('Coefficients: B0=%.4f, B1=%.4f' % (b0, b1))

########################################################################
print('\nStochastic Gradient Descent Method of Estimation')
l_rate = 0.001
n_epoch = 1000
coef = GradientDescent.coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)

#########################################################################
print('\nMaximum Likelihood Method of Estimation')


def lik(parameters):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    y_exp = 0
    for i in np.arange(0, len(x)):
        y_exp = m * x + b
    L = (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 / (2 * sigma ** 2) * sum((y - y_exp) ** 2))
    return L


x = np.array(df.loc[:, ['LifeEx']])
y = np.array(df.loc[:, ['LogIncome']])
theta_start = np.array([1, 1, 1])
res = minimize(lik, theta_start, method='Nelder-Mead', options={'disp': True})


print(results)
plt.scatter(x, y)
plt.plot(x, res['x'][0] * x + res['x'][1])
plt.show()

###########################################################################
