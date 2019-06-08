import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numdifftools as ndt


def lik(parameters):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    for i in np.arange(0, len(x)):
        y_exp = m * x + b
    L = (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 /  (2 * sigma ** 2) * sum((y - y_exp) ** 2))
    return L



df = pd.read_csv('LifeExpectancy.csv', names= ['LifeEx', 'LogIncome'])
X = df.loc[:,['LifeEx']]
Y = df.loc[:,['LogIncome']]

x = np.array(X)
y = np.array(Y)

theta_start = np.array([1,1,1])
res = minimize(lik, theta_start , method = 'Nelder-Mead', options={'disp': True})

Hfun = ndt.Hessian(lik, full_output=True)
hessian_ndt, info = Hfun(res['x'])
se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
results = pd.DataFrame({'parameters':res['x'],'std err':se})
results.index=['Intercept','Slope','Sigma']
print(results)

plt.scatter(x,y)
plt.plot(x, res['x'][0] * x + res['x'][1])
plt.show()

#http://rlhick.people.wm.edu/posts/estimating-custom-mle.html
#https://stackoverflow.com/questions/29324222/how-can-i-do-a-maximum-likelihood-regression-using-scipy-optimize-minimize