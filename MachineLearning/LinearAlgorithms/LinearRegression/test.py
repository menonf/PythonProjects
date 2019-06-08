# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import pymc3 as pm
# import seaborn as sns
#
# def glm_mcmc_inference(df, iterations):
#     basic_model = pm.Model()
#     with basic_model:
#         pm.GLM.from_formula('y ~ x', df, family=pm.glm.families.Normal())
#         start = pm.find_MAP()
#         step = pm.NUTS()  # Use the No-U-Turn Sampler
#         trace = pm.sample(iterations, step, start, progressbar=True)     # Calculate the trace
#
#     return trace
#
# if __name__ == "__main__":
#     df = pd.read_csv('LifeExpectancy.csv', names=['y', 'x'])
#     sns.lmplot(x='x', y='y', data=df)
#     plt.show()
#
#     trace = glm_mcmc_inference(df, iterations=5000)
#     pm.traceplot(trace[500:])
#     plt.show()
#
#     DataPoints = df.count(axis = 'rows')[0]
#
#     # Plot a sample of posterior regression lines
#     #sns.lmplot(x="x", y="y", data=df, size=10, fit_reg=False)
#     pm.plot_posterior_predictive_glm(trace, samples=100,  label='posterior predictive regression lines')
#     plt.legend(loc=0)
#     plt.show()
#
#     plt.plot(trace)
eval = np.linspace(0, 100, 100)

print(eval)