import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

def glm_mcmc_inference(df, iterations):
    basic_model = pm.Model()
    with basic_model:
        pm.GLM.from_formula('y ~ x', df)
        start = pm.find_MAP()
        step = pm.NUTS()  # Use the No-U-Turn Sampler
        trace = pm.sample(iterations, step, start,random_seed=42, progressbar=True)     # Calculate the trace
    return trace

if __name__ == "__main__":
    df = pd.read_csv('LifeExpectancy.csv', names=['y', 'x'])
    sns.lmplot(x='x', y='y', data=df, size=10, ci=None)
    plt.show()

    trace = glm_mcmc_inference(df, iterations=5000)
    pm.traceplot(trace[500:])
    plt.show()

    # Plot a sample of posterior regression lines
    sns.lmplot(x='x', y='y', data=df, ci=None, size=10, fit_reg=True)
    pm.plot_posterior_predictive_glm(trace, samples=100, c='lightgreen', label='Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.show()
