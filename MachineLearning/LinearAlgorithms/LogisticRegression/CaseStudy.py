import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        output_error_signal = target - predictions      # Update weights with log likelihood gradient

        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))    # Print log-likelihood every so often
    return weights


np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

simulated_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

plt.figure(figsize=(12, 8))
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c=simulated_labels)
plt.show()

weights = logistic_regression(simulated_features, simulated_labels, num_steps=50000, learning_rate=5e-5, add_intercept=True)
print(weights)


X_const = sm.add_constant(simulated_features)
logit_model = sm.Logit(simulated_labels, X_const)
result = logit_model.fit()
print(result.summary2())


model = LogisticRegression(fit_intercept=True, C=1e9)
mdl = model.fit(simulated_features, simulated_labels)
print(model.coef_)
