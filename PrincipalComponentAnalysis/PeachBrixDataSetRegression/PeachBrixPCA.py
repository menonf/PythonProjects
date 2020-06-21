import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

"""https://nirpyresearch.com/principal-component-regression-python/"""
data = pd.read_csv('peach_spectra+brixvalues.csv')
X = data.values[:, 1:]
y = data['Brix']

wl = np.arange(1100, 2300, 2)  # wavelengths
# Plot absorbance spectra
with plt.style.context('ggplot'):
    plt.plot(wl, X.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
plt.show()


def pcr(X, y, pc):
    """ Principal Component Regression in Python """
    """Step 1: PCA on input data"""

    # Define the PCA object
    pca = PCA()

    # Preprocessing (1): first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Preprocess (2) Standardize features by removing the mean and scaling to unit variance
    Xstd = StandardScaler().fit_transform(d1X[:, :])

    # Run PCA producing the reduced variable Xred and select the first pc components
    Xreg = pca.fit_transform(Xstd)[:, :pc]

    """Step 2: regression on selected principal components"""

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Fit
    regr.fit(Xreg, y)

    # Calibration
    y_c = regr.predict(Xreg)

    # Cross-validation
    y_cv = cross_val_predict(regr, Xreg, y, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    return y_cv, score_c, score_cv, mse_c, mse_cv


predicted, r2r, r2cv, mser, mscv = pcr(X,y, pc=6)
# Regression plot
z = np.polyfit(y, predicted, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y, predicted, c='red', edgecolors='k')
ax.plot(y, z[1]+z[0]*y, c='blue', linewidth=1)
ax.plot(y, y, color='green', linewidth=1)
plt.title('$R^{2}$ (CV): '+str(r2cv))
plt.xlabel('Measured $^{\circ}$Brix')
plt.ylabel('Predicted $^{\circ}$Brix')
plt.show()