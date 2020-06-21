import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""https://towardsdatascience.com/applying-pca-to-the-yield-curve-4d2023e555b3"""
# Import Bank of England spot curve data from excel
df = pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",
                   index_col=0, header=3, dtypes="float64", sheet_name="4. spot curve", skiprows=[4])

df = df.iloc[:, 0:20]                                       # Select all of the data up to 10 years
df = df.dropna(how="any")                                   # Drop nan values
df_std = ((df - df.mean()) / df.std())                      # Standardise the data in the df into z scores
corr = df_std.corr()                                        # Correlation Matrix --> Covariance matrix (due to Std data)

# Perform EigenDecomposition
eigenvalues, eigenvectors = np.linalg.eig(corr)
df_eigval = pd.DataFrame({"Eigenvalues": eigenvalues}, index=range(1, 21))
df_eigvec = pd.DataFrame(eigenvectors, index=range(1, 21))

# Work out explained proportion
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])
df_eigval.style.format({"Explained proportion": "{:.2%}"})  # Format as percentage

principal_components = df_std.dot(eigenvectors)
plt.plot(principal_components[0], label="First Principle Component",)
plt.plot(df[10],  label="10 Year yield Curve")
plt.legend()
plt.show()

# Calculate 10Y-2M slope
df_s = pd.DataFrame(data=df)
df_s = df_s[[2, 10]]
df_s["slope"] = df_s[10] - df_s[2]

plt.plot(principal_components[1], label="Second Principle Component")
plt.plot(df_s["slope"], label="10Y-2M slope")
plt.legend()
plt.show()

plt.plot(principal_components[2], label="Third Principle Component-Curvature")
plt.legend()
plt.show()
