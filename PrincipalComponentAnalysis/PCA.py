import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Bank of England spot curve data from excel
df = pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",
                   index_col=0, header=3, dtypes="float64", sheet_name="4. spot curve", skiprows=[4])

df = df.iloc[:, 0:20]                                       # Select all of the data up to 10 years
df = df.dropna(how="any")                                   # Drop nan values
df_std = ((df - df.mean()) / df.std())                      # Standardise the data in the df into z scores
corr_matrix_array = np.array(np.cov(df_std, rowvar=False))  # Covariance matrix --> Correlation Matrix(due to Std data)

# Perform EigenDecomposition
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix_array)
df_eigval = pd.DataFrame({"Eigenvalues": eigenvalues}, index=range(1, 21))
df_eigvec = pd.DataFrame(eigenvectors, index=range(1, 21))

# Work out explained proportion
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])
df_eigval.style.format({"Explained proportion": "{:.2%}"})  # Format as percentage

principal_components = df_std.dot(eigenvectors)
plt.plot(principal_components[0])
plt.title("First Principle Component")
plt.show()
plt.plot(principal_components[1])
plt.title("Second Principle Component")
plt.show()
plt.plot(principal_components[2])
plt.title("Third Principle Component")
plt.show()

# Calculate 10Y-2M slope
df_s = pd.DataFrame(data=df)
df_s = df_s[[2, 10]]
df_s["slope"] = df_s[10] - df_s[2]
df_s.plot()
plt.show()
