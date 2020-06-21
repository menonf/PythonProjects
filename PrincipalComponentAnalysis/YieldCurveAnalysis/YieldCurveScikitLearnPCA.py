import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Import Bank of England spot curve data from excel
df = pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",
                   index_col=0, header=3, dtypes="float64", sheet_name="4. spot curve", skiprows=[4])

df = df.iloc[:, 0:20]                                       # Select all of the data up to 10 years
df = df.dropna(how="any")                                   # Drop nan values
scaled_features = StandardScaler().fit_transform(df)
scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

PCA = PCA().fit(scaled_features_df)

plt.plot(np.cumsum(PCA.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

principal_components = scaled_features_df.dot(PCA.components_)
plt.plot(principal_components[0], label="First Principle Component",)
plt.legend()
plt.show()

plt.plot(principal_components[1], label="Second Principle Component",)
plt.legend()
plt.show()

plt.plot(principal_components[2], label="Third Principle Component",)
plt.legend()
plt.show()
