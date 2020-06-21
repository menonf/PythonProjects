from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


"""https://plotly.com/python/v3/ipython-notebooks/principal-component-analysis/"""
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)  # drops the empty line at file-end

# Split data table into data X and class labels y
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Standardising
X_std = StandardScaler().fit_transform(X)

# Covariance Matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)

# Perform an eigendecomposition on the covariance matrix:
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Perform a Singular Vector Decomposition (SVD) to improve the computational efficiency
u, s, v = np.linalg.svd(X_std.T)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Explained variance ratio
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
plt.plot(np.cumsum(var_exp))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Reduce the 4-dimensional feature space to a 3-dimensional feature subspace, by choosing the "top 3" eigenvectors
# with the highest eigenvalues to construct our d×k-dimensional eigenvector matrix W.
matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),  eig_pairs[1][1].reshape(4, 1),  eig_pairs[2][1].reshape(4, 1)))

# Projection Onto the New Feature Space
# In this last step we will use the 4×2-dimensional projection matrix W to transform our samples onto the new subspace
# via the equation Y=X×W, where Y is a 150×2 matrix of our transformed samples
Y = X_std.dot(matrix_w)
data = []
colors = {'Iris-setosa': '#0D76BF', 'Iris-versicolor': '#00cc96', 'Iris-virginica': '#EF553B'}
for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):
    trace = dict(
        type='scatter3d',
        x=Y[y == name, 0],
        y=Y[y == name, 1],
        z=Y[y == name, 2],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
    showlegend=True,
    scene=dict(
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2'),
        zaxis=dict(title='PC3')
    )
)

fig = go.Figure(data=data, layout=layout)
fig.show()
