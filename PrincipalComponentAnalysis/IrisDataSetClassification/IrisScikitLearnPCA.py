from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd


# https://plotly.com/python/v3/ipython-notebooks/principal-component-analysis/
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)  # drops the empty line at file-end

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

X_std = StandardScaler().fit_transform(X)
sklearn_pca = sklearnPCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)

data = []
colors = {'Iris-setosa': '#0D76BF', 'Iris-versicolor': '#00cc96', 'Iris-virginica': '#EF553B'}
for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):
    trace = dict(
        type='scatter3d',
        x=Y_sklearn[y == name, 0],
        y=Y_sklearn[y == name, 1],
        z=Y_sklearn[y == name, 2],
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
