import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.DataFrame(data=np.random.normal(0, 1, (20, 10)))

pca = PCA(n_components=5)
print(pd.DataFrame(pca.components_))