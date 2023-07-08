import warnings
import quandl
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np


from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)


warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


api_key = 'puxYR3syR5S-DyzWszgu'
oil = quandl.get('EIA/PET_RWTC_D', api_key=api_key).squeeze()

oil.plot(lw=2, title='WTI Crude Oil Price', figsize=(12, 4))
sns.despine()
plt.tight_layout()
plt.show()