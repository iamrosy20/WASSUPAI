# Look for some methods to find outlier: Z-Score, Isolation Forest, Local Outlier Factor, and DBSCAN

# Load wine dataset
from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
wine = pd.DataFrame(data=data.data, columns=data.feature_names)
print(wine)

# ---

### 1. Z-Score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(wine))
threshold = 3
wine_z = wine[(z < threshold).all(axis=1)]
print(wine_z)

# ---

### 2. Isolation Forest
from sklearn.ensemble import IsolationForest

clf = IsolationForest(random_state=42)
pred = clf.fit_predict(wine)
wine_if = wine[pred == 1]
print(wine_if)

# ---

### 3. Local Outlier Factor (LOF)
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor()
pred = lof.fit_predict(wine)
wine_lof = wine[pred == 1]
print(wine_lof)

# ---

### 4. DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=8.0, min_samples=5)
pred = dbscan.fit_predict(wine)
wine_dbscan = wine[pred != -1]
print(wine_dbscan)
