# 4 representative sampling techniques

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Create random dataset for simple random sampling, systematic sampling, and stratified sampling
data = pd.DataFrame({
    'id': range(1, 101),
    'value': np.random.rand(100)
})

# Simple Random Sampling
sample_random = data.sample(n=10)
print(sample_random)

# Systematic Sampling
k = len(data) // 10     # interval = 10
start = np.random.randint(0, k)     # choose start point randomly
sample_systematic = data.iloc[start::k]
print(sample_systematic)

# Stratified Sampling
data['strata'] = np.where(data['value'] >= data['value'].median(), 1, 0)   # split the dataset
sss = StratifiedShuffleSplit(n_splits=1, test_size=10, random_state=0)
for train_index, test_index in sss.split(data, data['strata']):
    sample_stratified = data.iloc[test_index]
print(sample_stratified)

# Create random dataset for cluster sampling
np.random.seed(0)
data = pd.DataFrame({
    'id': range(1, 101),
    'value': np.random.rand(100),
    'cluster': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100)     # allocate cluster randomly
})

# Cluster Sampling
selected_clusters = np.random.choice(['A', 'B', 'C', 'D', 'E'], 2, replace=False)   # choose two clusters randomly
sample_cluster = data[data['cluster'].isin(selected_clusters)]
print(sample_cluster)
