# Scale the breast cancer dataset using StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, and Normalizer

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4301)

# Initialize Scalers
std = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
norm = Normalizer()

# Scale the dataset
# StandardScaler
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
# MinMaxScaler
X_train_mms = mms.fit_transform(X_train)
X_test_mms = mms.transform(X_Test)
# MaxAbsScaler
X_train_mas = mas.fit_transform(X_train)
X_test_mas = mas.transform(X_test)
# RobustScaler
X_train_rbs = rbs.fit_transform(X_train)
X_test_rbs = rbs.transform(X_test)
# Normalizer
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)

# Reshape the training data
X_train_ss = X_train.reshape(13650, 1)            # No scaling
X_train_std_ss = X_train_std.reshape(13650, 1)    # StandardScaler
X_train_mms_ss = X_train_mms.reshape(13650, 1)    # MinMaxScaler
X_train_mas_ss = X_train_mas.reshape(13650, 1)    # MaxAbsScaler
X_train_rbs_ss = X_train_rbs.reshape(13650, 1)    # RobustScaler
X_train_norm_ss = X_train_norm.reshape(13650, 1)  # Normalizer

# Plot distribution of the training data
# No scaling
plt.hist(X_train_ss, bins=30, color='red', alpha=0.7)
plt.title("No scaling")
plt.show()
# StandardScaler
plt.hist(X_train_std_ss, bins=30, alpha=0.7, density=True)
plt.title("StandardScaler")
plt.show()
# MinMaxScaler
plt.hist(X_train_mms_ss, bins=30, alpha=0.7, density=True)
plt.title("MinMaxScaler")
plt.show()
# MaxAbsScaler
plt.hist(X_train_mas_ss, bins=30, alpha=0.6, density=True)
plt.title("MaxAbsScaler")
plt.show()
# RobustScaler
plt.hist(X_train_rbs_ss, bins=30, alpha=0.6, density=True)
plt.title("RobustScaler")
plt.show()
# Normalizer
plt.hist(X_train_norm_ss, bins=30, alpha=0.6, density=True)
plt.title("Normalizer")
plt.show()

# Initialize Decision Tree Classifier
dtc = DecisionTreeClassifier()

# Fit the model
dtc.fit(X_train, y_train)        # No scaling
dtc.fit(X_train_std, y_train)    # StandardScaler
dtc.fit(X_train_mms, y_train)    # MinMaxScaler
dtc.fit(X_train_mas, y_train)    # MaxAbsScaler
dtc.fit(X_train_rbs, y_train)    # RobustScaler
dtc.fit(X_train_norm, y_train)   # Normalizer

# Print accuracy
print("No scaling accuracy: ", round(dtc.score(X_test, y_test), 4))           # 0.9211
print("StandardScaler accuracy: ", round(dtc.score(X_test_std, y_test), 4))   # 0.9123
print("MinMaxScaler accuracy: ", round(dtc.score(X_test_mms, y_test), 4))     # 0.9123
print("MaxAbsScaler accuracy: ", round(dtc.score(X_test_mas, y_test), 4))     # 0.9123
print("RobustScaler accuracy: ", round(dtc.score(X_test_rbs, y_test), 4))     # 0.8860
print("Normalizer accuracy: ", round(dtc.score(X_test_norm, y_test), 4))      # 0.9386
