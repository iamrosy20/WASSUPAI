# Finding the optimal number of clusters k with elbow method, and then applying K-Means Clustering to iris dataset

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
iris = load_iris()
X = iris.data

# Find the optimal k, using elbow method
inertia = []
for n_clusters in range(1, 11):
  kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
  kmeans.fit(X)
  inertia.append(kmeans.inertia_)

# Plot the elbow graph: we can find that optimal k is 3
plt.plot(range(1, 11), inertia)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

# Do KMeans Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Predict with KMeans
y_clusters = kmeans.fit_predict(X)

# Visualize the result
plt.figure(figsize=(12, 6))
colors = np.array(['red', 'green', 'blue'])
plt.scatter(X[:, 0], X[:, 1], c=colors[y_clusters], s=50, cmap='viridis')

# Display centroids of clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("KMeans Clustering on Iris Dataset")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()
