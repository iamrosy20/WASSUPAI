# Caculating sihouette score with k=3, and plotting the result of K-Means Clustering using iris dataset

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silihouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
iris = load_iris()
X = iris.data

# Do KMeans Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_clusters = kmeans.fit_predict(X)

# Caculate Silhouette Score
silhouette_avg = silhouette_score(X, y_clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Allocate cluster number to each cluster point
iris_with_clusters = np.column_stack((iris.data, y_clusters))

# Visualize the result
plt.figure(figsize=(12, 6))
colors = np.array(['red', 'green', 'blue'])
plt.scatter(X[:, 0], X[:, 1], c=colors[y_clusters], s=50, camp='viridis')

# Display centroids of clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("KMeans Clustering on Iris Dataset")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()

# Check features of each clusters
print("\nCluster characteristics:")
for i in range(kmeans.n_clusters):
  cluster_data = iris_with_clusters[iris_with_clusters[:, -1] == i]
  print(f"\nCluster {i} mean values:"
  print(np.mean(cluster_data, axis=0))
