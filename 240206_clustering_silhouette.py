# Finding the optimal cluster number k, using silhouette score

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate synthetic data for example
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# Range of clusters to try
range_n_clusters = list(range(2, 11))

# Perform KMeans clustering for different numbers of clusters
silhouette_avg_scores = []
for n_clusters in range_n_clusters:
  clusterer = KMeans(n_clusters = n_clusters, random_state=10)
  cluster_labels = clusterer.fit_predict(X)
  
  # Caculate the average silhouette score
  silhouette_avg = silhouette_score(X, cluster_labels)
  silhouette_avg_scores.append(silhouette_avg)
  print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
plt.title("Silhouette Scores for Various Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Average Silhouette Score")
plt.show()
