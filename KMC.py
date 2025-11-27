import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
dataset = pd.read_csv('/Users/abc/Downloads/Mall_Customers.csv')
print(dataset)

# Select the features
x = dataset.iloc[:, [3, 4]].values
print(x)

# Elbow method
WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), WCSS, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Apply k-means to the dataset
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_means = kmeans.fit_predict(x)
print(y_means)

# Visualizing the clusters
plt.figure(figsize=(10, 5))
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='yellow', label='Cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s=100, c='green', label='Cluster 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s=100, c='black', label='Cluster 5')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#heirarchial clustering


