
# Mall Customer Segmentation

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r'C:\Users\fatai\Desktop\ML AND AI Learning\Mall_Customers.csv') #Update with your path


# Display first few rows
print("First 5 rows of data:")
print(data.head())

# Check info
print("\n Dataset Info:")
print(data.info())

#Select relevant features

# Using 'Annual Income (k$)' and 'Spending Score (1-100)'
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


# Feature scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Determine optimal number of clusters (Elbow Method)

wcss = []  # within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# Apply K-Means with optimal clusters

# Optimal K = 10 (from elbow curve)
kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
data['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=80, label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=80, label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=80, label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=80, label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], s=80, label='Cluster 5')
plt.scatter(X_scaled[y_kmeans == 5, 0], X_scaled[y_kmeans == 5, 1], s=80, label='Cluster 6')
plt.scatter(X_scaled[y_kmeans == 6, 0], X_scaled[y_kmeans == 6, 1], s=80, label='Cluster 7')
plt.scatter(X_scaled[y_kmeans == 7, 0], X_scaled[y_kmeans == 7, 1], s=80, label='Cluster 8')
plt.scatter(X_scaled[y_kmeans == 8, 0], X_scaled[y_kmeans == 8, 1], s=80, label='Cluster 9')
plt.scatter(X_scaled[y_kmeans == 9, 0], X_scaled[y_kmeans == 9, 1], s=80, label='Cluster 10')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=250, c='black', marker='X', label='Centroids')

plt.title('Customer Segments (K-Means Clustering, K=10)')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()

print("\nClustered Data Sample:")
print(data.head())

