import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data from the CSV file
data = pd.read_csv("mfcc_data_pca.csv")

# Separate the features (PCA components) from the labels
X = data.iloc[:, :-3].values

# Initialize an empty list to store the silhouette scores for each value of k
silhouette_scores = []

# Try k from 2 to 10 clusters and calculate the silhouette score for each value of k
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=16)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print("Silhouette score for k=%d: %f" % (k, score))

# Plot the silhouette scores versus the number of clusters (k)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score for Optimal k')
plt.show()
