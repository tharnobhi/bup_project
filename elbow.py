import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data from the CSV file
data = pd.read_csv("mfcc_data_pca.csv")

# Separate the features (PCA components) from the labels
X = data.iloc[:, :-3].values

# Initialize an empty list to store the within-cluster sum of squares (inertia) for each value of k
inertias = []

# Try k from 1 to 10 clusters and calculate the within-cluster sum of squares (inertia) for each value of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=8)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the within-cluster sum of squares (inertia) versus the number of clusters (k)
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster sum of squares (inertia)')
plt.xticks(range(1, 11))  # modify x-axis ticks
plt.title('Elbow Method for Optimal k')
plt.show()
