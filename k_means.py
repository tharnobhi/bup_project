import pandas as pd
from sklearn.cluster import KMeans

# Load the data from the PCA-transformed CSV file
data_pca = pd.read_csv("mfcc_data_pca.csv")

# Separate the features (principal components) from the labels
X = data_pca.iloc[:, :-3].values

# Perform K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=16)
kmeans.fit(X)

# Add the cluster labels to the data frame
data_pca["cluster"] = kmeans.labels_

# Save the data frame to a new CSV file
data_pca.to_csv("mfcc_data_clusters_k3.csv", index=False)
