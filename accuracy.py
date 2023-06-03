import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the MFCC data from the CSV file
data = pd.read_csv("mfcc_data_pca.csv")

# Extract the MFCC features and labels from the data
X = data.iloc[:, :-3]
y = data.iloc[:, -1]

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=16).fit(X)

# Extract the cluster labels
labels = kmeans.labels_

# Calculate the accuracy of the clustering
correct = 0
for i in range(len(y)):
    if y[i].split("_")[0] == "arnob" and labels[i] == 0:
        correct += 1
    elif y[i].split("_")[0] == "marzia" and labels[i] == 1:
        correct += 1
    elif y[i].split("_")[0] == "thirdPerson" and labels[i] == 2:
        correct += 1

accuracy = correct / len(y) * 100  # Calculate accuracy as percentage
print("Accuracy: {:.2f}%".format(accuracy))

# Plot the results
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("MFCC 1")
plt.ylabel("MFCC 2")
plt.show()
