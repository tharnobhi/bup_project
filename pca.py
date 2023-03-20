import pandas as pd
from sklearn.decomposition import PCA

# Load the data from the CSV file
data = pd.read_csv("mfcc_data.csv")

# Separate the features (MFCCs) from the labels
X = data.iloc[:, :-3].values
a = data.iloc[:, -3:-2].values
b = data.iloc[:, -2:-1].values
c = data.iloc[:, -1].values


# Normalize the features (subtract the mean and divide by the standard deviation)
X_norm = (X - X.mean()) / X.std()

# Perform PCA with 10 components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_norm)

# Print the explained variance ratio of each component
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Save the PCA-transformed data and labels to a new CSV file
data_pca = pd.DataFrame(X_pca, columns=["pc_" + str(i) for i in range(10)])
data_pca["date"] = a
data_pca["time"] = b
data_pca["label"] = c

data_pca.to_csv("mfcc_data_pca.csv", index=False)
