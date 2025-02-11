import numpy as np
import faiss


# Load the dataset
data = np.loadtxt('dataset/fashion-minst/fashion-mnist_test.csv', delimiter=',', dtype=np.float32, skiprows=1)

# Ensure the data is in the correct format
data = data.astype(np.float32)

# Build the FAISS index
d = data.shape[1]  # dimension of the vectors
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(data)  # add vectors to the index

# Perform a similarity search
k = 5  # number of nearest neighbors
query_vector = data[0]  # example query vector
D, I = index.search(query_vector.reshape(1, -1), k)  # search

print("Distances:", D)
print("Indices:", I)