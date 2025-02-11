import numpy as np
import faiss


# Load thedataset
data = np.loadtxt('dataset/fashion-minst/fashion-mnist_test.csv', delimiter=',', dtype=np.float32, skiprows=1)
data = data.astype(np.float32)

      
def create_faiss_index(dataset_path, algo):
    """
    Create and return a FAISS FLATL2 index from feature vectors.

    Args:
    - feature_vectors: Numpy array of feature vectors.
    - use_quantization: Boolean to enable or disable quantization.

    Returns:
    - A faiss Index object representing the created index.
    """
    data = np.loadtxt(dataset_path, delimiter=',', dtype=np.float32, skiprows=1)
    data = data.astype(np.float32)
    d = data.shape[1]
    if algo == 'IVFFlat':
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 100)
        if not index.is_trained:
            index.train(data)
    if algo == 'HSNW':
        index = faiss.IndexHNSWFlat(d, 32)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(data)
    return index
    
def save_faiss_index(index, index_path):
    """
    Save the FAISS index to disk.

    Args:
    - index: The FAISS index to save.
    - index_path: Path to the file where to save the index.
    """
    faiss.write_index(index, index_path)
    
# Build the FAISS index
index = create_faiss_index('dataset/fashion-minst/fashion-mnist_test.csv', 'HNSW')
save_faiss_index(index,'ind')

k = 5  
query_vector = data[0] 
D, I = index.search(query_vector.reshape(1, -1), k) 

print("Distances:", D)
print("Indices:", I)

