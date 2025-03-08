import numpy as np
import h5py
import multiprocessing
from tqdm import tqdm
from utils.load_dataset import load_dataset

def compute_exact_neighbors_chunk(start, end, data, queries, k):
    """Calcule les vrais voisins exacts pour un sous-ensemble de requêtes."""
    true_neighbors_chunk = np.zeros((end - start, k), dtype=np.int32)
    
    for i, query in enumerate(tqdm(queries[start:end], desc=f"Calcul {start}-{end}")):
        distances = np.linalg.norm(data - query, axis=1)
        true_neighbors_chunk[i] = np.argsort(distances)[:k]  
    
    return start, true_neighbors_chunk

def compute_exact_neighbors(data, queries, k=10, num_workers=4, save_path="results/ground_truth.hdf5"):
    """Calcule et sauvegarde les vrais voisins en parallèle."""
    num_queries = queries.shape[0]
    chunk_size = num_queries // num_workers
    
    pool = multiprocessing.Pool(num_workers)
    results = []

    for i in range(num_workers):
        start = i * chunk_size
        end = num_queries if i == num_workers - 1 else (i + 1) * chunk_size
        results.append(pool.apply_async(compute_exact_neighbors_chunk, (start, end, data, queries, k)))

    pool.close()
    pool.join()

    true_neighbors = np.zeros((num_queries, k), dtype=np.int32)
    for res in results:
        start, chunk = res.get()
        true_neighbors[start:start+chunk.shape[0]] = chunk

    with h5py.File(save_path, "w") as f:
        f.create_dataset("true_neighbors", data=true_neighbors)

if __name__ == "__main__":
    dataset_path = "datasets/fashion-mnist-784-euclidean.hdf5"
    data = load_dataset(dataset_path)
    queries = data[:1000]

    compute_exact_neighbors(data, queries, k=10, num_workers=4)
