import time
import numpy as np
import h5py
import multiprocessing
from tqdm import tqdm
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.load_dataset import load_dataset

def compute_exact_neighbors_chunk(start, end, data, queries, k):
    """Calcule les vrais voisins exacts pour un sous-ensemble de requêtes."""
    print(f"[Processus] Calcul des voisins exacts pour les requêtes {start} à {end}...", flush=True)
    
    true_neighbors_chunk = np.zeros((end - start, k), dtype=np.int32)
    
    for i, query in enumerate(tqdm(queries[start:end], desc=f"Process {start}-{end}")):
        distances = np.linalg.norm(data - query, axis=1)
        true_neighbors_chunk[i] = np.argsort(distances)[:k]  
    
    return start, true_neighbors_chunk

def compute_exact_neighbors(data, queries, k=10, num_workers=4, save_path="results/ground_truth.hdf5"):
    """Calcule et sauvegarde les vrais voisins en parallèle."""
    print("\n[INFO] Début du calcul des vrais voisins exacts...", flush=True)
    start_time = time.time()

    num_queries = queries.shape[0]
    chunk_size = num_queries // num_workers
    
    pool = multiprocessing.Pool(num_workers)
    results = []

    for i in range(num_workers):
        start = i * chunk_size
        end = num_queries if i == num_workers - 1 else (i + 1) * chunk_size
        print(f"[INFO] Démarrage du worker {i} pour les requêtes {start} à {end}...", flush=True)
        results.append(pool.apply_async(compute_exact_neighbors_chunk, (start, end, data, queries, k)))

    pool.close()
    pool.join()

    print("\n[INFO] Fusion des résultats des différents workers...", flush=True)
    true_neighbors = np.zeros((num_queries, k), dtype=np.int32)
    for res in results:
        start, chunk = res.get()
        true_neighbors[start:start+chunk.shape[0]] = chunk

    print(f"[INFO] Sauvegarde des résultats dans {save_path}...", flush=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("true_neighbors", data=true_neighbors)

    end_time = time.time()
    print(f"[INFO] Calcul terminé en {end_time - start_time:.2f} secondes.", flush=True)

if __name__ == "__main__":
    dataset_path = "datasets/fashion-mnist-784-euclidean.hdf5"

    print("[INFO] Chargement du dataset...", flush=True)
    data = load_dataset(dataset_path)
    print(f"[INFO] Dataset chargé avec {data.shape[0]} éléments.", flush=True)

    print("[INFO] Sélection des 1000 premières requêtes...", flush=True)
    queries = data[:1000]
    
    print("[INFO] Sauvegarde des requêtes dans results/queriesGT.hdf5...", flush=True)
    with h5py.File("results/queriesGT.hdf5", "w") as f:
        f.create_dataset("queries", data=queries)

    compute_exact_neighbors(data, queries, k=10, num_workers=4)
