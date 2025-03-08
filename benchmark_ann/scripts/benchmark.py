import time
import numpy as np
import argparse
import h5py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_dataset import load_dataset, load_ground_truth
from methods import FAISSIndexer  

def compute_recall(true_neighbors, approx_neighbors, k=10):
    """Calcule le recall@k"""
    recall_values = [
        len(set(true[:k]) & set(approx[:k])) / k
        for true, approx in zip(true_neighbors, approx_neighbors)
    ]
    return np.mean(recall_values)

def load_queries(queries_path, num_queries):
    """Charge les requêtes pré-sauvegardées pour assurer la cohérence du benchmark"""
    with h5py.File(queries_path, "r") as f:
        return f["queries"][:num_queries]

def benchmark_faiss(dataset_path, queries_path, k=10, num_queries=1000):
    """Évalue FAISS (IVF-PQ)"""
    data = load_dataset(dataset_path)
    queries = load_queries(queries_path, num_queries)  # Chargement des requêtes sauvegardées
    true_neighbors = load_ground_truth("results/ground_truth.hdf5")[:num_queries]

    print("\nTesting FAISS (IVF-PQ)...")
    indexer = FAISSIndexer(dim=data.shape[1])

    try:
        indexer.load_index("results/faiss_index.ivfpq")
        print("Index FAISS chargé depuis le fichier.")
    except:
        print("Aucun index sauvegardé, entraînement en cours...")
        indexer.train(data)
        indexer.save_index("results/faiss_index.ivfpq")

    start_time = time.time()
    _, approx_neighbors = indexer.search(queries, k)
    end_time = time.time()

    recall = compute_recall(true_neighbors, approx_neighbors, k)
    search_time = (end_time - start_time) / num_queries

    print(f"Recall@{k}: {recall:.4f}, Time: {search_time:.6f} sec/query")

    return {"recall": recall, "time": search_time}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion-mnist-784-euclidean.hdf5",
                        help="Nom du fichier HDF5 à utiliser")
    args = parser.parse_args()

    dataset_path = f"datasets/{args.dataset}"
    queries_path = "results/queriesGT.hdf5"  # Chemin vers les requêtes sauvegardées

    benchmark_faiss(dataset_path, queries_path, k=10, num_queries=1000)

# Exemple d'exécution : 
# python scripts/benchmark.py --dataset=fashion-mnist-784-euclidean.hdf5
"""
If each method has different parameters, we don't know if one method is better or if its hyperparameters are just better chosen?
"""

""" 

def compute_recall(true_neighbors, approx_neighbors, k=10):
    recall_values = [
        len(set(true[:k]) & set(approx[:k])) / k
        for true, approx in zip(true_neighbors, approx_neighbors)
    ]
    return np.mean(recall_values)

def load_queries(queries_path, num_queries):
    with h5py.File(queries_path, "r") as f:
        return f["queries"][:num_queries]

def benchmark_faiss_exact(dataset_path, queries_path, k=10, num_queries=1000):
    data = load_dataset(dataset_path)
    queries = load_queries(queries_path, num_queries)  # Chargement des requêtes sauvegardées
    true_neighbors = load_ground_truth("results/ground_truth.hdf5")[:num_queries]

    print("\nTesting FAISS (Exact Search)...")
    indexer = FAISSExactIndexer(dim=data.shape[1])  # Utilisation de FAISSExactIndexer
    indexer.train(data)  # Pas d'entraînement nécessaire pour IndexFlatL2

    start_time = time.time()
    _, exact_neighbors = indexer.search(queries, k)
    end_time = time.time()

    recall = compute_recall(true_neighbors, exact_neighbors, k)
    search_time = (end_time - start_time) / num_queries

    print(f"Recall@{k}: {recall:.4f}, Time: {search_time:.6f} sec/query")

    return {"recall": recall, "time": search_time}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion-mnist-784-euclidean.hdf5",
                        help="Nom du fichier HDF5 à utiliser")
    args = parser.parse_args()

    dataset_path = f"datasets/{args.dataset}"
    queries_path = "results/queriesGT.hdf5"  # Chemin vers les requêtes sauvegardées

    benchmark_faiss_exact(dataset_path, queries_path, k=10, num_queries=1000)

# Exemple d'exécution : 
# python scripts/benchmark_exact.py --dataset=fashion-mnist-784-euclidean.hdf5



"""