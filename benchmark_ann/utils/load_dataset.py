import h5py
import numpy as np
from sklearn.preprocessing import normalize

def load_dataset(path, normalize_data=True):
    """Charge un dataset HDF5 et normalise les données si nécessaire."""
    with h5py.File(path, "r") as f:
        print(f"Clés disponibles : {list(f.keys())}")
        dataset = f["train"][:]  # Modifier selon le dataset
        if normalize_data:
            dataset = normalize(dataset, axis=1)
    return dataset

def load_ground_truth(path):
    """Charge le fichier des vrais voisins pré-calculés."""
    with h5py.File(path, "r") as f:
        return f["true_neighbors"][:]
