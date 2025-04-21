import faiss
import numpy as np
import h5py
import os

DATASET = "fashion-mnist-784-euclidean.hdf5"
OUTPUT = "fashion-mnist-784-euclidean-ground_truth.hdf5"
K = 10

path = os.path.join("data", DATASET)
with h5py.File(path, 'r') as f:
    xb = f['train'][:]
    xq = f['test'][:]

# Appliquer la normalisation avant de calculer la vérité terrain
print("Application de la normalisation avant calcul de la vérité terrain...")
print(f"Forme des données - Train: {xb.shape}, Test: {xq.shape}")
print(f"Avant normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")

# 1. Standardisation (centrage-réduction)
mean = np.mean(xb, axis=0)
std = np.std(xb, axis=0)
std[std < 1e-5] = 1.0  # Éviter division par zéro

xb = (xb - mean) / std
xq = (xq - mean) / std

print(f"Après normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")

# Calcul de la vérité terrain avec FAISS
print("Calcul de la vérité terrain...")
index = faiss.IndexFlatL2(xb.shape[1])
index.add(xb)
D, I = index.search(xq, K)

# Sauvegarde des résultats
with h5py.File(os.path.join("data", OUTPUT), 'w') as f:
    f.create_dataset("neighbors", data=I)
    # Sauvegarder les paramètres de normalisation
    f.create_dataset("norm_mean", data=mean)
    f.create_dataset("norm_std", data=std)

print("Ground truth enregistré dans data/ground_truth.hdf5")