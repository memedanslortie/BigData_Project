import numpy as np
import h5py
import os

def load_dataset(name):
    path = os.path.join("data", f"{name}.hdf5")
    gt_path = os.path.join("data", "ground_truth.hdf5")

    with h5py.File(path, 'r') as f:
        xb = f['train'][:]
        xq = f['test'][:]

    with h5py.File(gt_path, 'r') as f:
        gt = f['neighbors'][:]
        # Charger les paramètres de normalisation s'ils existent
        if 'norm_mean' in f and 'norm_std' in f:
            mean = f['norm_mean'][:]
            std = f['norm_std'][:]
        else:
            # Sinon, calculer les paramètres à partir des données d'entraînement
            mean = np.mean(xb, axis=0)
            std = np.std(xb, axis=0)
            std[std < 1e-5] = 1.0

    if gt.shape[0] != xq.shape[0]:
        print(f"ground truth truncated: {gt.shape[0]} → {xq.shape[0]}")
        gt = gt[:xq.shape[0]]

    # Appliquer la même normalisation que dans compute_ground_truth.py
    print(f"Normalisation des données pour {name}...")
    print(f"  Forme des données - Train: {xb.shape}, Test: {xq.shape}")
    print(f"  Avant normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")
    
    xb = (xb - mean) / std
    xq = (xq - mean) / std
    
    print(f"  Après normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")

    return xb, xq, gt