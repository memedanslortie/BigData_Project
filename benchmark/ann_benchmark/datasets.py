import numpy as np
import h5py
import os

def load_dataset(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    
    path = os.path.join(data_dir, f"{name}.hdf5")
    gt_path = os.path.join(data_dir, f"{name}_ground_truth.hdf5")
    
    print(f"Chargement des données depuis {path}")
    with h5py.File(path, 'r') as f:
        xb = f['train'][:]
        xq = f['test'][:]

    print(f"Chargement du ground truth depuis {gt_path}")
    with h5py.File(gt_path, 'r') as f:
        gt = f['neighbors'][:]
        metric = f.attrs.get('metric', 'l2').lower()
        normalized = f.attrs.get('normalized', False)

        if 'norm_mean' in f and 'norm_std' in f:
            mean = f['norm_mean'][:]
            std = f['norm_std'][:]
            print("Paramètres de normalisation chargés depuis le ground truth")
        else:
            mean = np.mean(xb, axis=0)
            std = np.std(xb, axis=0)
            std[std < 1e-5] = 1.0
            print("Paramètres de normalisation calculés à partir des données")

    if gt.shape[0] != xq.shape[0]:
        print(f"Ground truth tronqué: {gt.shape[0]} → {xq.shape[0]}")
        gt = gt[:xq.shape[0]]

    print(f"Forme des données - Train: {xb.shape}, Test: {xq.shape}")

    # Appliquer la normalisation uniquement si la distance n'est pas cosine/angular
    if metric in ['cosine', 'angular']:
        print(f"Pas de standardisation : les données sont supposées déjà L2-normalisées (métrique = {metric})")
    else:
        print(f"Normalisation des données pour {name} (métrique = {metric})...")
        print(f"  Avant normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")
        xb = (xb - mean) / std
        xq = (xq - mean) / std
        print(f"  Après normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")

    return xb.astype(np.float32), xq.astype(np.float32), gt
