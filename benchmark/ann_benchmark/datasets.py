import numpy as np
import h5py
import os

def load_dataset(name):
    # Utilisation de chemins absolus pour garantir l'accès aux fichiers
    # Utilisation de chemins relatifs pour plus de portabilité
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Chemin vers le jeu de données
    path = os.path.join(data_dir, f"{name}.hdf5")
    
    # Chemin vers le ground truth spécifique à ce dataset
    gt_path = os.path.join(data_dir, f"{name}_ground_truth.hdf5")
    
    print(f"Chargement des données depuis {path}")
    with h5py.File(path, 'r') as f:
        xb = f['train'][:]
        xq = f['test'][:]

    print(f"Chargement du ground truth depuis {gt_path}")
    with h5py.File(gt_path, 'r') as f:
        gt = f['neighbors'][:]
        
        # Charger les paramètres de normalisation s'ils existent
        if 'norm_mean' in f and 'norm_std' in f:
            mean = f['norm_mean'][:]
            std = f['norm_std'][:]
            print("Paramètres de normalisation chargés depuis le ground truth")
        else:
            # Sinon, calculer les paramètres à partir des données d'entraînement
            mean = np.mean(xb, axis=0)
            std = np.std(xb, axis=0)
            std[std < 1e-5] = 1.0
            print("Paramètres de normalisation calculés à partir des données")

    if gt.shape[0] != xq.shape[0]:
        print(f"Ground truth tronqué: {gt.shape[0]} → {xq.shape[0]}")
        gt = gt[:xq.shape[0]]

    # Appliquer la même normalisation que dans compute_ground_truth.py
    print(f"Normalisation des données pour {name}...")
    print(f"  Forme des données - Train: {xb.shape}, Test: {xq.shape}")
    print(f"  Avant normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")
    
    xb = (xb - mean) / std
    xq = (xq - mean) / std
    
    print(f"  Après normalisation - Min: {xb.min():.4f}, Max: {xb.max():.4f}, Moyenne: {xb.mean():.4f}")

    return xb, xq, gt