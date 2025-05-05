import faiss
import numpy as np
import h5py
import os
import argparse

def detect_metric_from_name(dataset_name):
    """Détermine la métrique à utiliser en fonction du nom du dataset"""
    dataset_name = dataset_name.lower()
    
    if 'lastfm' in dataset_name:
        return 'cosine' 
    
    if any(term in dataset_name for term in ['angular', 'cosine']):
        return 'cosine'
    elif any(term in dataset_name for term in ['dot', 'ip', 'inner']):
        return 'ip'
    elif any(term in dataset_name for term in ['jaccard', 'hamming']):
        return 'jaccard'
    else:
        return 'l2' 

def calculate_ground_truth(dataset_path, output_path, k=100, metric=None):
    """
    Calcule la vérité terrain pour un dataset donné en utilisant la métrique spécifiée.
    
    Args:
        dataset_path: Chemin vers le fichier HDF5 du dataset
        output_path: Chemin pour sauvegarder la vérité terrain
        k: Nombre de plus proches voisins à calculer
        metric: Métrique à utiliser ('l2', 'ip', 'cosine', 'jaccard'). Si None, détection automatique.
    """
    with h5py.File(dataset_path, 'r') as f:
        xb = f['train'][:]
        xq = f['test'][:]
        
    print(f"Forme des données - Train: {xb.shape}, Test: {xq.shape}")
    
    if metric is None:
        dataset_name = os.path.basename(dataset_path)
        metric = detect_metric_from_name(dataset_name)
    
    print(f"Calcul de la vérité terrain avec métrique: {metric}")
    
    if metric == 'cosine':
        print("Application de la normalisation L2 pour distance cosinus/angulaire...")
        xb_normalized = xb.copy().astype(np.float32)
        xq_normalized = xq.copy().astype(np.float32)
        
        # Normalisation L2
        faiss.normalize_L2(xb_normalized)
        faiss.normalize_L2(xq_normalized)
        
        # Pour cosine, on utilise le produit scalaire sur données normalisées
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb_normalized)
        D, I = index.search(xq_normalized, k)
        
    elif metric == 'ip':
        print("Utilisation du produit scalaire direct...")
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        D, I = index.search(xq, k)
        
    elif metric == 'jaccard':
        print("Conversion pour distance de Jaccard (approximation Hamming)...")
        if not np.all(np.isin(xb, [0, 1])):
            print("Warning: Les données ne semblent pas être binaires, conversion forcée.")
            xb = (xb > xb.mean()).astype(np.float32)
            xq = (xq > xq.mean()).astype(np.float32)
            
        I = np.zeros((xq.shape[0], k), dtype=int)
        for i in range(xq.shape[0]):
            # Calculer les scores de Jaccard (intersection / union)
            intersection = np.minimum(xq[i], xb).sum(axis=1)
            union = np.maximum(xq[i], xb).sum(axis=1)
            scores = intersection / np.maximum(union, 1e-10) 
            I[i] = np.argsort(-scores)[:k] 
            
        D = None 
            
    else: 
        print("Utilisation de la distance euclidienne (L2)...")
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)
        D, I = index.search(xq, k)
    
    # Sauvegarde des résultats
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("neighbors", data=I)
        f.attrs['metric'] = metric
        
        if metric == 'cosine':
            f.attrs['normalized'] = True
    
    print(f"Ground truth enregistré dans {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcule la vérité terrain pour un dataset ANN.')
    parser.add_argument('--dataset', type=str, required=True, help='Nom du fichier HDF5 du dataset')
    parser.add_argument('--metric', type=str, choices=['l2', 'ip', 'cosine', 'jaccard'], 
                        help='Métrique à utiliser (détection automatique si non spécifiée)')
    parser.add_argument('--k', type=int, default=100, help='Nombre de plus proches voisins à calculer')
    args = parser.parse_args()
    
    dataset_path = os.path.join("data", args.dataset)
    dataset_basename = os.path.splitext(args.dataset)[0]
    metric_suffix = args.metric if args.metric else detect_metric_from_name(args.dataset)
    output_path = os.path.join("data", f"{dataset_basename}_{metric_suffix}_ground_truth.hdf5")
    
    calculate_ground_truth(dataset_path, output_path, args.k, args.metric)