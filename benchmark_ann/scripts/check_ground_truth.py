import h5py
import numpy as np

def check_ground_truth(file_path):
    """Vérifie le fichier HDF5 contenant les vrais voisins et affiche des informations utiles."""
    with h5py.File(file_path, "r") as f:
        print("\n[INFO] Vérification du fichier Ground Truth...\n")

        # Afficher les clés présentes
        keys = list(f.keys())
        print(f"[INFO] Clés disponibles dans {file_path}: {keys}")

        if "true_neighbors" not in keys:
            print("[ERREUR] Clé 'true_neighbors' absente dans le fichier HDF5.")
            return
        
        # Charger les vrais voisins
        true_neighbors = f["true_neighbors"]
        print(f"[INFO] Shape des vrais voisins: {true_neighbors.shape}")

        # Afficher quelques exemples
        num_samples = min(5, true_neighbors.shape[0])
        print("\n[INFO] Exemples de vrais voisins:")
        for i in range(num_samples):
            print(f"  Requête {i}: {true_neighbors[i]}")

        # Vérifier les indices pour s'assurer qu'ils sont dans la bonne plage
        max_index = np.max(true_neighbors)
        min_index = np.min(true_neighbors)
        print(f"\n[INFO] Indices des voisins: Min = {min_index}, Max = {max_index}")

if __name__ == "__main__":
    file_path = "/Volumes/SSD/M1VMI/S2/big_data/benchmark_ann/results/ground_truth.hdf5"
    check_ground_truth(file_path)
