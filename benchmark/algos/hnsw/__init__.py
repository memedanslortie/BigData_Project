import numpy as np
import time
import hnswlib

class HNSW:
    def __init__(self, M=16, ef_construction=200, ef=100, random_seed=42, space='l2'):
        """
        Initialise l'index HNSW.
        
        Args:
            M: Nombre maximum de connexions par point (16 par défaut)
            ef_construction: Taille du pool dynamique utilisé pendant la construction (200 par défaut)
            ef: Facteur d'exploration pendant la recherche (100 par défaut)
            random_seed: Graine aléatoire pour la reproductibilité (42 par défaut)
            space: Type de distance ('l2' ou 'cosine' ou 'ip')
        """
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.random_seed = random_seed
        self.space = space
        self.index = None
        self.last_search_time = 0
    
    def fit(self, data):
        """
        Construit l'index HNSW avec les données d'entrée.
        
        Args:
            data: Données d'entrée pour la construction de l'index (numpy array)
        """
        n_samples = data.shape[0]
        dim = data.shape[1]
        
        # Initialiser l'index
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(
            max_elements=n_samples,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.random_seed
        )
        
        # Ajouter les éléments à l'index
        self.index.add_items(data)
        
        # Configurer le facteur d'exploration pour la recherche
        self.index.set_ef(self.ef)
        
        return self
    
    def query(self, xq, k):
        """
        Effectue des requêtes de k plus proches voisins.
        
        Args:
            xq: Vecteurs de requête
            k: Nombre de voisins à retourner
            
        Returns:
            Indices des k plus proches voisins pour chaque vecteur de requête
        """
        start = time.time()
        
        # Recherche des plus proches voisins
        labels, _ = self.index.knn_query(xq, k=k)
        
        self.last_search_time = time.time() - start
        
        return labels