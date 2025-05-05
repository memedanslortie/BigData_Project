import numpy as np
import time
from annoy import AnnoyIndex

class Annoy:
    def __init__(self, n_trees=10, search_k=-1, metric='euclidean'):
        """
        Initialisation de l'algorithme Annoy.
        
        Paramètres:
        n_trees (int): Nombre d'arbres. Plus de valeur = plus précis mais plus lent à construire
        search_k (int): Nombre de nœuds à inspecter lors de la recherche (-1 = n_trees * n * 2)
        metric (str): Métrique de distance ('euclidean', 'angular', 'manhattan', 'hamming', etc.)
        """
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = self._convert_metric(metric)
        self.index = None
        self.dim = None
        self.last_search_time = 0
    
    def _convert_metric(self, metric):
        """Convertit une métrique standard en métrique Annoy"""
        metric_map = {
            'euclidean': 'euclidean',
            'l2': 'euclidean',
            'cosine': 'angular',
            'angular': 'angular',
            'manhattan': 'manhattan',
            'l1': 'manhattan',
            'hamming': 'hamming',
            'jaccard': 'hamming', 
            'dot': 'dot'
        }
        return metric_map.get(metric.lower(), 'euclidean')
    
    def fit(self, X):
        """Construire l'index Annoy à partir des données d'apprentissage."""
        n, self.dim = X.shape
        
        self.index = AnnoyIndex(self.dim, self.metric)
        
        for i, x in enumerate(X):
            self.index.add_item(i, x.astype('float32'))
        
        start = time.time()
        self.index.build(self.n_trees)
        self.build_time = time.time() - start
        
        return self
    
    def query(self, X, k):
        """
        Rechercher les k plus proches voisins pour chaque point de X.
        
        Paramètres:
        X: données de requête de forme (n_queries, dim)
        k: nombre de voisins à rechercher
        
        Retour:
        I: indices des k plus proches voisins pour chaque requête
        """
        n_queries = X.shape[0]
        I = np.zeros((n_queries, k), dtype=np.int32)
        
        start = time.time()
        for i, x in enumerate(X):
            search_k = self.search_k
            if search_k == -1:
                search_k = self.n_trees * X.shape[0]
                
            neighbors = self.index.get_nns_by_vector(
                x.astype('float32'), k, search_k=search_k, include_distances=False)
            I[i, :len(neighbors)] = neighbors
        
        self.last_search_time = time.time() - start
        
        return I

