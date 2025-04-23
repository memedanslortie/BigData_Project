import numpy as np
import time
from annoy import AnnoyIndex

class Annoy:
    def __init__(self, n_trees=10, search_k=-1):
        """
        Initialisation de l'algorithme Annoy.
        
        Paramètres:
        n_trees (int): Nombre d'arbres. Plus de valeur = plus précis mais plus lent à construire
        search_k (int): Nombre de nœuds à inspecter lors de la recherche (-1 = n_trees * n * 2)
        """
        self.n_trees = n_trees
        self.search_k = search_k
        self.index = None
        self.dim = None
        self.last_search_time = 0
    
    def fit(self, X):
        """Construire l'index Annoy à partir des données d'apprentissage."""
        n, self.dim = X.shape
        
        # Initialisation de l'index
        self.index = AnnoyIndex(self.dim, 'euclidean')
        
        # Ajout des points au index
        for i, x in enumerate(X):
            self.index.add_item(i, x.astype('float32'))
        
        # Construction de l'index
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
            # Recherche des k plus proches voisins
            # search_k contrôle le compromis précision/vitesse à la recherche
            search_k = self.search_k
            if search_k == -1:
                search_k = self.n_trees * X.shape[0]
                
            neighbors = self.index.get_nns_by_vector(
                x.astype('float32'), k, search_k=search_k, include_distances=False)
            I[i, :len(neighbors)] = neighbors
        
        self.last_search_time = time.time() - start
        
        return I
    
    def save(self, filename):
        """Sauvegarder l'index dans un fichier."""
        if self.index:
            self.index.save(filename)
    
    def load(self, filename):
        """Charger l'index depuis un fichier."""
        if not self.index and self.dim:
            self.index = AnnoyIndex(self.dim, 'euclidean')
            self.index.load(filename)