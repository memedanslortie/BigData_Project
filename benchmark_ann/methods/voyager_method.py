import numpy as np
from voyager import Index, Space

class VoyagerIndexer:
    def __init__(self, dim):
        """Initialise un index Voyager"""
        self.dim = dim
        self.index = Index(Space.Euclidean, num_dimensions=dim)

    def add_items(self, data):
        """Ajoute des données à l'index"""
        self.index.add_items(data)

    def search(self, query, k):
        """Recherche des k plus proches voisins"""
        neighbors, distances = self.index.query(query, k)
        return distances, neighbors

    def save_index(self, path="results/voyager_index.voy"):
        """Sauvegarde l'index Voyager"""
        self.index.save(path)

    def load_index(self, path="results/voyager_index.voy"):
        """Charge un index Voyager pré-enregistré"""
        self.index = Index.load(path)
