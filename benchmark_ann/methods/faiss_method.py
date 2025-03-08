import faiss
import numpy as np

class FAISSIndexer:
    def __init__(self, dim, nlist=100, nprobe=10):
        """Initialise un index FAISS IVF-PQ"""
        self.dim = dim
        self.nlist = nlist  
        self.nprobe = nprobe  

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, 8, 8)
        self.trained = False

    def train(self, data):
        """Entraîne FAISS et ajoute les données"""
        if not self.trained:
            self.index.train(data)
            self.trained = True
        self.index.add(data)

    def search(self, queries, k):
        """Recherche des k plus proches voisins"""
        self.index.nprobe = self.nprobe  
        distances, indices = self.index.search(queries, k)
        return distances, indices

    def save_index(self, path="results/faiss_index.ivfpq"):
        """Sauvegarde l'index FAISS"""
        faiss.write_index(self.index, path)

    def load_index(self, path="results/faiss_index.ivfpq"):
        """Charge un index FAISS pré-enregistré"""
        self.index = faiss.read_index(path)
        self.trained = True
