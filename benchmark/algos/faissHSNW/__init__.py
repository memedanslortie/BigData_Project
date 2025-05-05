import faiss
import time
import numpy as np
import os

class FaissHNSW:
    def __init__(self, M=16, efConstruction=200, efSearch=100, n_threads=None, metric='l2'):
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.metric = metric.lower()
        self.index = None
        self.last_search_time = 0
        
        # Définir le nombre de threads
        if n_threads is None:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        self.n_threads = n_threads
        
        faiss.omp_set_num_threads(self.n_threads)
        print(f"FAISS HNSW configuré pour utiliser {self.n_threads} threads")
    
    def fit(self, xb):
        d = xb.shape[1]
        
        if self.metric == 'l2' or self.metric == 'euclidean':
            self.index = faiss.IndexHNSWFlat(d, self.M)
        elif self.metric == 'cosine' or self.metric == 'angular':
            self.index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
            xb = xb.copy()
            faiss.normalize_L2(xb)
            self.normalized = True
        elif self.metric == 'inner_product' or self.metric == 'dot':
            self.index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
            self.normalized = False
        else:
            raise ValueError(f"Métrique non supportée: {self.metric}")
        
        self.index.hnsw.efConstruction = self.efConstruction
        
        start = time.time()
        self.index.add(xb)
        build_time = time.time() - start
        print(f"Index HNSW construit en {build_time:.2f}s avec {self.n_threads} threads")
        
        self.index.hnsw.efSearch = self.efSearch
        
        return self
    
    def query(self, xq, k):
        start = time.time()
        
        if hasattr(self, 'normalized') and self.normalized:
            xq = xq.copy()
            faiss.normalize_L2(xq)
            
        _, I = self.index.search(xq, k)
        self.last_search_time = time.time() - start
        return I

