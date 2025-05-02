import faiss
import numpy as np
import time
import os

class FaissIVFPQ:
    def __init__(self, nlist=100, m=8, nbits=8, nprobe=10, k_reorder=0, n_threads=None):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.k_reorder = k_reorder
        self.index = None
        self.xb = None
        self.adjusted_m = None
        
        # Définir le nombre de threads (utiliser tous les cœurs disponibles par défaut)
        if n_threads is None:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        self.n_threads = n_threads
        
        faiss.omp_set_num_threads(self.n_threads)

    def _find_adjusted_m(self, d, target_m):
        """
        Trouve une valeur de m qui divise d et qui est la plus proche possible de target_m.
        """
        divisors = [i for i in range(1, d + 1) if d % i == 0]
        return min(divisors, key=lambda x: abs(x - target_m))

    def fit(self, xb):
        d = xb.shape[1]
        
        # Ajuster m pour qu'il soit un diviseur de d
        self.adjusted_m = self._find_adjusted_m(d, self.m)
        if self.adjusted_m != self.m:
            print(f"Ajustement automatique: m={self.m} -> m={self.adjusted_m} (pour être diviseur de d={d})")
        
        print(f"Utilisation de {self.n_threads} threads pour l'entraînement")
        
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.adjusted_m, self.nbits)
        
        # Parallélisation de l'entraînement
        start = time.time()
        self.index.train(xb)
        self.index.add(xb)
        train_time = time.time() - start
        print(f"Index construit en {train_time:.2f}s avec {self.n_threads} threads")
        
        self.index.nprobe = self.nprobe
        self.xb = xb

    def query(self, xq, k):
        start = time.time()
        
        D, I = self.index.search(xq, max(k, self.k_reorder))  # chercher + que k si reorder
        
        if self.k_reorder > 0:
            from concurrent.futures import ThreadPoolExecutor
            
            def reorder_single(i):
                candidates = self.xb[I[i]]
                dists = np.linalg.norm(candidates - xq[i], axis=1)
                topk = np.argsort(dists)[:k]
                return I[i][topk]
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                reordered_I = list(executor.map(reorder_single, range(xq.shape[0])))
            
            I = np.array(reordered_I)
            
        end = time.time()
        self.last_search_time = end - start
        return I