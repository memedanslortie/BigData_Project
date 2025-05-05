import numpy as np
import time
import hnswlib
import multiprocessing

class HNSW:
    def __init__(self, M=16, ef_construction=200, ef=100, random_seed=42, space='l2', n_threads=None):
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.random_seed = random_seed
        self.space = self._convert_metric(space)
        self.index = None
        self.last_search_time = 0
        
        if n_threads is None:
            n_threads = multiprocessing.cpu_count()
        self.n_threads = n_threads
        print(f"HNSW (hnswlib) configuré pour utiliser {self.n_threads} threads")
    
    def _convert_metric(self, metric):
        """Convertit une métrique standard en métrique hnswlib"""
        metric_map = {
            'euclidean': 'l2',
            'l2': 'l2',
            'cosine': 'cosine',
            'angular': 'cosine',
            'ip': 'ip',
            'inner_product': 'ip',
            'dot': 'ip'
        }
        return metric_map.get(metric.lower(), 'l2')
    
    def fit(self, data):
        n_samples = data.shape[0]
        dim = data.shape[1]
    
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(
            max_elements=n_samples,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.random_seed
        )
        
        self.index.set_num_threads(self.n_threads)
        
        start = time.time()
        self.index.add_items(data)
        build_time = time.time() - start
        print(f"Index HNSW construit en {build_time:.2f}s avec {self.n_threads} threads")
        
        self.index.set_ef(self.ef)
        
        return self
    
    def query(self, xq, k):
        self.index.set_num_threads(self.n_threads)
        
        start = time.time()
        labels, _ = self.index.knn_query(xq, k=k)
        self.last_search_time = time.time() - start
        
        return labels

