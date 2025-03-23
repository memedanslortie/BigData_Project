import numpy as np
import time
from voyager import Index, Space

class VoyagerANN:
    def __init__(self, M=12, ef_construction=200, query_ef=100, random_seed=1):
        self.M = M
        self.ef_construction = ef_construction
        self.query_ef = query_ef
        self.random_seed = random_seed
        self.index = None
        self.last_search_time = 0

    def fit(self, xb):
        dim = xb.shape[1]
        self.index = Index(
            space=Space.Euclidean,
            num_dimensions=dim,
            M=self.M,
            ef_construction=self.ef_construction,
            random_seed=self.random_seed,
            max_elements=xb.shape[0]  
        )

        self.index.add_items(xb.astype(np.float32)) 

    def query(self, xq, k):
        start = time.time()
        ids, _ = self.index.query(xq.astype(np.float32), k=k, query_ef=self.query_ef)
        end = time.time()

        self.last_search_time = end - start
        return ids
