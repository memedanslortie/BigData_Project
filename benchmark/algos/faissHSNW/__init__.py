import faiss
import time
import numpy as np

class FaissHNSW:

    def __init__(self, M=16, efConstruction=200, efSearch=100):


        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.index = None
        self.last_search_time = 0
    
    def fit(self, xb):

        d = xb.shape[1]
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(d, self.M)
        
        # Set construction-time parameters
        self.index.hnsw.efConstruction = self.efConstruction
        
        # Add vectors to the index
        self.index.add(xb)
        
        # Set search-time parameters
        self.index.hnsw.efSearch = self.efSearch
        
        return self
    
    def query(self, xq, k):

        start = time.time()
        _, I = self.index.search(xq, k)
        self.last_search_time = time.time() - start
        return I