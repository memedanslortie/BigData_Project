from annoy import AnnoyIndex
import time
import numpy as np

class Annoy:
    def __init__(self, dimension=None, metric='angular', **kwargs):
        self.index = None
        self.dimension = dimension
        self.metric = metric
        
        # Get additional parameters from kwargs
        self.n_trees = kwargs.get('n_trees', 10)
        self.search_k = kwargs.get('search_k', -1)  # Default to -1 (auto)
        
        # If dimension was not provided directly, try to get it from kwargs
        if self.dimension is None and 'dimension' in kwargs:
            self.dimension = kwargs['dimension']
    
    def fit(self, data, n_trees=None):
        # If dimension is still None, try to infer it from the first data point
        if self.dimension is None and len(data) > 0:
            self.dimension = len(data[0])
        
        if self.dimension is None:
            raise ValueError("Dimension must be provided either in __init__ or inferred from data")
            
        if n_trees is None:
            n_trees = self.n_trees
        
        self.index = AnnoyIndex(self.dimension, self.metric)
        
        # Convert data to float32 to ensure compatibility
        for i, vector in enumerate(data):
            self.index.add_item(i, np.array(vector, dtype=np.float32))
            
        self.index.build(n_trees)
        return self
        
    def query(self, xq, k, search_k=None):
        if search_k is None:
            search_k = self.search_k
        
        start = time.time()
        results = []
        
        for vector in xq:
            vector = np.array(vector, dtype=np.float32)
            indices = self.index.get_nns_by_vector(vector, k, search_k=search_k)
            results.append(indices)
        self.last_search_time = time.time() - start
        
        # Convert to numpy array
        return np.array(results)