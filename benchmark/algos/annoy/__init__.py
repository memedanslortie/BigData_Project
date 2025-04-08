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
        """
        Query the index with vectors in xq, returning k nearest neighbors for each.
        
        Args:
            xq: Query vectors
            k: Number of neighbors to return
            search_k: Number of nodes to inspect during search (higher = more accurate but slower)
        
        Returns:
            numpy.ndarray: Array of indices of nearest neighbors, shape (len(xq), k)
        """
        if search_k is None:
            search_k = self.search_k
        
        start = time.time()
        results = []
        
        for vector in xq:
            # Convert vector to float32 to ensure compatibility
            vector = np.array(vector, dtype=np.float32)
            # Important: Use search_k parameter for better recall
            indices = self.index.get_nns_by_vector(vector, k, search_k=search_k)
            results.append(indices)
        
        # Store the total search time as an attribute
        self.last_search_time = time.time() - start
        
        # Convert to numpy array
        return np.array(results)