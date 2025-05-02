import ngtpy
import numpy as np
import time
import os
import shutil

class QSGNGT:
    """
    Implementation of QSG-NGT (Quantized Sparse Graph with Neighborhood Graph and Tree)
    Based on NGT library with improved memory efficiency
    """
    def __init__(self, edge_size=10, creation_edge_size=40, search_edge_size=40, 
                 epsilon=0.1, dimension=None, quantization_level=None, metric='L2'):
        """
        Initialize QSG-NGT index
        
        Parameters:
        edge_size (int): Number of edges for each node in the graph
        creation_edge_size (int): Number of edges during graph creation
        search_edge_size (int): Number of edges to explore during search
        epsilon (float): Epsilon for search expansion (higher = more accurate but slower)
        dimension (int): Vector dimension
        metric (str): Distance metric ('L2' or 'Cosine')
        """
        self.edge_size = edge_size
        self.creation_edge_size = creation_edge_size
        self.search_edge_size = search_edge_size
        self.epsilon = epsilon
        self.dimension = dimension
        self.metric = metric
        
        # Index path - will be created in a temp directory
        self.index_path = "qsgngt_index"
        
        # Initialize last_search_time
        self.last_search_time = 0
    
    def fit(self, xb):
        """
        Build QSG-NGT index with the training data
        
        Parameters:
        xb (numpy.ndarray): Training vectors, shape (n, d) where d is the dimension
        """
        # Clean up any existing index
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        
        # If dimension was not provided, infer from data
        if self.dimension is None:
            self.dimension = xb.shape[1]
        
        # Create index with specified parameters
        ngtpy.create(
            path=self.index_path,
            dimension=self.dimension,
            edge_size_for_creation=self.creation_edge_size,
            edge_size_for_search=self.search_edge_size,
            distance_type=self.metric.upper(), 
            object_type='Float'
        )
        
        # Open the index
        self.index = ngtpy.Index(self.index_path)
        
        # Insert all vectors with batch insertion
        for i, vector in enumerate(xb):
            self.index.insert(vector.astype(np.float32)) 
        
        # Build the index
        self.index.build_index()
        
        # Set the number of edges
        if hasattr(self.index, 'set_number_of_edges'):
            self.index.set_number_of_edges(self.edge_size)
        
        return self
    
    def query(self, xq, k):
        """
        Query the index with test vectors
        
        Parameters:
        xq (numpy.ndarray): Query vectors, shape (nq, d)
        k (int): Number of nearest neighbors to retrieve
        
        Returns:
        numpy.ndarray: Indices of nearest neighbors, shape (nq, k)
        """
        start = time.time()
        
        results = []
        for query in xq:
            # Ensure query is float32
            query = query.astype(np.float32)
            
            # Search with epsilon-based exploration
            neighbors = self.index.search(
                query, 
                size=k, 
                epsilon=self.epsilon,
                edge_size=self.search_edge_size
            )
            
            # Extract indices (first element in each tuple)
            indices = [int(neighbor[0]) for neighbor in neighbors]
            
            # Add padding if needed
            if len(indices) < k:
                indices += [-1] * (k - len(indices))
            
            results.append(indices[:k])  # Ensure we have exactly k results
        
        self.last_search_time = time.time() - start
        
        return np.array(results, dtype=np.int32)
    
    def __del__(self):
        """Clean up resources when object is deleted"""
        if hasattr(self, 'index') and self.index is not None:
            self.index.close()
        
        # Remove index directory if it exists
        if os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
            except:
                pass