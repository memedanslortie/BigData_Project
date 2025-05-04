import numpy as np
import time
import nmslib

class NMSLIB_HNSW:
    def __init__(self, M=16, ef_construction=200, ef=100, random_seed=42, space='l2'):
        """
        Initialize the NMSLIB HNSW index.
        
        Args:
            M: Maximum number of connections per element (default 16)
            ef_construction: Size of the dynamic list during construction (default 200)
            ef: Size of the dynamic list during search (default 100)
            random_seed: Random seed for reproducibility (default 42)
            space: Distance metric ('l2', 'cosine', 'ip')
        """
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.random_seed = random_seed
        
        # Map distance metrics to NMSLIB format
        if space == 'cosine':
            self.space = 'cosinesimil'
        elif space == 'ip':
            self.space = 'negdotprod'  # Negative inner product for maximization
        else:
            self.space = 'l2'  # Default to Euclidean
        
        self.index = None
        self.last_search_time = 0
        
    def fit(self, data):
        """
        Build the NMSLIB HNSW index with the input data.
        
        Args:
            data: Input data (numpy array)
            
        Returns:
            float: Build time in seconds
        """
        # Initialize the index
        self.index = nmslib.init(method='hnsw', space=self.space, data_type=nmslib.DataType.DENSE_VECTOR)
        
        # Add data to the index
        start_time = time.time()
        
        # Convert data to float32 to ensure compatibility with NMSLIB
        data_float = data.astype(np.float32)
        self.index.addDataPointBatch(data_float)
        
        # Set index parameters
        index_params = {
            'M': self.M,
            'efConstruction': self.ef_construction,
            'post': 0,
            'skip_optimized_index': 1,  # Disable post-optimization to match other implementations
            'indexThreadQty': 4  # Use multiple threads for indexing
        }
        
        # Build the index
        print(f"Building NMSLIB HNSW index with M={self.M}, ef_construction={self.ef_construction}")
        self.index.createIndex(index_params, print_progress=True)
        
        build_time = time.time() - start_time
        
        # Set search parameters
        query_params = {
            'efSearch': self.ef
        }
        self.index.setQueryTimeParams(query_params)
        
        return build_time
    
    def query(self, queries, k):
        """
        Search for k nearest neighbors for each query.
        
        Args:
            queries: Query vectors (numpy array)
            k: Number of nearest neighbors to find
            
        Returns:
            indices: Indices of k nearest neighbors for each query
            distances: Corresponding distances
        """
        # Initialize arrays to store results
        num_queries = queries.shape[0]
        indices = np.zeros((num_queries, k), dtype=np.int32)
        distances = np.zeros((num_queries, k), dtype=np.float32)
        
        if self.index is None:
            print("Warning: Index not initialized")
            return indices, distances
        
        try:
            # Set search parameters (can be adjusted for each query if needed)
            query_params = {
                'efSearch': self.ef
            }
            self.index.setQueryTimeParams(query_params)
            
            # Convert queries to float32
            queries_float = queries.astype(np.float32)
            
            start_time = time.time()
            
            # Process each query
            for i in range(num_queries):
                try:
                    # Execute search
                    ids, dists = self.index.knnQuery(queries_float[i], k=k)
                    
                    # Store results
                    num_results = min(len(ids), k)
                    indices[i, :num_results] = ids[:num_results]
                    distances[i, :num_results] = dists[:num_results]
                    
                except Exception as e:
                    print(f"Error searching for query {i}: {e}")
            
            self.last_search_time = (time.time() - start_time) / num_queries
            
        except Exception as e:
            print(f"Error during query: {e}")
        
        # For inner product space, negate distances to match other algorithms
        if self.space == 'negdotprod':
            distances = -distances
            
        return indices, distances
    
    def get_build_params(self):
        """
        Return parameters used to build the index.
        
        Returns:
            Dict: Build parameters
        """
        return {
            'M': self.M,
            'ef_construction': self.ef_construction,
            'random_seed': self.random_seed,
            'space': self.space
        }
    
    def get_query_params(self):
        """
        Return parameters used for querying.
        
        Returns:
            Dict: Query parameters
        """
        return {
            'ef': self.ef
        }
    
    def get_search_time(self):
        """
        Return the average search time per query.
        
        Returns:
            float: Average search time per query in seconds
        """
        return self.last_search_time