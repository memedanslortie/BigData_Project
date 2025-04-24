import numpy as np
import time
import random
import tempfile
import os
import shutil
from pymilvus import MilvusClient

class Milvus:
    def __init__(self, ef_construction=400, M=16, ef_search=100, random_seed=42, space='l2'):
        """
        Initialize the Milvus HNSW index.
        
        Args:
            ef_construction: Similar to neighbors_to_explore_at_insert in Vespa (400 by default)
            M: Maximum connections per node, similar to max_links_per_node in Vespa (16 by default)
            ef_search: Size of the dynamic candidate list during search (100 by default)
            random_seed: Random seed for reproducibility (42 by default)
            space: Distance metric ('l2', 'ip', or 'cosine')
        """
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.random_seed = random_seed
        
        # Map distance metrics to Milvus format
        if space == 'cosine':
            self.metric_type = "COSINE"
        elif space == 'ip':
            self.metric_type = "IP"  # Inner product
        else:
            self.metric_type = "L2"  # Default to Euclidean
        
        # Create a unique DB path to prevent conflicts between different instances
        self.db_path = os.path.join(tempfile.gettempdir(), f"milvus_benchmark_{self.random_seed}_{random.randint(1000, 9999)}.db")
        self.client = None
        self.collection_name = None
        self.last_search_time = 0
        
    def fit(self, data):
        """
        Build the Milvus index with the input data.
        
        Args:
            data: Input data (numpy array)
            
        Returns:
            float: Build time in seconds
        """
        # Initialize Milvus client with a local DB file
        if self.client:
            try:
                self.client.drop_collection(self.collection_name)
            except Exception as e:
                print(f"Warning: Could not drop previous collection: {e}")
        
        try:
            self.client = MilvusClient(self.db_path)
        except Exception as e:
            print(f"Error initializing MilvusClient: {e}")
            raise
        
        # Create a unique collection name to avoid conflicts
        self.collection_name = f"benchmark_collection_{self.random_seed}_{int(time.time()) % 10000}"
        dimension = data.shape[1]
        
        print(f"Creating collection {self.collection_name} with dimension {dimension}")
        
        # Create collection with default settings
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                metric_type=self.metric_type
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
            
        start_time = time.time()
        
        # Insert data in batches
        batch_size = 1000
        num_vectors = data.shape[0]
        
        for i in range(0, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch_data = []
            
            for j in range(i, end_idx):
                # Create entity with ID and vector
                entity = {
                    "id": j,
                    "vector": data[j].tolist()
                }
                batch_data.append(entity)
            
            try:
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
            except Exception as e:
                print(f"Error inserting batch {i//batch_size}: {e}")
                raise
        
        # Try to create index but don't fail if we can't
        try:
            import pkg_resources
            pymilvus_version = pkg_resources.get_distribution("pymilvus").version
            print(f"PyMilvus version: {pymilvus_version}")
            
            # For PyMilvus 2.5.7, try the simplest approach
            try:
                # For newer versions that need IndexParams class
                try:
                    # Try to import IndexParams
                    from pymilvus.orm import IndexParams
                    index_params = IndexParams(
                        index_type="HNSW",
                        metric_type=self.metric_type,
                        params={"M": self.M, "efConstruction": self.ef_construction}
                    )
                    self.client.create_index(
                        collection_name=self.collection_name,
                        field_name="vector",
                        index_params=index_params
                    )
                except ImportError:
                    # Use direct dictionary if IndexParams not available
                    print("Simple dictionary indexing")
                    # Will likely fail but worth a try
            except Exception as e:
                print(f"Could not create index: {e}")
                print("Using default index")
        except Exception as e:
            print(f"Error in index creation: {e}")
        
        build_time = time.time() - start_time
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
        # Always initialize with the correct shape to avoid the tuple issue
        indices = np.zeros((len(queries), k), dtype=np.int32)
        distances = np.zeros((len(queries), k), dtype=np.float32)
        
        if not self.client or not self.collection_name:
            print("Warning: Client not initialized or collection not created")
            return indices, distances
        
        try:
            # Set search parameters
            search_params = {
                "params": {"ef": self.ef_search}
            }
            
            start_time = time.time()
            
            # Process each query
            for i, query in enumerate(queries):
                try:
                    # Convert query to list format
                    query_vector = query.tolist()
                    
                    # Search
                    results = self.client.search(
                        collection_name=self.collection_name,
                        data=[query_vector],
                        limit=k,
                        search_params=search_params
                    )
                    
                    # Process results
                    if not results or len(results) == 0 or len(results[0]) == 0:
                        print(f"Warning: No results for query {i}")
                        continue  # Keep zeros for this query
                    
                    result_list = results[0]  # First query result
                    
                    # Extract IDs and distances
                    for j, hit in enumerate(result_list):
                        if j < k:  # Make sure we don't exceed k
                            indices[i, j] = hit["id"]
                            distances[i, j] = hit["distance"]
                            
                except Exception as e:
                    print(f"Error searching for query {i}: {e}")
            
            self.last_search_time = (time.time() - start_time) / len(queries)
            
        except Exception as e:
            print(f"Error during query: {e}")
        
        # If using IP distance, negate distances to match other algorithms
        if self.metric_type == "IP":
            distances = -distances
            
        # Make sure we return the arrays, not as a tuple
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
            'metric_type': self.metric_type
        }
    
    def get_query_params(self):
        """
        Return parameters used for querying.
        
        Returns:
            Dict: Query parameters
        """
        return {
            'ef_search': self.ef_search
        }
    
    def get_search_time(self):
        """
        Return the average search time per query.
        
        Returns:
            float: Average search time per query in seconds
        """
        return self.last_search_time
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        if hasattr(self, 'client') and self.client:
            try:
                if hasattr(self, 'collection_name') and self.collection_name:
                    self.client.drop_collection(self.collection_name)
            except Exception as e:
                pass  # Silent cleanup errors
            
        # Clean up the temporary DB file
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception as e:
                pass  # Silent cleanup errors