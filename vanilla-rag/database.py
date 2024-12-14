import os
import pickle
import heapq
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

setup_logging()


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: Optional[Dict] = None

class VectorDB:
    def __init__(
        self, 
        dimension: int, 
        index_size: int = 100,
        db_location: str = "db.pkl",
    ):
        """
        Initialize the vector database.
        
        Args:
            dimension: The dimension of vectors to be stored
            index_size: Number of clusters for approximate search
        """
        self.dimension = dimension
        self.index_size = index_size
        self.db_location = db_location
        self.vectors: List[VectorRecord] = []
        self.ids = set()
        self._is_indexed = False
        self.centroids = None
        self.centroid_map = {}  # Maps centroid index to vector indices
        self._load_db()

    def _load_db(self) -> None:
        """Attempt to load the database from disk."""
        if os.path.exists(self.db_location):
            try:
                with open(self.db_location, 'rb') as f:
                    saved_state = pickle.load(f)
                    
                # Verify the loaded data has the correct dimension
                if saved_state['dimension'] != self.dimension:
                    raise ValueError(
                        f"Loaded database dimension ({saved_state['dimension']}) "
                        f"doesn't match initialized dimension ({self.dimension})"
                    )
                    
                self.vectors = saved_state['vectors']
                self.ids = set([x.id for x in self.vectors])
                self._is_indexed = saved_state['indexed']
                self.centroids = saved_state['centroids']
                self.centroid_map = saved_state['centroid_map']
                logging.info(f"Successfully loaded database from {self.db_location}")
            except Exception as e:
                logging.error(f"Error loading database: {str(e)}")
                # Initialize empty state if load fails
                self.vectors = []
                self._is_indexed = False
                self.centroids = None
                self.centroid_map = {}

    def store(self) -> None:
        """Store the current database state to disk."""
        try:
            state = {
                'dimension': self.dimension,
                'vectors': self.vectors,
                'indexed': self._is_indexed,
                'centroids': self.centroids,
                'centroid_map': self.centroid_map
            }
            
            with open(self.db_location, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"Successfully stored database to {self.db_location}")
        except Exception as e:
            logging.error(f"Error storing database: {str(e)}")
            raise

    def add(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add a vector to the database."""
        if id in self.ids:
            raise ValueError(f"Document id must be unique.")

        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")
            
        vector = np.array(vector, dtype=np.float32)
        record = VectorRecord(id=id, vector=vector, metadata=metadata)
        self.vectors.append(record)
        self.ids.add(id)
        self._is_indexed = False
        
    def build_index(self) -> None:
        """Build an approximate nearest neighbors index using k-means clustering."""
        if len(self.vectors) < self.index_size:
            return
            
        # Extract numpy array of all vectors
        vectors = np.array([v.vector for v in self.vectors])
        
        # Perform k-means clustering
        centroids, labels = self._kmeans(vectors, self.index_size)
        
        # Build centroid mapping
        self.centroids = centroids
        self.centroid_map = {}
        for i, label in enumerate(labels):
            if label not in self.centroid_map:
                self.centroid_map[label] = []
            self.centroid_map[label].append(i)
            
        self._is_indexed = True

    def _kmeans(self, vectors: np.ndarray, k: int, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering implementation."""
        # Randomly initialize centroids
        n_samples = vectors.shape[0]
        centroid_indices = np.random.choice(n_samples, k, replace=False)
        centroids = vectors[centroid_indices]
        
        for _ in range(max_iters):
            # Assign points to nearest centroids
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([vectors[labels == i].mean(axis=0) for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return centroids, labels
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def search(
            self, 
            query_vector: np.ndarray, 
            k: int = 5,
            threshold: float = None,
            use_index: bool = True,
        ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Search for k nearest vectors using cosine similarity.
        Returns list of tuples (id, similarity_score, metadata).
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension must be {self.dimension}")
            
        if not use_index or not self._is_indexed:
            # Exact search
            similarities = [(
                record.id,
                self._cosine_similarity(query_vector, record.vector),
                record.metadata
            ) for record in self.vectors]
            if threshold:
                similarities = [s for s in similarities if s[1] > threshold]
            return heapq.nlargest(k, similarities, key=lambda x: x[1])
            
        # Approximate search using index
        # Find nearest centroid
        centroid_distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_centroid = np.argmin(centroid_distances)
        
        # Search only within the nearest cluster
        candidate_indices = self.centroid_map[nearest_centroid]
        similarities = [(
            self.vectors[idx].id,
            self._cosine_similarity(query_vector, self.vectors[idx].vector),
            self.vectors[idx].metadata
        ) for idx in candidate_indices]
        
        return heapq.nlargest(min(k, len(similarities)), similarities, key=lambda x: x[1])
