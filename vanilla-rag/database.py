import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import heapq

@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: Optional[Dict] = None

class VectorDB:
    def __init__(self, dimension: int, index_size: int = 10):
        """
        Initialize the vector database.
        
        Args:
            dimension: The dimension of vectors to be stored
            index_size: Number of clusters for approximate search
        """
        self.dimension = dimension
        self.index_size = index_size
        self.vectors: List[VectorRecord] = []
        self.indexed = False
        self.centroids = None
        self.centroid_map = {}  # Maps centroid index to vector indices
        
    def add(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add a vector to the database."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")
            
        vector = np.array(vector, dtype=np.float32)
        record = VectorRecord(id=id, vector=vector, metadata=metadata)
        self.vectors.append(record)
        self.indexed = False
        
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
            
        self.indexed = True
        
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
        
    def search(self, query_vector: np.ndarray, k: int = 5, use_index: bool = True) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Search for k nearest vectors using cosine similarity.
        Returns list of tuples (id, similarity_score, metadata).
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension must be {self.dimension}")
            
        if not use_index or not self.indexed:
            # Exact search
            similarities = [(
                record.id,
                self._cosine_similarity(query_vector, record.vector),
                record.metadata
            ) for record in self.vectors]
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
