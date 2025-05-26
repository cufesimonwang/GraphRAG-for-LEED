from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Defines the common interface for retrieving information.
    """

    def __init__(self, config: Dict[str, Any], neo4j_manager: Any = None):
        """
        Initialize the retriever with configuration.

        Args:
            config: A dictionary containing configuration parameters.
            neo4j_manager: An instance of Neo4jManager for graph-based retrievers.
        """
        self.config = config
        self.neo4j_manager = neo4j_manager

    @abstractmethod
    def retrieve(self, query: str, top_k: int = None) -> Dict[str, List[Any]]:
        """
        Retrieve relevant information based on the query.

        Args:
            query: The user's query string.
            top_k: The maximum number of results to return. 
                   If None, uses a default value from the config.

        Returns:
            A dictionary containing the retrieval results, typically in the format:
            {
              "chunks": [
                  {"text": "chunk_text_1", "score": 0.9, "source": "doc_A", "page": 1, "chunk_id": "chunk_xyz"},
                  ...
              ], // For text/vector based retrieval
              "kg_paths": [
                  {"path_text": "EntityA -> relates_to -> EntityB", "score": 0.85, "nodes": [...], "edges": [...]},
                  ...
              ], // For graph-based retrieval
              "metadata": {
                  "query": query,
                  "retrieval_strategy": self.__class__.__name__,
                  "top_k": top_k,
                  // ... other relevant metadata
              }
            }
            The exact keys ('chunks', 'kg_paths') might vary based on the retriever type,
            but the overall structure should be a dictionary of lists.
        """
        pass

    def _get_top_k(self, top_k: int = None) -> int:
        """
        Helper method to determine the effective top_k value.
        Uses the provided top_k or falls back to config.
        """
        if top_k is not None:
            return top_k
        # Assumes retriever-specific config might have its own top_k
        # or a general retrieval top_k.
        # Example: self.config.get('retriever_settings', {}).get('top_k', 5)
        # This needs to be adapted based on actual config structure for each retriever.
        # For now, a placeholder from a potential general retrieval config:
        return self.config.get('retrieval', {}).get('top_k', 5)
