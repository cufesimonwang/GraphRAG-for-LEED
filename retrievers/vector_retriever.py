import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .base import BaseRetriever

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Retrieves relevant text chunks using vector similarity search with FAISS.
    """

    def __init__(self, config: Dict[str, Any], neo4j_manager: Any = None):
        """
        Initialize the VectorRetriever.

        Args:
            config: A dictionary containing configuration parameters.
            neo4j_manager: An instance of Neo4jManager (not used by this retriever but part of BaseRetriever).
        """
        super().__init__(config, neo4j_manager=None) # Explicitly pass None for neo4j_manager

        paths_config = self.config.get('paths', {})
        embedding_config = self.config.get('embedding', {})

        # Load FAISS index
        faiss_index_path_str = paths_config.get('faiss_index')
        if not faiss_index_path_str:
            logger.error("FAISS index path ('paths.faiss_index') not found in configuration.")
            raise ValueError("FAISS index path missing from configuration.")
        self.faiss_index_path = Path(faiss_index_path_str)
        if not self.faiss_index_path.exists():
            logger.error(f"FAISS index file not found at: {self.faiss_index_path}")
            raise FileNotFoundError(f"FAISS index file not found at: {self.faiss_index_path}")
        try:
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            logger.info(f"Successfully loaded FAISS index from {self.faiss_index_path} with {self.faiss_index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {self.faiss_index_path}: {e}")
            raise

        # Load document chunks
        # This path should point to the master JSONL file used to build the FAISS index.
        rag_chunks_path_str = paths_config.get('rag_chunks_jsonl')
        if not rag_chunks_path_str:
            logger.error("RAG chunks JSONL path ('paths.rag_chunks_jsonl') not found in configuration.")
            raise ValueError("RAG chunks JSONL path missing from configuration.")
        self.rag_chunks_path = Path(rag_chunks_path_str)
        if not self.rag_chunks_path.exists():
            logger.error(f"RAG chunks file not found at: {self.rag_chunks_path}")
            raise FileNotFoundError(f"RAG chunks file not found at: {self.rag_chunks_path}")
        
        self.chunks: List[Dict[str, Any]] = []
        try:
            with open(self.rag_chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.chunks.append(json.loads(line.strip()))
            logger.info(f"Successfully loaded {len(self.chunks)} chunks from {self.rag_chunks_path}.")
            if self.faiss_index.ntotal != len(self.chunks):
                logger.warning(f"Mismatch between FAISS index size ({self.faiss_index.ntotal}) and loaded chunks ({len(self.chunks)}). Ensure they correspond.")
        except Exception as e:
            logger.error(f"Error loading chunks from {self.rag_chunks_path}: {e}")
            raise

        # Initialize embedding model
        self.model_provider = embedding_config.get('model_provider', 'sentence-transformers')
        self.embedding_model_name: Optional[str] = None
        self.embedding_dim: Optional[int] = embedding_config.get('dimension')

        if self.model_provider == 'sentence-transformers':
            self.embedding_model_name = embedding_config.get('st_model_name')
            if not self.embedding_model_name:
                logger.error("SentenceTransformer model name ('embedding.st_model_name') not specified.")
                raise ValueError("SentenceTransformer model name missing.")
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Initialized SentenceTransformer model: {self.embedding_model_name}")
                if not self.embedding_dim: # Infer if not provided
                    self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    logger.info(f"Inferred embedding dimension for ST model: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Error initializing SentenceTransformer model '{self.embedding_model_name}': {e}")
                raise
        elif self.model_provider == 'openai':
            self.embedding_model_name = embedding_config.get('openai_model')
            if not self.embedding_model_name:
                logger.error("OpenAI embedding model name ('embedding.openai_model') not specified.")
                raise ValueError("OpenAI embedding model name missing.")
            try:
                self.embedding_model = OpenAI(api_key=self.config.get('llm', {}).get('api_key')) # Assumes API key is under 'llm'
                logger.info(f"Initialized OpenAI client for embeddings (model: {self.embedding_model_name}).")
                if not self.embedding_dim: # Should be specified in config for OpenAI
                     logger.warning(f"Embedding dimension for OpenAI model {self.embedding_model_name} not specified in config. This is required.")
                     # Or raise ValueError("Embedding dimension for OpenAI model must be specified in config.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                raise
        else:
            logger.error(f"Unsupported embedding model provider: {self.model_provider}")
            raise ValueError(f"Unsupported embedding model provider: {self.model_provider}")
        
        if not self.embedding_dim:
            logger.error("Embedding dimension could not be determined. Please specify 'embedding.dimension' in config.")
            raise ValueError("Embedding dimension missing.")
        
        if self.faiss_index.d != self.embedding_dim:
            logger.error(f"Mismatch between FAISS index dimension ({self.faiss_index.d}) and model embedding dimension ({self.embedding_dim}).")
            raise ValueError("FAISS index and embedding model dimension mismatch.")


    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the initialized model.
        """
        text = text.replace("\n", " ") # Normalize newlines
        if self.model_provider == 'sentence-transformers':
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist() # type: ignore
        elif self.model_provider == 'openai':
            if not isinstance(self.embedding_model, OpenAI): # Should not happen if init is correct
                raise TypeError("OpenAI client not properly initialized for embeddings.")
            response = self.embedding_model.embeddings.create(input=[text], model=self.embedding_model_name)
            return response.data[0].embedding
        else:
            # Should not be reached if __init__ is correct
            raise ValueError(f"Embeddings requested for unsupported provider: {self.model_provider}")

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, List[Any]]:
        """
        Retrieve relevant text chunks based on the query using vector similarity.
        """
        effective_top_k = self._get_top_k(top_k)
        
        logger.info(f"Retrieving top {effective_top_k} chunks for query: '{query[:100]}...'")
        query_embedding = self._get_embedding(query)
        
        # FAISS search expects a 2D array (batch of embeddings)
        query_embedding_np = np.array([query_embedding], dtype='float32')
        
        try:
            distances, indices = self.faiss_index.search(query_embedding_np, effective_top_k)
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return {
                "chunks": [], 
                "kg_paths": [], 
                "metadata": {"query": query, "error": str(e)}
            }

        formatted_results: List[Dict[str, Any]] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            
            if idx < 0 or idx >= len(self.chunks): # FAISS can return -1 if not enough neighbors
                logger.warning(f"FAISS search returned invalid index {idx}. Skipping.")
                continue
                
            chunk = self.chunks[idx]
            # L2 distance to similarity score (0 to 1, higher is better)
            # This is a common way, but might need adjustment based on distance distribution
            similarity_score = 1 / (1 + float(dist)) 
            
            formatted_results.append({
                "text": chunk.get('text', ''),
                "score": similarity_score,
                "doc_id": chunk.get('doc_id'),
                "section": chunk.get('section'),
                "page": chunk.get('page'),
                # Assuming ContentExtractor adds a unique chunk_id to each object in the JSONL
                # If not, this might be None or an error if 'chunk_id' is expected by consumers.
                "chunk_id": chunk.get('chunk_id', f"chunk_idx_{idx}") 
            })
        
        logger.info(f"Retrieved {len(formatted_results)} chunks.")
        return {
            "chunks": formatted_results,
            "kg_paths": [], # VectorRetriever does not produce KG paths
            "metadata": {
                "query": query,
                "retrieval_strategy": "VectorRetriever",
                "top_k": effective_top_k,
                "embedding_model": self.embedding_model_name
            }
        }

if __name__ == '__main__':
    # This is a placeholder for basic testing.
    # It requires:
    # 1. A `config/config.yaml` with `paths.faiss_index`, `paths.rag_chunks_jsonl`, and `embedding` sections.
    # 2. The FAISS index file and the JSONL chunk file to exist at those paths.
    # 3. If using OpenAI, the API key needs to be available.
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Running basic VectorRetriever test...")

    # Create dummy config, FAISS index, and chunks file for test
    test_config_dir = Path("config")
    test_config_dir.mkdir(exist_ok=True)
    test_config_path = test_config_dir / "test_vector_retriever_config.yaml"
    
    test_data_dir = Path("test_data_vector_retriever")
    test_data_dir.mkdir(exist_ok=True)
    faiss_index_file = test_data_dir / "test.index"
    chunks_jsonl_file = test_data_dir / "test_chunks.jsonl"

    # Dummy config
    dummy_config_data = {
        "paths": {
            "faiss_index": str(faiss_index_file),
            "rag_chunks_jsonl": str(chunks_jsonl_file)
        },
        "embedding": {
            "model_provider": "sentence-transformers",
            "st_model_name": "all-MiniLM-L6-v2", # A common small model
            "dimension": 384 
        },
        "retrieval": {"top_k": 3}
    }
    with open(test_config_path, 'w') as f:
        yaml.dump(dummy_config_data, f)

    # Dummy chunks
    dummy_chunks_data = [
        {"doc_id": "doc1", "section": "Intro", "text": "This is the first test document.", "page": 1, "chunk_id": "doc1_chunk0"},
        {"doc_id": "doc1", "section": "Body", "text": "Another sentence from the first document.", "page": 1, "chunk_id": "doc1_chunk1"},
        {"doc_id": "doc2", "section": "Main", "text": "Second document, with different content.", "page": 1, "chunk_id": "doc2_chunk0"},
        {"doc_id": "doc2", "section": "Conclusion", "text": "Final thoughts from the second document.", "page": 2, "chunk_id": "doc2_chunk1"}
    ]
    with open(chunks_jsonl_file, 'w', encoding='utf-8') as f:
        for chunk in dummy_chunks_data:
            f.write(json.dumps(chunk) + '\n')
            
    # Create dummy FAISS index
    try:
        temp_st_model = SentenceTransformer(dummy_config_data["embedding"]["st_model_name"])
        embeddings = temp_st_model.encode([chunk['text'] for chunk in dummy_chunks_data])
        dimension = embeddings.shape[1]
        if dimension != dummy_config_data["embedding"]["dimension"]:
             logger.warning(f"Actual dimension {dimension} != configured dimension {dummy_config_data['embedding']['dimension']}. Updating config for test.")
             dummy_config_data["embedding"]["dimension"] = dimension # Fix for the test
             with open(test_config_path, 'w') as f:
                yaml.dump(dummy_config_data, f)


        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype='float32'))
        faiss.write_index(index, str(faiss_index_file))
        logger.info(f"Dummy FAISS index created at {faiss_index_file} with {index.ntotal} vectors.")

        retriever = VectorRetriever(config=dummy_config_data)
        test_query = "first document"
        results = retriever.retrieve(query=test_query)
        
        logger.info(f"\nResults for query '{test_query}':")
        for chunk in results.get("chunks", []):
            logger.info(f"  Text: '{chunk['text']}', Score: {chunk['score']:.4f}, Source: {chunk['doc_id']}")
        logger.info(f"Metadata: {results.get('metadata')}")

    except Exception as e:
        logger.error(f"Error during VectorRetriever test: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        if test_config_path.exists():
            test_config_path.unlink()
        if faiss_index_file.exists():
            faiss_index_file.unlink()
        if chunks_jsonl_file.exists():
            chunks_jsonl_file.unlink()
        if test_data_dir.exists():
            try:
                test_data_dir.rmdir() # Only removes if empty
            except OSError:
                 logger.warning(f"Could not remove test_data_dir {test_data_dir} as it might not be empty or access issues.")
        logger.info("VectorRetriever test finished. Dummy files cleaned up.")
