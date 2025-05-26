import logging
from typing import Dict, Any

from .base import BaseRetriever
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridRetriever
from kg_construction.neo4j_manager import Neo4jManager # Adjust path if necessary based on final structure

logger = logging.getLogger(__name__)

def get_retriever(config: Dict[str, Any], neo4j_manager: Neo4jManager = None) -> BaseRetriever:
    """
    Factory function to instantiate and return the appropriate retriever based on config.

    Args:
        config: The main configuration dictionary.
        neo4j_manager: An instance of Neo4jManager, required for graph/hybrid retrievers.

    Returns:
        An instance of a class derived from BaseRetriever.

    Raises:
        ValueError: If the specified retriever strategy is unknown or
                    if neo4j_manager is not provided for graph/hybrid retrievers.
    """
    # Corrected to access retriever config under 'retrieval' as per previous subtasks
    retriever_main_config = config.get('retrieval', {}) 
    strategy = retriever_main_config.get('strategy', 'hybrid') # Default to hybrid

    logger.info(f"Attempting to create retriever with strategy: {strategy}")

    if strategy == "vector":
        logger.info("Initializing VectorRetriever.")
        # VectorRetriever's __init__ expects: config, neo4j_manager (optional, defaults to None)
        # It uses config['paths']['faiss_index'], config['paths']['rag_chunks_jsonl'], config['embedding']
        return VectorRetriever(config=config, neo4j_manager=None) 
    
    elif strategy == "graph":
        if not neo4j_manager:
            logger.error("Neo4jManager is required for GraphRetriever.")
            raise ValueError("Neo4jManager must be provided for 'graph' strategy.")
        logger.info("Initializing GraphRetriever.")
        # GraphRetriever's __init__ expects: config, neo4j_manager
        # It uses config['retrieval']['graph_retriever'], config['llm'], config['prompts']
        return GraphRetriever(config=config, neo4j_manager=neo4j_manager)
    
    elif strategy == "hybrid":
        if not neo4j_manager:
            logger.error("Neo4jManager is required for HybridRetriever (for its GraphRetriever component).")
            raise ValueError("Neo4jManager must be provided for 'hybrid' strategy.")
        
        logger.info("Initializing HybridRetriever, which requires VectorRetriever and GraphRetriever.")
        
        vector_retriever_instance = VectorRetriever(config=config, neo4j_manager=None) 
        graph_retriever_instance = GraphRetriever(config=config, neo4j_manager=neo4j_manager)
        
        logger.info("Initializing HybridRetriever with its sub-retrievers.")
        # HybridRetriever's __init__ expects: config, neo4j_manager, vector_retriever, graph_retriever
        # It uses config['retrieval']['hybrid_retriever']
        return HybridRetriever(config=config, 
                               vector_retriever=vector_retriever_instance,
                               graph_retriever=graph_retriever_instance,
                               neo4j_manager=neo4j_manager) 
    
    else:
        logger.error(f"Unknown retriever strategy: {strategy}")
        raise ValueError(f"Unknown retriever strategy: {strategy}")

# Example usage (optional, for testing within the file)
if __name__ == '__main__':
    import os
    import faiss
    import numpy as np
    import json
    import yaml # For loading/dumping full config for Neo4jManager in test
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    
    # Create a more complete dummy config that Neo4jManager can use
    # And paths that VectorRetriever expects
    current_dir = Path(__file__).parent
    test_config_path = current_dir.parent / "config" / "test_factory_config.yaml" # Place in config folder
    test_data_output_dir = current_dir.parent / "data" / "output" # For FAISS and chunks
    test_processed_dir = current_dir.parent / "data" / "processed" # For jsonl_output_dir
    
    test_data_output_dir.mkdir(parents=True, exist_ok=True)
    test_processed_dir.mkdir(parents=True, exist_ok=True)
    
    faiss_index_path = test_data_output_dir / "test_factory.index"
    rag_chunks_jsonl_path = test_data_output_dir / "test_factory_chunks.jsonl"
    # Path for jsonl_output_dir (used by ContentExtractor, but KGBuilder reads from it)
    # This path is also used by KGBuilder to find JSONL files.
    # For factory test, it's not directly used by get_retriever but good to be aware of related paths.
    jsonl_output_dir_path = test_processed_dir / "jsonl_output_factory_test" 
    jsonl_output_dir_path.mkdir(exist_ok=True)


    dummy_config = {
        "llm": { 
            "provider": "openai", 
            "model_name": "gpt-3.5-turbo-instruct", 
            "api_key": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE") 
        },
        "embedding": { 
            "model_provider": "sentence-transformers",
            "st_model_name": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "paths": { 
            "faiss_index": str(faiss_index_path),
            "rag_chunks_jsonl": str(rag_chunks_jsonl_path),
            "prompts_file": str(current_dir.parent / "config" / "prompts.yaml"), # Assuming prompts.yaml exists
            "jsonl_output_dir": str(jsonl_output_dir_path) # For ContentExtractor/KGBuilder context
        },
        "retrieval": {
            "strategy": "hybrid", 
            "top_k": 3,
            "graph_retriever": {
                "k_hop": 1,
                "query_parsing_strategy": "llm", 
                "max_entities_from_query": 2,
                "paths_per_entity_limit": 2,
                "llm_model_for_parsing": "gpt-3.5-turbo-instruct" # Specify if different
            },
            "hybrid_retriever": {
                "rag_weight": 0.6,
                "graph_weight": 0.4,
                "use_reranker": False 
            }
        },
        "neo4j": { 
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password") 
        },
        "prompts": { # Ensure prompts structure matches what GraphRetriever expects
            "retrieval": {
                "query_to_entity_prompt": "Extract up to {max_entities_from_query} entities from: {query}. JSON list."
            }
        }
    }

    # Save this dummy config to be read by Neo4jManager
    test_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_config_path, 'w') as f:
        yaml.dump(dummy_config, f)

    logger.info(f"Dummy config saved to: {test_config_path}")
    logger.info(f"FAISS index will be at: {faiss_index_path}")
    logger.info(f"RAG chunks will be at: {rag_chunks_jsonl_path}")


    dimension = dummy_config['embedding']['dimension']
    num_dummy_vectors = 5
    dummy_vectors = np.float32(np.random.random((num_dummy_vectors, dimension)))
    dummy_index = faiss.IndexFlatL2(dimension)
    if num_dummy_vectors > 0:
        dummy_index.add(dummy_vectors)
    faiss.write_index(dummy_index, str(faiss_index_path))

    dummy_chunks_content = []
    for i in range(num_dummy_vectors):
        dummy_chunks_content.append({
            "doc_id": f"doc_factory_{i//2}", "section": f"section_factory_{i%2}", 
            "text": f"This is dummy chunk text for factory test {i}.", "page": i//3,
            "chunk_id": f"chunk_factory_{i}"
        })
    with open(rag_chunks_jsonl_path, 'w') as f:
        for chunk_item in dummy_chunks_content:
            f.write(json.dumps(chunk_item) + '\n')

    test_neo4j_manager = None
    if dummy_config['retrieval']['strategy'] in ['graph', 'hybrid']:
        try:
            # Neo4jManager expects a config *file path*
            test_neo4j_manager = Neo4jManager(config_path=str(test_config_path))
            logger.info("Dummy Neo4jManager initialized for testing.")
            # Create a dummy constraint for graph retriever to work
            try:
                test_neo4j_manager.create_constraints_and_indexes() # Use the method
            except Exception as e_neo:
                logger.warning(f"Could not create Neo4j constraint/indexes for testing: {e_neo}. Graph retriever might fail if DB is empty.")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4jManager for testing: {e}. Graph/Hybrid retriever tests might fail or be limited.")
            test_neo4j_manager = None


    try:
        # Test Hybrid
        logger.info("\n--- Testing Hybrid Retriever ---")
        dummy_config['retrieval']['strategy'] = 'hybrid'
        retriever_instance_hybrid = get_retriever(dummy_config, test_neo4j_manager)
        logger.info(f"Successfully created retriever: {type(retriever_instance_hybrid).__name__}")
        results_hybrid = retriever_instance_hybrid.retrieve("dummy query for hybrid")
        logger.info(f"Hybrid results: Chunks: {len(results_hybrid.get('chunks',[]))}, Paths: {len(results_hybrid.get('kg_paths',[]))}")


        # Test Vector
        logger.info("\n--- Testing Vector Retriever ---")
        dummy_config['retrieval']['strategy'] = 'vector'
        retriever_instance_vector = get_retriever(dummy_config, test_neo4j_manager) # neo4j_manager is None for vector
        logger.info(f"Successfully created retriever: {type(retriever_instance_vector).__name__}")
        results_vector = retriever_instance_vector.retrieve("dummy query for vector")
        logger.info(f"Vector results: Chunks: {len(results_vector.get('chunks',[]))}")


        # Test Graph (if neo4j_manager was initialized)
        if test_neo4j_manager:
            logger.info("\n--- Testing Graph Retriever ---")
            dummy_config['retrieval']['strategy'] = 'graph'
            retriever_instance_graph = get_retriever(dummy_config, test_neo4j_manager)
            logger.info(f"Successfully created retriever: {type(retriever_instance_graph).__name__}")
            results_graph = retriever_instance_graph.retrieve("dummy query for graph") # OpenAI key might be needed here
            logger.info(f"Graph results: Paths: {len(results_graph.get('kg_paths',[]))}")
        else:
            logger.warning("Skipping Graph Retriever test as Neo4jManager was not initialized.")

    except ValueError as e:
        logger.error(f"Error creating retriever: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during factory test: {e}", exc_info=True)
    finally:
        if test_neo4j_manager:
            test_neo4j_manager.close()
            logger.info("Closed dummy Neo4jManager connection.")
        
        # Clean up dummy files
        if faiss_index_path.exists(): faiss_index_path.unlink()
        if rag_chunks_jsonl_path.exists(): rag_chunks_jsonl_path.unlink()
        if test_config_path.exists(): test_config_path.unlink()
        # Clean up dummy dir if empty
        try: jsonl_output_dir_path.rmdir()
        except OSError: pass
        try: test_data_output_dir.rmdir()
        except OSError: pass
        try: test_config_path.parent.rmdir() # Remove config folder if empty
        except OSError: pass


        logger.info("Retriever factory test finished.")
