import logging
from typing import Dict, List, Any, Optional
from pathlib import Path # Added for main test
import yaml # Added for main test
import json # Added for main test
import numpy as np # Added for main test
import faiss # Added for main test


from sentence_transformers import CrossEncoder, SentenceTransformer # Added SentenceTransformer for main test

from .base import BaseRetriever
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from kg_construction.neo4j_manager import Neo4jManager # For type hinting and main test

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Combines results from VectorRetriever and GraphRetriever.
    Optionally reranks vector chunks and applies configured weighting.
    """

    def __init__(self, 
                 config: Dict[str, Any], 
                 neo4j_manager: Neo4jManager, # Neo4jManager is now explicitly required
                 vector_retriever: VectorRetriever, 
                 graph_retriever: GraphRetriever):
        """
        Initialize the HybridRetriever.

        Args:
            config: A dictionary containing configuration parameters.
            neo4j_manager: An instance of Neo4jManager (passed to BaseRetriever and potentially used if needed).
            vector_retriever: An instance of VectorRetriever.
            graph_retriever: An instance of GraphRetriever.
        """
        super().__init__(config, neo4j_manager) # Pass neo4j_manager to BaseRetriever
        
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

        # Config for HybridRetriever itself is under 'retrieval', not 'retriever'
        # Correcting config access based on typical structure and subtask description
        hybrid_config = self.config.get('retrieval', {}).get('hybrid_retriever', {})
        self.rag_weight = hybrid_config.get('rag_weight', 0.5)
        self.graph_weight = hybrid_config.get('graph_weight', 0.5)
        self.fusion_method = hybrid_config.get('fusion_method', 'weighted_sum') 
        
        self.use_reranker = hybrid_config.get('use_reranker', False)
        self.reranker: Optional[CrossEncoder] = None
        if self.use_reranker:
            reranker_model_name = hybrid_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                logger.info(f"Initialized CrossEncoder reranker model: {reranker_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoder model '{reranker_model_name}': {e}")
                self.use_reranker = False 

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, List[Any]]:
        """
        Retrieve relevant information using both vector and graph retrievers,
        optionally rerank vector chunks, and apply weights.
        """
        effective_top_k = self._get_top_k(top_k) # Uses general retrieval:top_k from BaseRetriever
        logger.info(f"HybridRetriever called for query: '{query[:100]}...', top_k: {effective_top_k}")

        vector_results_dict = self.vector_retriever.retrieve(query, top_k=effective_top_k)
        retrieved_chunks = vector_results_dict.get("chunks", [])
        
        if self.use_reranker and self.reranker and retrieved_chunks:
            logger.info(f"Reranking {len(retrieved_chunks)} vector chunks...")
            rerank_pairs = [(query, chunk['text']) for chunk in retrieved_chunks]
            try:
                rerank_scores = self.reranker.predict(rerank_pairs, show_progress_bar=False)
                for chunk, new_score in zip(retrieved_chunks, rerank_scores):
                    chunk['original_score'] = chunk.get('score') 
                    chunk['score'] = float(new_score) 
                retrieved_chunks.sort(key=lambda x: x['score'], reverse=True)
                logger.info("Finished reranking vector chunks.")
            except Exception as e:
                 logger.error(f"Error during reranking: {e}")
        
        graph_results_dict = self.graph_retriever.retrieve(query, top_k=effective_top_k)
        retrieved_kg_paths = graph_results_dict.get("kg_paths", [])

        # Apply weights. Note: This modifies scores in-place.
        if self.fusion_method == 'weighted_sum': # Or other methods that directly use weights on scores
            for chunk in retrieved_chunks:
                chunk['score'] = chunk.get('score', 0.0) * self.rag_weight 
            for path in retrieved_kg_paths:
                path['score'] = path.get('score', 0.0) * self.graph_weight
        
        # Sort by potentially modified scores before trimming
        retrieved_chunks.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        retrieved_kg_paths.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        final_chunks = retrieved_chunks[:effective_top_k]
        final_kg_paths = retrieved_kg_paths[:effective_top_k]

        logger.info(f"Hybrid retrieval complete. Returning {len(final_chunks)} chunks and {len(final_kg_paths)} KG paths.")
        
        return {
            "chunks": final_chunks,
            "kg_paths": final_kg_paths,
            "metadata": {
                "query": query,
                "retrieval_strategy": "HybridRetriever",
                "top_k_effective": effective_top_k,
                "fusion_method": self.fusion_method,
                "rag_weight": self.rag_weight,
                "graph_weight": self.graph_weight,
                "reranker_used_for_chunks": self.use_reranker and bool(self.reranker),
                "vector_retriever_metadata": vector_results_dict.get("metadata", {}),
                "graph_retriever_metadata": graph_results_dict.get("metadata", {})
            }
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running basic HybridRetriever test...")

    # Dummy config setup
    test_config_dir = Path("config")
    test_config_dir.mkdir(exist_ok=True)
    test_config_path = test_config_dir / "test_hybrid_retriever_config.yaml"
    test_data_dir = Path("test_data_hybrid") 
    test_data_dir.mkdir(exist_ok=True)
    faiss_index_file = test_data_dir / "test.index"
    chunks_jsonl_file = test_data_dir / "test_chunks.jsonl"
    dummy_prompts_file = test_config_dir / "dummy_prompts_hybrid.yaml"

    dummy_config_data = {
        "paths": {
            "faiss_index": str(faiss_index_file),
            "rag_chunks_jsonl": str(chunks_jsonl_file),
            "prompts_file": str(dummy_prompts_file) 
        },
        "embedding": {"model_provider": "sentence-transformers", "st_model_name": "all-MiniLM-L6-v2", "dimension": 384},
        "llm": {"provider": "openai", "model_name": "gpt-3.5-turbo-instruct", "api_key": "YOUR_OPENAI_API_KEY_OR_SKIP"},
        "retrieval": {
            "top_k": 3, 
            "graph_retriever": {"k_hop": 1, "query_parsing_strategy": "llm", "max_entities_from_query": 1, "paths_per_entity_limit": 1},
            "hybrid_retriever": {"rag_weight": 0.7, "graph_weight": 0.3, "use_reranker": True, "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        },
        "prompts": { "retrieval": { "query_to_entity_prompt": "Extract 1 entity from: {query}. JSON list."}},
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "neo4j"}
    }
    if dummy_config_data["llm"]["api_key"] == "YOUR_OPENAI_API_KEY_OR_SKIP":
        logger.warning("OpenAI API key not set. LLM parsing for GraphRetriever in test might fail or use stubs.")
        dummy_config_data["retrieval"]["graph_retriever"]["query_parsing_strategy"] = "heuristic"

    with open(test_config_path, 'w') as f: yaml.dump(dummy_config_data, f)
    with open(dummy_prompts_file, 'w') as f: yaml.dump(dummy_config_data["prompts"], f)
    
    dummy_chunks_data = [
        {"doc_id": "doc1", "text": "LEED v4 emphasizes commissioning.", "chunk_id": "d1c0"},
        {"doc_id": "doc2", "text": "Acoustic performance is key in schools.", "chunk_id": "d2c0"},
    ]
    with open(chunks_jsonl_file, 'w', encoding='utf-8') as f:
        for chunk in dummy_chunks_data: f.write(json.dumps(chunk) + '\n')
    
    nm_test_hr = None
    try:
        temp_st_model = SentenceTransformer(dummy_config_data["embedding"]["st_model_name"])
        embeddings = temp_st_model.encode([chunk['text'] for chunk in dummy_chunks_data])
        index = faiss.IndexFlatL2(dummy_config_data["embedding"]["dimension"])
        index.add(np.array(embeddings, dtype='float32'))
        faiss.write_index(index, str(faiss_index_file))

        nm_test_hr = Neo4jManager(config_path=str(test_config_path))
        with nm_test_hr.driver.session(database=nm_test_hr.database) as s:
            s.run("MERGE (:Entity {name: 'LEED v4', type: 'Standard', doc_id:'test_hr', chunk_id:'hr_c1'})")
            s.run("MERGE (:Entity {name: 'commissioning', type: 'Process', doc_id:'test_hr', chunk_id:'hr_c2'})")
            s.run("MATCH (e1:Entity {name:'LEED v4'}), (e2:Entity {name:'commissioning'}) MERGE (e1)-[:HAS_TOPIC]->(e2)")

        vec_retriever = VectorRetriever(config=dummy_config_data) 
        graph_retriever = GraphRetriever(config=dummy_config_data, neo4j_manager=nm_test_hr) 
        
        hybrid_retriever = HybridRetriever(dummy_config_data, nm_test_hr, vec_retriever, graph_retriever)
        
        test_query = "LEED v4 commissioning"
        results = hybrid_retriever.retrieve(query=test_query)
        
        logger.info(f"\nHybrid Results for query '{test_query}':")
        logger.info("Chunks:")
        for chunk in results.get("chunks", []): logger.info(f"  Text: '{chunk['text']}', Score: {chunk.get('score', -1):.4f}")
        logger.info("KG Paths:")
        for path in results.get("kg_paths", []): logger.info(f"  Path: {path['text']} (Score: {path.get('score', -1):.4f})")
        logger.info(f"Metadata: {json.dumps(results.get('metadata'), indent=2)}")

    except Exception as e:
        logger.error(f"Error during HybridRetriever test: {e}", exc_info=True)
    finally:
        if nm_test_hr: nm_test_hr.close()
        if test_config_path.exists(): test_config_path.unlink(missing_ok=True)
        if dummy_prompts_file.exists(): dummy_prompts_file.unlink(missing_ok=True)
        if faiss_index_file.exists(): faiss_index_file.unlink(missing_ok=True)
        if chunks_jsonl_file.exists(): chunks_jsonl_file.unlink(missing_ok=True)
        if test_data_dir.exists():
            try: test_data_dir.rmdir()
            except OSError: logger.warning(f"Could not remove test_data_dir {test_data_dir}")
        logger.info("HybridRetriever test finished.")