import logging
import json
from typing import Dict, List, Any, Optional

from neo4j.graph import Path as Neo4jPath, Node, Relationship

from .base import BaseRetriever
from kg_construction.neo4j_manager import Neo4jManager # Assuming Neo4jManager is accessible

# LLM Client imports - choose based on config (OpenAI or local/SentenceTransformer-based for simple tasks)
from openai import OpenAI
# from sentence_transformers import SentenceTransformer # If using ST for some local LLM task

logger = logging.getLogger(__name__)

class GraphRetriever(BaseRetriever):
    """
    Retrieves relevant information by querying a Neo4j knowledge graph
    based on entities extracted from the user's query.
    """

    def __init__(self, config: Dict[str, Any], neo4j_manager: Neo4jManager):
        """
        Initialize the GraphRetriever.

        Args:
            config: A dictionary containing configuration parameters.
            neo4j_manager: An instance of Neo4jManager.
        """
        super().__init__(config, neo4j_manager)
        if not self.neo4j_manager:
            raise ValueError("Neo4jManager instance is required for GraphRetriever.")

        retriever_config = self.config.get('retrieval', {}).get('graph_retriever', {})
        self.k_hop = retriever_config.get('k_hop', 2)
        self.query_parsing_strategy = retriever_config.get('query_parsing_strategy', 'llm')
        self.max_entities_from_query = retriever_config.get('max_entities_from_query', 3)
        self.paths_per_entity_limit = retriever_config.get('paths_per_entity_limit', 10)
        
        # Initialize LLM client for query parsing if strategy is 'llm'
        self.llm_client: Optional[Any] = None
        if self.query_parsing_strategy == 'llm':
            llm_config = self.config.get('llm', {})
            provider = llm_config.get('provider', 'openai') # Default to openai
            # model_for_parsing = retriever_config.get('llm_model_for_parsing', llm_config.get('model_name'))
            # Using main model_name for now, can be overridden by llm_model_for_parsing if needed
            self.llm_model_name = retriever_config.get('llm_model_for_parsing', llm_config.get('model_name'))


            if provider == 'openai':
                self.llm_client = OpenAI(api_key=llm_config.get('api_key'))
                if not llm_config.get('api_key'):
                    logger.warning("OpenAI API key not found in config for GraphRetriever query parsing.")
            # Add other providers like 'anthropic' or 'local' if needed, similar to other components
            # elif provider == 'local':
            #     self.llm_client = SentenceTransformer(model_for_parsing) # Or other local setup
            else:
                logger.warning(f"LLM provider '{provider}' for query parsing not fully supported or configured. Query parsing might fail.")
            
            if not self.llm_client:
                 logger.error("LLM client for query parsing could not be initialized.")
                 # Potentially raise error or fallback to heuristic
                 # raise ValueError("LLM client for GraphRetriever query parsing failed to initialize.")
        
        # Load prompts (assuming they are loaded in a central place or passed via config if not using a global loader)
        # For this example, assuming prompts are accessible via self.config or a helper
        self.prompts = config.get('prompts', {}) # Expecting prompts to be pre-loaded into main config
        if not self.prompts and self.query_parsing_strategy == 'llm':
            logger.warning("Prompts not found in config. LLM-based query parsing might fail.")


    def _get_llm_response(self, prompt: str) -> str:
        """Helper to get response from the configured LLM client."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized for GraphRetriever.")

        provider = self.config.get('llm', {}).get('provider', 'openai')
        
        try:
            if provider == 'openai':
                if not isinstance(self.llm_client, OpenAI): # Type check
                    raise TypeError("LLM client is not an OpenAI instance.")
                completion = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.get('llm',{}).get('temperature', 0.1), # Low temp for extraction
                    max_tokens=100 # Usually enough for a list of entities
                )
                return completion.choices[0].message.content.strip()
            # Add other providers (anthropic, local ST model for simple tasks) here if necessary
            else:
                logger.error(f"Unsupported LLM provider for _get_llm_response: {provider}")
                return "" # Fallback
        except Exception as e:
            logger.error(f"Error getting LLM response for query parsing: {e}")
            return "" # Fallback

    def _parse_query_to_entities(self, query: str) -> List[str]:
        """
        Extracts key entities from the user's query using the configured strategy.
        """
        if self.query_parsing_strategy == "llm":
            if not self.llm_client:
                logger.warning("LLM parsing strategy selected but LLM client not initialized. Returning empty list.")
                return []
            
            prompt_template = self.prompts.get('retrieval', {}).get('query_to_entity_prompt')
            if not prompt_template:
                logger.error("Query to entity prompt not found in prompts configuration.")
                return []
            
            prompt = prompt_template.format(query=query)
            llm_response_str = self._get_llm_response(prompt)
            
            try:
                # Try to find JSON array within the response
                match = json.loads(llm_response_str)
                if isinstance(match, list):
                    # Limit the number of entities
                    return [str(entity) for entity in match[:self.max_entities_from_query]] 
                else:
                    logger.warning(f"LLM response for entity parsing was not a list: {llm_response_str}")
                    return []
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON from LLM response for entity parsing: {llm_response_str}")
                # Fallback: try simple comma separation if it's just a string of items
                if llm_response_str and not llm_response_str.startswith("["):
                    return [e.strip() for e in llm_response_str.split(',')][:self.max_entities_from_query]
                return []
        
        elif self.query_parsing_strategy == "heuristic":
            # Placeholder for heuristic implementation (e.g., NLTK noun phrases)
            logger.warning("Heuristic query parsing not implemented yet. Returning empty list.")
            # Example (requires NLTK and setup):
            # import nltk
            # tokenized = nltk.word_tokenize(query)
            # tagged = nltk.pos_tag(tokenized)
            # # Extract nouns or noun phrases - this is very basic
            # entities = [word for word, tag in tagged if tag.startswith('NN')]
            # return entities[:self.max_entities_from_query]
            return []
        else:
            logger.warning(f"Unknown query_parsing_strategy: {self.query_parsing_strategy}")
            return []

    @staticmethod
    def _execute_path_query(tx, cypher_query: str, entity_name: str, k_hop: int, limit: int) -> List[Dict[str, Any]]:
        """Transaction function to execute a path query and format results."""
        result = tx.run(cypher_query, entity_name=entity_name, k_hop=k_hop, limit=limit)
        paths_data = []
        for record in result:
            path_obj: Neo4jPath = record["path"]
            nodes_data = []
            for node in path_obj.nodes:
                node_props = dict(node.items()) # Get all properties
                nodes_data.append({
                    "id": node.element_id,
                    "labels": list(node.labels), # Get all labels
                    "name": node_props.get('name', 'Unknown Name'), # Assuming 'name' property exists
                    "type": node_props.get('type', next(iter(node.labels), 'UnknownType')) # Fallback type
                })
            
            edges_data = []
            for rel in path_obj.relationships:
                edges_data.append({
                    "id": rel.element_id,
                    "start_node_id": rel.start_node.element_id,
                    "end_node_id": rel.end_node.element_id,
                    "type": rel.type
                })
            paths_data.append({"nodes": nodes_data, "edges": edges_data})
        return paths_data

    def _query_neo4j_for_paths(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Queries Neo4j for k-hop paths starting from the given entities.
        """
        all_paths: List[Dict[str, Any]] = []
        if not self.neo4j_manager or not self.neo4j_manager.driver:
            logger.error("Neo4jManager not available for querying.")
            return all_paths

        # Note: Using string formatting for k_hop is generally safe if k_hop is validated as int.
        # Parameterization for variable length paths like [*1..$k_hop] is not directly supported in Cypher params for the range.
        cypher_query = (
            f"MATCH path = (e:Entity {{name: $entity_name}})-[*1..{self.k_hop}]-(neighbor:Entity) "
            "RETURN path LIMIT $limit"
        )
        
        with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
            for entity_name in entities:
                try:
                    logger.info(f"Querying paths for entity: {entity_name}, k-hop: {self.k_hop}, limit: {self.paths_per_entity_limit}")
                    paths_for_entity = session.read_transaction(
                        self._execute_path_query, 
                        cypher_query, 
                        entity_name=entity_name, 
                        k_hop=self.k_hop,
                        limit=self.paths_per_entity_limit
                    )
                    all_paths.extend(paths_for_entity)
                    logger.debug(f"Found {len(paths_for_entity)} paths for entity '{entity_name}'.")
                except Exception as e:
                    logger.error(f"Error querying Neo4j for paths for entity '{entity_name}': {e}")
        
        # Optional: Deduplicate paths if they are identical (e.g., by comparing node/edge element_ids sequence)
        # For now, returning all found paths.
        logger.info(f"Total paths retrieved from Neo4j: {len(all_paths)}")
        return all_paths

    def _translate_path_to_natural_language(self, path_data: Dict[str, Any]) -> str:
        """
        Translates a graph path (nodes and edges) into a human-readable string.
        """
        if not path_data or not path_data.get("nodes"):
            return "Empty path"

        parts = []
        nodes = path_data["nodes"]
        edges = path_data.get("edges", []) # Edges might be empty for single node paths if k_hop allows 0

        parts.append(f"{nodes[0].get('name', 'Unknown Node')} ({nodes[0].get('type', 'UnknownType')})")
        
        current_node_idx = 0
        for i, edge in enumerate(edges):
            # Find the next node in the path sequence
            # This logic assumes edges are ordered and connect sequentially as per Neo4j Path object
            # and that node list is also ordered.
            
            # This part needs careful handling of node order if not guaranteed by Neo4j Path processing
            # For now, assume nodes[i+1] is the target of edges[i]
            if i + 1 < len(nodes):
                target_node = nodes[i+1]
                parts.append(f"-[{edge.get('type', 'RELATED_TO')}]->")
                parts.append(f"{target_node.get('name', 'Unknown Node')} ({target_node.get('type', 'UnknownType')})")
            else: # Should not happen if path is consistent
                parts.append(f"-[{edge.get('type', 'RELATED_TO')}]-> [Missing Target Node]")


        return " ".join(parts)

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, List[Any]]:
        """
        Retrieves graph paths related to entities parsed from the query.
        """
        effective_top_k = self._get_top_k(top_k)
        logger.info(f"GraphRetriever called for query: '{query}', top_k: {effective_top_k}")

        parsed_entities = self._parse_query_to_entities(query)
        if not parsed_entities:
            logger.info("No entities parsed from query. Returning empty result.")
            return {"chunks": [], "kg_paths": [], "metadata": {"query": query, "parsed_entities": [], "retrieval_strategy": "GraphRetriever"}}

        logger.info(f"Parsed entities from query: {parsed_entities}")
        
        retrieved_paths_raw = self._query_neo4j_for_paths(parsed_entities)
        
        formatted_kg_paths: List[Dict[str, Any]] = []
        for path_data in retrieved_paths_raw:
            path_text = self._translate_path_to_natural_language(path_data)
            # Placeholder score; could be based on path length, node/edge properties, etc.
            score = 1.0 / (1 + len(path_data.get("edges", []))) # Simple score: shorter paths are better

            formatted_kg_paths.append({
                "text": path_text, # Natural language representation
                "score": score,
                "nodes": path_data["nodes"], # Raw node data
                "edges": path_data["edges"]  # Raw edge data
            })
        
        # Sort by score and take top_k (if more paths than top_k)
        # This sorting happens after collecting paths for all entities.
        # If paths_per_entity_limit is used, this top_k might be applied to an already limited set.
        formatted_kg_paths.sort(key=lambda x: x["score"], reverse=True)
        
        final_results = formatted_kg_paths[:effective_top_k]
        logger.info(f"Returning {len(final_results)} KG paths after processing and ranking.")

        return {
            "chunks": [], # GraphRetriever does not return chunks directly
            "kg_paths": final_results,
            "metadata": {
                "query": query,
                "parsed_entities": parsed_entities,
                "retrieval_strategy": "GraphRetriever",
                "top_k_input": top_k, # The original top_k requested
                "top_k_effective": effective_top_k, # The top_k used after config fallback
                "k_hop_configured": self.k_hop,
                "paths_per_entity_limit_configured": self.paths_per_entity_limit,
                "total_raw_paths_found": len(retrieved_paths_raw)
            }
        }

if __name__ == '__main__':
    # This is a placeholder for basic testing.
    # It requires:
    # 1. A `config/config.yaml` with `neo4j` connection details, `llm` (if LLM parsing), 
    #    `retrieval.graph_retriever` settings, and `prompts.retrieval.query_to_entity_prompt`.
    # 2. A running Neo4j instance populated by KnowledgeGraphBuilder.
    
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more verbose test output
    logger.info("Running basic GraphRetriever test...")

    # Create a dummy config for testing
    test_config_dir = Path("config")
    test_config_dir.mkdir(exist_ok=True)
    test_config_path = test_config_dir / "test_graph_retriever_config.yaml"

    dummy_config_data = {
        "neo4j": {
            "uri": "bolt://localhost:7687", # Standard local Neo4j URI
            "user": "neo4j",
            "password": "password", # Replace with your test DB password
            "database": "neo4j"
        },
        "llm": { # Required if query_parsing_strategy is "llm"
            "provider": "openai", 
            "model_name": "gpt-3.5-turbo-instruct", # Cheaper completion model for this simple task
            "api_key": "YOUR_OPENAI_API_KEY" # IMPORTANT: Replace or use env var
        },
        "retrieval": {
            "top_k": 5,
            "graph_retriever": {
                "k_hop": 1, # Keep k_hop small for testing
                "query_parsing_strategy": "llm", 
                "max_entities_from_query": 2,
                "paths_per_entity_limit": 3,
                 # "llm_model_for_parsing": "gpt-3.5-turbo-instruct" # Example if using a specific model
            }
        },
        "prompts": { # Prompts needed for LLM parsing
            "retrieval": {
                "query_to_entity_prompt": """Extract up to {max_entities_from_query} key entities or concepts from the following query related to LEED standards or sustainability.
Focus on specific terms that are likely to be nodes in a knowledge graph.
Return them as a JSON list of strings.
Query: "{query}"
""" # Note: {max_entities_from_query} is not directly used by .format, but good for prompt context
            }
        }
    }
    # Replace with your actual OpenAI API key if testing LLM parsing
    if dummy_config_data["llm"]["api_key"] == "YOUR_OPENAI_API_KEY":
        logger.warning("OpenAI API key not set in dummy config. LLM parsing will likely fail.")
        # Fallback to heuristic for the test if no API key
        # dummy_config_data["retrieval"]["graph_retriever"]["query_parsing_strategy"] = "heuristic"


    with open(test_config_path, 'w') as f:
        yaml.dump(dummy_config_data, f)
    
    nm = None
    gr = None
    try:
        nm = Neo4jManager(config_path=str(test_config_path))
        
        # Ensure some data exists for testing - this should ideally be done by running kg_builder first
        # For isolated test, create minimal data:
        with nm.driver.session(database=nm.database) as s:
            s.run("MERGE (e1:Entity {name: 'LEED v4', type: 'Standard', doc_id:'test_doc', chunk_id:'test_chunk_1'})")
            s.run("MERGE (e2:Entity {name: 'acoustic performance', type: 'Requirement', doc_id:'test_doc', chunk_id:'test_chunk_2'})")
            s.run("MERGE (e1)-[:HAS_REQUIREMENT]->(e2)")
            logger.info("Dummy data ensured in Neo4j for testing GraphRetriever.")

        gr = GraphRetriever(config=dummy_config_data, neo4j_manager=nm)
        
        test_query = "What are the acoustic performance requirements for LEED v4?"
        logger.info(f"\nTesting GraphRetriever with query: \"{test_query}\"")
        results = gr.retrieve(query=test_query)
        
        logger.info(f"Retrieved {len(results.get('kg_paths', []))} KG paths.")
        for i, path_info in enumerate(results.get("kg_paths", [])):
            logger.info(f"  Path {i+1}: {path_info['text']} (Score: {path_info['score']:.4f})")
            # logger.info(f"    Nodes: {path_info['nodes']}")
            # logger.info(f"    Edges: {path_info['edges']}")
        logger.info(f"Metadata: {results.get('metadata')}")

    except Exception as e:
        logger.error(f"Error during GraphRetriever test: {e}", exc_info=True)
    finally:
        if gr and gr.neo4j_manager: # GraphRetriever now holds the manager
            gr.neo4j_manager.close() # Close via the retriever's manager instance
        elif nm: # If retriever init failed but manager was created
             nm.close()
        if test_config_path.exists():
            test_config_path.unlink()
        logger.info("GraphRetriever test finished.")
