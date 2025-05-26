import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver, exceptions

logger = logging.getLogger(__name__)

class Neo4jManager:
    """
    Manages interactions with a Neo4j database, including data import and schema setup.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the Neo4jManager, loading configuration and setting up the driver.
        Args:
            config_path: Path to the configuration YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

        neo4j_config = config.get('neo4j')
        if not neo4j_config:
            logger.error("Neo4j configuration missing from config file.")
            raise ValueError("Neo4j configuration missing.")

        self.uri = neo4j_config.get('uri')
        self.user = neo4j_config.get('user')
        self.password = neo4j_config.get('password')
        self.database = neo4j_config.get('database', 'neo4j') # Default to 'neo4j' database

        if not all([self.uri, self.user, self.password]):
            logger.error("Neo4j URI, user, or password missing from configuration.")
            raise ValueError("Neo4j connection details incomplete.")

        try:
            self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri} (database: {self.database})")
        except exceptions.AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}. Check credentials for user '{self.user}'.")
            raise
        except exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}: {e}. Ensure Neo4j is running.")
            raise
        except Exception as e: # Catch any other driver-related errors
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def _execute_write_query(self, query: str, parameters: Dict[str, Any] = None):
        """Helper to execute a write query within a session and transaction."""
        try:
            with self.driver.session(database=self.database) as session:
                session.write_transaction(lambda tx: tx.run(query, parameters))
        except exceptions.Neo4jError as e:
            logger.error(f"Neo4j write query failed: {query} with params {parameters}. Error: {e}")
            raise # Re-raise after logging for higher-level handling if needed
        except Exception as e:
            logger.error(f"An unexpected error occurred during Neo4j write: {e}")
            raise


    def create_constraints_and_indexes(self):
        """
        Creates constraints and indexes in the Neo4j database for optimal performance
        and data integrity.
        """
        logger.info("Creating constraints and indexes...")
        # Unique constraint on Entity name
        constraint_query = "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        self._execute_write_query(constraint_query)
        logger.info("Ensured unique constraint on Entity(name).")

        # Optional: Index on Entity type
        index_type_query = "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)"
        self._execute_write_query(index_type_query)
        logger.info("Ensured index on Entity(type).")
        
        # Optional: Index on Entity doc_id
        index_doc_id_query = "CREATE INDEX entity_doc_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.doc_id)"
        self._execute_write_query(index_doc_id_query)
        logger.info("Ensured index on Entity(doc_id).")
        logger.info("Constraints and indexes setup complete.")

    def _add_entity_tx(self, tx, name: str, type: str, original_names: List[str], doc_id: str, chunk_id: str):
        """
        Transaction function to create or merge an entity node.
        Properties: name (unique), type, original_names (list), doc_id, chunk_id.
        """
        query = (
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.type = $type, e.original_names = $original_names, e.doc_id = $doc_id, e.chunk_id = $chunk_id, e.created_at = timestamp() "
            "ON MATCH SET e.type = $type, e.original_names = coalesce(e.original_names, []) + [n IN $original_names WHERE NOT n IN e.original_names], e.doc_id = $doc_id, e.chunk_id = $chunk_id, e.updated_at = timestamp() " # Simplified update for original_names
            "RETURN e"
        )
        # Ensure original_names is always a list, even if empty
        parameters = {
            "name": name, 
            "type": type, 
            "original_names": original_names if original_names else [], 
            "doc_id": doc_id, 
            "chunk_id": chunk_id
        }
        tx.run(query, parameters)

    def _add_relation_tx(self, tx, head_name: str, tail_name: str, relation_type: str, doc_id: str, chunk_id: str, source_text_chunk: str):
        """
        Transaction function to create a relationship between two Entity nodes.
        Matches head and tail nodes by name. Relationship type is dynamic.
        Properties: doc_id, chunk_id, source_text_chunk.
        """
        # Ensure relation_type is a valid Neo4j label (alphanumeric, underscores)
        # Basic sanitization, more robust validation might be needed if types are very dynamic
        safe_relation_type = "".join(c if c.isalnum() else '_' for c in relation_type.upper())
        if not safe_relation_type:
            logger.warning(f"Skipping relation due to invalid type after sanitization: {relation_type}")
            return

        query = (
            f"MATCH (h:Entity {{name: $head_name}}) "
            f"MATCH (t:Entity {{name: $tail_name}}) "
            f"MERGE (h)-[r:{safe_relation_type}]->(t) "
            "ON CREATE SET r.doc_id = $doc_id, r.chunk_id = $chunk_id, r.source_text_chunk = $source_text_chunk, r.created_at = timestamp() "
            "ON MATCH SET r.doc_id = $doc_id, r.chunk_id = $chunk_id, r.source_text_chunk = $source_text_chunk, r.updated_at = timestamp() " # Example of updating, could be more nuanced
        )
        parameters = {
            "head_name": head_name,
            "tail_name": tail_name,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source_text_chunk": source_text_chunk
        }
        tx.run(query, parameters)

    def import_data(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """
        Imports entities and relations into Neo4j.
        Args:
            entities: List of entity dictionaries. Expected keys: 
                      'name', 'type', 'original_names', 'doc_id', 'chunk_id'.
            relations: List of relation dictionaries. Expected keys: 
                       'head', 'tail', 'relation', 'doc_id', 'chunk_id', 'source_text_chunk'.
        """
        if not entities and not relations:
            logger.info("No entities or relations provided for import.")
            return

        logger.info(f"Starting data import: {len(entities)} entities, {len(relations)} relations.")

        # Import Entities
        logger.info("Importing entities...")
        with self.driver.session(database=self.database) as session:
            for i, entity_data in enumerate(entities):
                try:
                    # Validate entity_data structure (basic)
                    if not all(k in entity_data for k in ['name', 'type', 'original_names', 'doc_id', 'chunk_id']):
                        logger.warning(f"Skipping entity due to missing keys: {entity_data.get('name', 'N/A')}. Data: {entity_data}")
                        continue
                    if not entity_data['name'] or not entity_data['name'].strip(): # Check for empty name
                         logger.warning(f"Skipping entity due to empty name. Data: {entity_data}")
                         continue

                    session.write_transaction(
                        self._add_entity_tx,
                        name=entity_data['name'],
                        type=entity_data['type'],
                        original_names=entity_data['original_names'],
                        doc_id=entity_data['doc_id'],
                        chunk_id=entity_data['chunk_id']
                    )
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i+1}/{len(entities)} entities.")
                except Exception as e:
                    logger.error(f"Error importing entity '{entity_data.get('name', 'N/A')}': {e}", exc_info=True)
        logger.info("Entity import complete.")

        # Import Relations
        logger.info("Importing relations...")
        with self.driver.session(database=self.database) as session:
            for i, rel_data in enumerate(relations):
                try:
                    # Validate relation_data structure (basic)
                    if not all(k in rel_data for k in ['head', 'tail', 'relation', 'doc_id', 'chunk_id', 'source_text_chunk']):
                        logger.warning(f"Skipping relation due to missing keys: {rel_data}. Head: {rel_data.get('head', 'N/A')}, Tail: {rel_data.get('tail', 'N/A')}")
                        continue
                    if not rel_data['head'] or not rel_data['head'].strip() or \
                       not rel_data['tail'] or not rel_data['tail'].strip() or \
                       not rel_data['relation'] or not rel_data['relation'].strip():
                        logger.warning(f"Skipping relation due to empty head, tail, or relation type. Data: {rel_data}")
                        continue

                    session.write_transaction(
                        self._add_relation_tx,
                        head_name=rel_data['head'],
                        tail_name=rel_data['tail'],
                        relation_type=rel_data['relation'],
                        doc_id=rel_data['doc_id'],
                        chunk_id=rel_data['chunk_id'],
                        source_text_chunk=rel_data['source_text_chunk']
                    )
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i+1}/{len(relations)} relations.")
                except Exception as e:
                    logger.error(f"Error importing relation between '{rel_data.get('head', 'N/A')}' and '{rel_data.get('tail', 'N/A')}': {e}", exc_info=True)
        logger.info("Relation import complete.")
        logger.info("Data import finished.")

if __name__ == '__main__':
    # This basic test assumes Neo4j is running and config.yaml is set up.
    # More comprehensive tests would use mocking or a dedicated test DB.
    logger.info("Running basic Neo4jManager test...")
    try:
        # Ensure config/config.yaml exists and is correctly populated for this test
        # Create a dummy config for testing if not present or use a dedicated test config
        test_config_path = Path("config/config.yaml")
        if not test_config_path.exists():
            logger.warning(f"Test config {test_config_path} not found. Creating a dummy one for the test.")
            test_config_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_config = {
                "neo4j": {
                    "uri": "bolt://localhost:7687", # Adjust if your Neo4j runs elsewhere
                    "user": "neo4j",
                    "password": "password", # PLEASE use a secure password or env variables
                    "database": "neo4j"
                }
            }
            with open(test_config_path, 'w') as f:
                yaml.dump(dummy_config, f)
        
        manager = Neo4jManager(config_path=str(test_config_path))
        
        logger.info("Creating constraints (idempotent operation)...")
        manager.create_constraints_and_indexes()
        
        logger.info("Preparing dummy data for import...")
        dummy_entities = [
            {"name": "Test Entity A", "type": "TEST_TYPE", "original_names": ["Entity A", "A"], "doc_id": "doc1", "chunk_id": "chunk1_0"},
            {"name": "Test Entity B", "type": "TEST_TYPE", "original_names": ["Entity B"], "doc_id": "doc1", "chunk_id": "chunk1_1"},
            {"name": "Another Entity C", "type": "OTHER_TYPE", "original_names": ["C"], "doc_id": "doc2", "chunk_id": "chunk2_0"}
        ]
        dummy_relations = [
            {"head": "Test Entity A", "tail": "Test Entity B", "relation": "TEST_RELATES_TO", "doc_id": "doc1", "chunk_id": "chunk1_0", "source_text_chunk": "Entity A is related to Entity B."},
            {"head": "Test Entity B", "tail": "Another Entity C", "relation": "ANOTHER_RELATION", "doc_id": "doc1", "chunk_id": "chunk1_1", "source_text_chunk": "Entity B has another relation with C."}
        ]
        
        logger.info("Importing dummy data...")
        manager.import_data(dummy_entities, dummy_relations)
        logger.info("Dummy data import finished.")
        
        # Example of how to query (not part of the class, just for testing)
        with manager.driver.session(database=manager.database) as s:
            res = s.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 5")
            logger.info("Querying some imported entities:")
            for record in res:
                logger.info(f"  Entity: {record['name']} (Type: {record['type']})")
            
            res_rel = s.run("MATCH (h:Entity)-[r]->(t:Entity) RETURN h.name AS head, type(r) AS rel_type, t.name AS tail LIMIT 5")
            logger.info("Querying some imported relations:")
            for record in res_rel:
                 logger.info(f"  Relation: {record['head']} -[{record['rel_type']}]-> {record['tail']}")

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except exceptions.ServiceUnavailable:
        logger.error("Neo4j is not available. Ensure the database is running and accessible.")
    except Exception as e:
        logger.error(f"An error occurred during the Neo4jManager test: {e}", exc_info=True)
    finally:
        if 'manager' in locals() and manager.driver:
            manager.close()
        logger.info("Neo4jManager test finished.")

# To run this test:
# 1. Ensure config/config.yaml has your Neo4j credentials.
# 2. Ensure Neo4j Desktop/Server is running.
# 3. Run `python -m kg_construction.neo4j_manager` from the project root.
#    (You might need to adjust PYTHONPATH or run as `python kg_construction/neo4j_manager.py`)Okay, I have created `kg_construction/neo4j_manager.py` with the `Neo4jManager` class.
The class includes:
-   `__init__` for loading config and initializing the Neo4j driver.
-   `close` for closing the driver.
-   `_execute_write_query` helper.
-   `create_constraints_and_indexes` for setting up unique constraints on `Entity(name)` and indexes on `Entity(type)` and `Entity(doc_id)`.
-   `_add_entity_tx` and `_add_relation_tx` as internal transaction functions for creating/merging nodes and relationships.
-   `import_data` as the main public method to import lists of entities and relations. It handles basic validation of input data.
-   A basic `if __name__ == '__main__':` block for testing (requires `config/config.yaml` to be set up).

Next, I need to address the **`doc_id`/`chunk_id` for entities in `KnowledgeGraphBuilder`**.
As noted in the plan, `final_canonical_entities` currently lacks `doc_id` and `chunk_id`, which are needed by `Neo4jManager.add_entity`.

I will modify `kg_construction/kg_builder.py` to include this. The strategy is:
1.  When collecting `all_extracted_triples`, each triple already contains `doc_id` and `chunk_id`.
2.  The `preliminary_entities` set will store `(name, type, doc_id, chunk_id)` tuples. For simplicity, I'll take the `doc_id`/`chunk_id` from the *first occurrence* of that (name, type) pair.
3.  The clustering will be done on `name` (embeddings), but we'll keep track of the associated `type`, `doc_id`, and `chunk_id`.
4.  The `_determine_canonical_representation` method will be adapted to also return a representative `doc_id` and `chunk_id` for the cluster (e.g., from the first entity in the cluster).
5.  The `final_canonical_entities` list will then be `[{"name": ..., "type": ..., "original_names": [...], "doc_id": ..., "chunk_id": ...}, ...]`.

After this modification to `KnowledgeGraphBuilder`, I will update its `__init__` and `build_knowledge_graph` methods to use the `Neo4jManager`.
