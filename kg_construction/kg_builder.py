import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any
import json
import yaml
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from kg_construction.kg_extractor import KnowledgeGraphExtractor
from kg_construction.neo4j_manager import Neo4jManager # Import Neo4jManager
from slugify import slugify

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Orchestrates the construction of a knowledge graph from processed text chunks.
    This involves:
    1. Extracting raw triples from text chunks using KnowledgeGraphExtractor.
    2. Performing entity deduplication using embeddings and clustering.
    3. Producing lists of canonical entities and relations.
    4. Importing the canonical data into Neo4j.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the knowledge graph builder with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kg_extractor = KnowledgeGraphExtractor(config_path)
        self.neo4j_manager = Neo4jManager(config_path) # Initialize Neo4jManager
        
        embedding_model_name = self.config.get('kg_builder', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Successfully initialized SentenceTransformer model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{embedding_model_name}': {e}")
            raise
            
        paths_config = self.config.get('paths', {})
        processed_dir = Path(paths_config.get('processed_dir', 'data/processed'))
        jsonl_output_dir_str = paths_config.get('jsonl_output_dir', str(processed_dir / 'jsonl_output'))
        self.jsonl_input_dir = Path(jsonl_output_dir_str)
        
        if not self.jsonl_input_dir.exists():
            logger.warning(f"JSONL input directory not found: {self.jsonl_input_dir}. KGBuilder might not find files to process.")

        clustering_config = self.config.get('kg_builder', {}).get('clustering', {})
        self.clustering_distance_threshold = clustering_config.get('distance_threshold', 0.75) 
        self.min_cluster_size = clustering_config.get('min_cluster_size_for_noise', 1)


    def _determine_canonical_representation(self, entity_cluster: List[Tuple[str, str, str, str]]) -> Tuple[str, str, List[str], str, str]:
        """
        Determines the canonical name, type, original names, and representative doc_id, chunk_id for a cluster.
        Args:
            entity_cluster: List of (name, type, doc_id, chunk_id) tuples.
        Returns:
            Tuple: (canonical_name, canonical_type, list_of_original_names, representative_doc_id, representative_chunk_id).
        """
        if not entity_cluster:
            return "Unknown", "THING", [], "unknown_doc", "unknown_chunk"

        # Determine canonical name (most frequent, then shortest)
        name_counts = Counter(name for name, type_, doc_id, chunk_id in entity_cluster if name.strip())
        if not name_counts: # All names were empty
            canonical_name = "Unknown_Entity_Cluster"
            # Attempt to get source from first item if cluster is not empty
            representative_doc_id = entity_cluster[0][2] if entity_cluster else "unknown_doc"
            representative_chunk_id = entity_cluster[0][3] if entity_cluster else "unknown_chunk"
        else:
            max_freq = max(name_counts.values())
            candidate_names = [name for name, freq in name_counts.items() if freq == max_freq]
            canonical_name = min(candidate_names, key=len)
            # Get doc_id and chunk_id from the first entity that matches the canonical name
            first_matching_entity = next((e for e in entity_cluster if e[0] == canonical_name), entity_cluster[0])
            representative_doc_id = first_matching_entity[2]
            representative_chunk_id = first_matching_entity[3]


        # Determine canonical type (most frequent for the canonical_name, then overall most frequent)
        types_for_canonical_name = [type_ for name, type_, _, _ in entity_cluster if name == canonical_name and type_.strip()]
        if types_for_canonical_name:
            type_counts_for_canonical_name = Counter(types_for_canonical_name)
            canonical_type = type_counts_for_canonical_name.most_common(1)[0][0]
        else: 
            overall_type_counts = Counter(type_ for name, type_, _, _ in entity_cluster if type_.strip())
            canonical_type = overall_type_counts.most_common(1)[0][0] if overall_type_counts else "THING"
            
        original_names = sorted(list(set(name for name, _, _, _ in entity_cluster if name.strip())))
        return canonical_name, canonical_type, original_names, representative_doc_id, representative_chunk_id

    def build_knowledge_graph(self) -> Dict[str, List[Any]]:
        """
        Builds the knowledge graph, performs deduplication, and imports data into Neo4j.
        """
        try:
            all_extracted_triples = []
            chunk_idx_counter = 0 

            if not self.jsonl_input_dir.exists() or not list(self.jsonl_input_dir.glob("*.jsonl")):
                logger.warning(f"No JSONL files found in {self.jsonl_input_dir}. Skipping KG building.")
                return {"entities": [], "relations": []}

            for jsonl_file_path in self.jsonl_input_dir.glob("*.jsonl"):
                logger.info(f"Processing JSONL file: {jsonl_file_path}")
                with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            chunk_data = json.loads(line.strip())
                            current_global_chunk_idx = chunk_idx_counter
                            chunk_idx_counter += 1
                            triples_from_chunk = self.kg_extractor.process_chunk(chunk_data, current_global_chunk_idx)
                            all_extracted_triples.extend(triples_from_chunk)
                        except json.JSONDecodeError as e:
                            logger.error(f"Skipping invalid JSON line {line_num+1} in {jsonl_file_path}: {e}")
                        except Exception as e: 
                            logger.error(f"Error processing chunk line {line_num+1} in {jsonl_file_path}: {e}", exc_info=True)
            
            logger.info(f"Total raw triples extracted: {len(all_extracted_triples)}")
            if not all_extracted_triples:
                return {"entities": [], "relations": []}

            # Store first source for each (name, type) pair
            # preliminary_entities_sources: Dict[Tuple[name, type], Tuple[doc_id, chunk_id]]
            preliminary_entities_sources: Dict[Tuple[str, str], Tuple[str, str]] = {} 
            for triple in all_extracted_triples:
                head_key = (triple['head_name'], triple['head_type'])
                tail_key = (triple['tail_name'], triple['tail_type'])
                source_info = (triple['doc_id'], triple['chunk_id'])

                if triple['head_name'].strip() and triple['head_type'].strip():
                    if head_key not in preliminary_entities_sources:
                        preliminary_entities_sources[head_key] = source_info
                if triple['tail_name'].strip() and triple['tail_type'].strip():
                    if tail_key not in preliminary_entities_sources:
                        preliminary_entities_sources[tail_key] = source_info
            
            # unique_entity_list_with_source: List[Tuple[name, type, doc_id, chunk_id]]
            unique_entity_list_with_source = sorted([
                (name, type_, doc_id, chunk_id) 
                for (name, type_), (doc_id, chunk_id) in preliminary_entities_sources.items()
                if name.strip() and type_.strip() 
            ])
            
            entity_names = [name for name, type_, doc_id, chunk_id in unique_entity_list_with_source]
            logger.info(f"Found {len(unique_entity_list_with_source)} unique, non-empty (name, type) pairs with source info.")

            if not entity_names:
                 logger.info("No valid entities to deduplicate after filtering.")
                 return {"entities": [], "relations": []}

            logger.info("Generating embeddings for entity names...")
            embeddings = self.embedding_model.encode(entity_names, show_progress_bar=True)
            
            logger.info(f"Clustering {len(entity_names)} entities with distance threshold: {self.clustering_distance_threshold}...")
            clustering_model = AgglomerativeClustering(
                n_clusters=None, affinity='cosine', linkage='average', 
                distance_threshold=self.clustering_distance_threshold
            )
            cluster_labels = clustering_model.fit_predict(embeddings)
            
            num_clusters = len(set(cluster_labels))
            logger.info(f"Found {num_clusters} clusters.")

            # Map original (name, type) to canonical (name, type)
            entity_to_canonical_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
            # Store details for canonical entities: (canon_name, canon_type) -> {details}
            final_canonical_entities_details: Dict[Tuple[str, str], Dict[str, Any]] = {}
            
            # Group entities by cluster label
            # clusters_data: Dict[cluster_label, List[Tuple[name, type, doc_id, chunk_id]]]
            clusters_data: Dict[int, List[Tuple[str, str, str, str]]] = {} 
            for i, (name, type_, doc_id, chunk_id) in enumerate(unique_entity_list_with_source):
                label = cluster_labels[i]
                if label not in clusters_data:
                    clusters_data[label] = []
                clusters_data[label].append((name, type_, doc_id, chunk_id))


            for cluster_id, items_in_cluster in clusters_data.items():
                # Determine canonical representation for this cluster
                canon_name, canon_type, original_names_in_cluster, rep_doc_id, rep_chunk_id = \
                    self._determine_canonical_representation(items_in_cluster)

                canonical_key = (canon_name, canon_type)
                if canonical_key not in final_canonical_entities_details:
                    final_canonical_entities_details[canonical_key] = {
                        "original_names": set(), 
                        "doc_id": rep_doc_id, 
                        "chunk_id": rep_chunk_id
                    }
                final_canonical_entities_details[canonical_key]["original_names"].update(original_names_in_cluster)
                
                # Map all original (name, type) pairs in this cluster to the canonical (name, type)
                for original_name, original_type, _, _ in items_in_cluster: # Original doc/chunk IDs are not needed for the map key
                    entity_to_canonical_map[(original_name, original_type)] = (canon_name, canon_type)
            
            final_canonical_entities_for_neo4j = [
                {"name": cn, "type": ct, 
                 "original_names": sorted(list(details["original_names"])),
                 "doc_id": details["doc_id"], 
                 "chunk_id": details["chunk_id"]}
                for (cn, ct), details in final_canonical_entities_details.items()
            ]
            logger.info(f"Number of canonical entities: {len(final_canonical_entities_for_neo4j)}")

            final_relations_for_neo4j = []
            for triple in all_extracted_triples:
                original_head_key = (triple['head_name'], triple['head_type'])
                original_tail_key = (triple['tail_name'], triple['tail_type'])
                
                # Use original as fallback if not found in map (should be rare if all entities processed)
                canonical_head_name, _ = entity_to_canonical_map.get(original_head_key, original_head_key)
                canonical_tail_name, _ = entity_to_canonical_map.get(original_tail_key, original_tail_key)
                
                if not canonical_head_name.strip() or not canonical_tail_name.strip():
                    logger.warning(f"Skipping relation due to empty canonical name: head='{canonical_head_name}', tail='{canonical_tail_name}' from triple {triple}")
                    continue

                final_relations_for_neo4j.append({
                    "head": canonical_head_name, 
                    "relation": triple['relation_type'],
                    "tail": canonical_tail_name, 
                    "doc_id": triple['doc_id'],
                    "chunk_id": triple['chunk_id'],
                    "source_text_chunk": triple.get('source_text_chunk', '') 
                })

            logger.info(f"Total relations after canonicalization: {len(final_relations_for_neo4j)}")
            
            # Neo4j Import
            logger.info("Preparing to import data into Neo4j...")
            self.neo4j_manager.create_constraints_and_indexes()
            self.neo4j_manager.import_data(final_canonical_entities_for_neo4j, final_relations_for_neo4j)
            logger.info("Data import into Neo4j completed.")
            
            # Return the data used for Neo4j import for potential further use/logging
            return {"entities": final_canonical_entities_for_neo4j, "relations": final_relations_for_neo4j}

        except Exception as e: # Catch-all for unexpected issues during the build process
            logger.error(f"A critical error occurred in build_knowledge_graph: {e}", exc_info=True)
            # Ensure Neo4j connection is closed even if build fails mid-way
            # Check if neo4j_manager exists and has a driver, as error could be in __init__
            if hasattr(self, 'neo4j_manager') and self.neo4j_manager and self.neo4j_manager.driver:
                 self.neo4j_manager.close()
                 logger.info("Neo4j connection closed due to an error in build_knowledge_graph.")
            return {"entities": [], "relations": []} # Return empty on critical failure
        # The finally block was removed from here to be placed in the caller (main) or if builder becomes a context manager


def main():
    logger.info("Starting Knowledge Graph Builder main (for testing)...")
    config_path = "config/config.yaml" 
    
    builder = None # Initialize builder to None for finally block
    try:
        builder = KnowledgeGraphBuilder(config_path=config_path)
        
        if not builder.jsonl_input_dir.exists() or not any(builder.jsonl_input_dir.glob("*.jsonl")):
            logger.info(f"Creating dummy JSONL data for testing in {builder.jsonl_input_dir}")
            builder.jsonl_input_dir.mkdir(parents=True, exist_ok=True)
            dummy_file_path = builder.jsonl_input_dir / "dummy_data.jsonl"
            with open(dummy_file_path, 'w', encoding='utf-8') as f:
                json.dump({"doc_id": "doc1", "section": "Intro", "text": "LEED v4 requires commissioning. LEED version 4 also mandates energy audits.", "page": 1}, f); f.write('\n')
                json.dump({"doc_id": "doc1", "section": "Details", "text": "The commissioning process is vital. Energy audits are part of EA credit.", "page": 2}, f); f.write('\n')
                json.dump({"doc_id": "doc2", "section": "Advanced", "text": "Advanced commissioning is good. LEED v4.1 is newer.", "page": 1}, f); f.write('\n')
                json.dump({"doc_id": "doc2", "section": "Metrics", "text": "Water efficiency is a key metric. Water conservation helps.", "page": 5}, f); f.write('\n')
                json.dump({"doc_id": "doc3", "section": "Lighting", "text": "Daylight harvesting improves energy performance. Light pollution should be minimized.", "page": 10}, f); f.write('\n')

        kg_data = builder.build_knowledge_graph() 
        
        logger.info("Knowledge Graph Building and Neo4j Import Complete (Testing)")
        if kg_data: 
            logger.info(f"Total canonical entities produced: {len(kg_data.get('entities', []))}")
            logger.info(f"Total relations produced: {len(kg_data.get('relations', []))}")

            output_file_path = Path("kg_builder_output_final.json") 
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, indent=2, ensure_ascii=False)
            logger.info(f"KG builder final output (entities/relations) saved to {output_file_path}")

            if kg_data.get('entities'):
                logger.info("\nSample Canonical Entities (up to 10):")
                for entity in kg_data['entities'][:10]:
                    logger.info(f"  Name: {entity['name']}, Type: {entity['type']}, Doc: {entity.get('doc_id', 'N/A')}, Chunk: {entity.get('chunk_id', 'N/A')}, Originals: {entity.get('original_names', [])}")
            
            if kg_data.get('relations'):
                logger.info("\nSample Relations (up to 10):")
                for rel in kg_data['relations'][:10]:
                    logger.info(f"  {rel['head']} -[{rel['relation']}]-> {rel['tail']} (Chunk: {rel['chunk_id']})")
        else:
            logger.warning("kg_data was not returned from build_knowledge_graph, likely due to an earlier error or empty input.")

    except FileNotFoundError: 
        logger.error(f"Configuration file not found at {config_path}. Ensure the path is correct.")
    except Exception as e: 
        logger.error(f"An error occurred during KG building main execution: {e}", exc_info=True)
    finally:
        if builder and hasattr(builder, 'neo4j_manager') and builder.neo4j_manager and builder.neo4j_manager.driver:
            builder.neo4j_manager.close() 
            logger.info("Neo4j connection closed from main's finally block.")


if __name__ == '__main__':
    main()