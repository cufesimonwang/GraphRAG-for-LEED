import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import yaml
from ..content_extractor import ContentExtractor
from .kg_extractor import KnowledgeGraphExtractor
from .graph_visualizer import GraphVisualizer
from .graph_manager import GraphManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Orchestrates the construction of a knowledge graph from documentation.
    
    This class coordinates the entire pipeline of:
    1. Text extraction from various file formats
    2. Entity and relation extraction using LLMs
    3. Graph construction and visualization
    4. Output generation in multiple formats
    
    The builder supports diagnostic mode via return_triples option, which can be useful for:
    - Debugging extraction issues
    - Tuning LLM prompts
    - Analyzing raw triple extraction results
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the knowledge graph builder with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.content_extractor = ContentExtractor(config_path)
        self.kg_extractor = KnowledgeGraphExtractor(config_path)
        self.graph_manager = GraphManager(config_path)
        self.graph_visualizer = GraphVisualizer(config_path)
        
        # Set up paths
        self.data_dir = Path(self.config['paths']['raw_pdf_dir'])
        self.output_dir = Path(self.config['paths']['faiss_index']).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get output prefix from config
        self.output_prefix = self.config.get('output', {}).get('graph_prefix', 'knowledge_graph')
        
        # Get mode and diagnostic settings
        self.mode = self.config.get('mode', 'graphrag')  # Options: 'rag', 'graphrag', 'hybrid'
        self.return_triples = self.config.get('output', {}).get('return_triples', False)
    
    def process_file(self, file_path: Path) -> Dict:
        """
        Process a single file and extract knowledge graph elements.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing:
            - entities: List of extracted entities
            - relations: List of extracted relations
            - chunks: List of text chunks (for RAG/hybrid mode)
            - metadata: File metadata
        """
        logger.info(f"Processing file: {file_path}")
        
        # Process file using kg extractor
        result = self.kg_extractor.process_file(file_path)
        
        logger.info(f"Extracted {len(result['entities'])} entities and {len(result['relations'])} relations")
        if self.mode in ['rag', 'hybrid']:
            logger.info(f"Generated {len(result['chunks'])} text chunks")
        
        return result
    
    def build_knowledge_graph(self) -> Dict:
        """
        Build knowledge graph from all files in the data directory.
        
        Returns:
            Dict containing:
            - entities: List of unique entities
            - relations: List of relations
            - chunks: List of text chunks (for RAG/hybrid mode)
            - triples: List of raw triples (if return_triples is True)
        """
        all_entities = {}  # Use dict for deduplication
        all_relations = []
        all_chunks = []
        all_triples = [] if self.return_triples else None
        
        # Find all supported files
        supported_files = [
            f for f in self.data_dir.glob('**/*')
            if f.suffix in self.content_extractor.supported_extensions
        ]
        logger.info(f"Found {len(supported_files)} supported files in {self.data_dir}")
        
        # Process all files
        for file_path in supported_files:
            result = self.process_file(file_path)
            
            # Deduplicate entities
            for entity in result['entities']:
                all_entities[entity['id']] = entity
            
            # Add relations
            all_relations.extend(result['relations'])
            
            # Add chunks for RAG/hybrid mode
            if self.mode in ['rag', 'hybrid']:
                all_chunks.extend(result['chunks'])
            
            # Add triples if in diagnostic mode
            if self.return_triples and 'triples' in result:
                all_triples.extend(result['triples'])
        
        # Build the graph
        logger.info("Building knowledge graph...")
        self.graph_manager.build_graph(list(all_entities.values()), all_relations)
        
        # Save results
        self._save_graph_outputs()
        self._save_graph_stats()
        
        # Save chunks for RAG/hybrid mode
        if self.mode in ['rag', 'hybrid']:
            self._save_chunks(all_chunks)
        
        # Save raw triples if in diagnostic mode
        if self.return_triples:
            self._save_raw_triples(all_triples)
        
        logger.info(f"Knowledge graph built successfully:")
        logger.info(f"- {len(all_entities)} unique entities")
        logger.info(f"- {len(all_relations)} relations")
        if self.mode in ['rag', 'hybrid']:
            logger.info(f"- {len(all_chunks)} text chunks")
        if self.return_triples:
            logger.info(f"- {len(all_triples)} raw triples")
        
        return {
            'entities': list(all_entities.values()),
            'relations': all_relations,
            'chunks': all_chunks,
            'triples': all_triples
        }
    
    def _save_graph_outputs(self):
        """Save the knowledge graph in various formats."""
        # Get output formats from config or use defaults
        output_formats = self.config.get('output', {}).get('graph_formats', 
            ['json', 'html', 'png', 'graphml'])
        
        # Save in each format
        for fmt in output_formats:
            output_path = self.output_dir / f'{self.output_prefix}.{fmt}'
            self.graph_visualizer.save_graph(self.graph_manager.graph, output_path, format=fmt)
            logger.info(f"Saved graph in {fmt} format to {output_path}")
    
    def _save_graph_stats(self):
        """Save graph statistics to a JSON file."""
        stats = self.graph_visualizer.get_graph_stats(self.graph_manager.graph)
        stats_path = self.output_dir / f'{self.output_prefix}_stats.json'
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved graph statistics to {stats_path}")
    
    def _save_chunks(self, chunks: List[str]):
        """Save text chunks to a JSONL file for RAG/hybrid mode."""
        chunks_path = self.output_dir / f'{self.output_prefix}_chunks.jsonl'
        
        with open(chunks_path, 'w') as f:
            for chunk in chunks:
                f.write(json.dumps({'text': chunk}) + '\n')
        logger.info(f"Saved {len(chunks)} text chunks to {chunks_path}")
    
    def _save_raw_triples(self, triples: List[Dict]):
        """Save raw triples to a JSON file for diagnostic purposes."""
        triples_path = self.output_dir / f'{self.output_prefix}_raw_triples.json'
        
        with open(triples_path, 'w') as f:
            json.dump(triples, f, indent=2)
        logger.info(f"Saved raw triples to {triples_path}")

def main():
    """
    Main entry point for building the knowledge graph.
    
    This function:
    1. Initializes the knowledge graph builder
    2. Builds the knowledge graph from all supported files
    3. Logs a summary of the results
    """
    # Initialize the builder
    builder = KnowledgeGraphBuilder()
    
    # Build the knowledge graph
    result = builder.build_knowledge_graph()
    
    # Log summary
    logger.info("Knowledge Graph Building Complete")
    logger.info(f"Total unique entities: {len(result['entities'])}")
    logger.info(f"Total relations: {len(result['relations'])}")
    if builder.mode in ['rag', 'hybrid']:
        logger.info(f"Total text chunks: {len(result['chunks'])}")
    if builder.return_triples:
        logger.info(f"Total raw triples: {len(result['triples'])}")

if __name__ == '__main__':
    main() 