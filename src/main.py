#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .content_extractor import ContentExtractor
from .kg.kg_extractor import KnowledgeGraphExtractor
from .kg.graph_visualizer import GraphVisualizer

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("graphrag.log")
        ]
    )

def process_files(
    input_path: Path,
    config_path: str,
    mode: str,
    output_dir: Optional[Path] = None
) -> None:
    """Process files in the input directory or a single file."""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    content_extractor = ContentExtractor(config_path)
    kg_extractor = KnowledgeGraphExtractor(config_path)
    graph_visualizer = GraphVisualizer(config_path)
    
    # Handle single file or directory
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = [f for f in input_path.glob("**/*") 
                          if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.txt']]
    
    # Process each file
    for file_path in files_to_process:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Extract content
            content = content_extractor.process_file(file_path, mode=mode)
            
            # Extract knowledge graph
            if mode in ["graphrag", "hybrid"]:
                kg_data = kg_extractor.process(content['text'])
                
                # Save graph
                if output_dir:
                    output_path = output_dir / f"{file_path.stem}_graph.json"
                    graph_visualizer.save_graph(kg_data, output_path)
                    logger.info(f"Saved graph to: {output_path}")
                
                # Print graph statistics
                stats = graph_visualizer.get_graph_stats(kg_data)
                logger.info(f"Graph statistics for {file_path.name}:")
                logger.info(f"  Nodes: {stats['nodes']}")
                logger.info(f"  Edges: {stats['edges']}")
                logger.info(f"  Node types: {stats['node_types']}")
                logger.info(f"  Edge types: {stats['edge_types']}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

def main():
    """Main entry point for the GraphRAG pipeline."""
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "graphrag", "hybrid"],
        default="hybrid",
        help="Processing mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory containing files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for processed files (optional)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Convert paths to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    # Validate input path
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    process_files(input_path, args.config, args.mode, output_dir)

if __name__ == "__main__":
    main() 