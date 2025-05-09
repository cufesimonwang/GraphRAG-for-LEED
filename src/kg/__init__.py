"""
Knowledge Graph (KG) package for document processing and graph construction.

This package provides tools for:
1. Extracting knowledge from various document formats
2. Building and managing knowledge graphs
3. Visualizing and analyzing graph structures

Main Components:
- KnowledgeGraphBuilder: Main orchestrator for building knowledge graphs
- GraphConstructor: Handles graph construction and visualization
- KnowledgeGraphExtractor: Extracts entities and relations from text
- TextExtractor: Extracts text from various file formats
"""

from .kg_builder import KnowledgeGraphBuilder
from .graph_manager import GraphConstructor
from .kg_extractor import KnowledgeGraphExtractor
from ..content_extractor import ContentExtractor
from .graph_manager import GraphManager
from .graph_visualizer import GraphVisualizer

__version__ = "0.1.0"
__author__ = "GraphRAG Team"

# Export main classes for easy importing
__all__ = [
    "KnowledgeGraphBuilder",
    "GraphConstructor",
    "KnowledgeGraphExtractor",
    "ContentExtractor",
    "GraphManager",
    "GraphVisualizer"
]
