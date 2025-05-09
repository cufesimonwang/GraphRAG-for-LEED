import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Handles graph visualization and styling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the graph visualizer."""
        self.config_path = config_path
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load graph visualization settings
        self.graph_config = self.config.get('graph', {})
        self.entity_colors = self.graph_config.get('entity_colors', {})
        self.layout_seed = self.graph_config.get('layout_seed', 42)
    
    def save_graph(self, graph: nx.DiGraph, output_path: Path, format: str = 'json') -> None:
        """
        Save the knowledge graph in various formats.
        
        Args:
            graph: NetworkX DiGraph to visualize
            output_path: Path to save the visualization
            format: Output format ('json', 'html', 'png', 'graphml')
        """
        if format == 'json':
            data = nx.node_link_data(graph)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == 'html':
            net = Network(
                height='750px',
                width='100%',
                bgcolor='#ffffff',
                font_color='black'
            )
            
            # Add nodes
            for node_id in graph.nodes():
                net.add_node(
                    node_id,
                    label=graph.nodes[node_id]['label'],
                    color=graph.nodes[node_id]['color'],
                    title=f"Type: {graph.nodes[node_id]['type']}"
                )
            
            # Add edges
            for source, target, data in graph.edges(data=True):
                net.add_edge(source, target, title=data['type'])
            
            net.save_graph(str(output_path))
            
        elif format == 'png':
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=self.layout_seed)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                graph, pos,
                node_color=[graph.nodes[node]['color'] for node in graph.nodes()],
                node_size=2000
            )
            
            # Draw edges
            nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)
            
            # Add labels
            nx.draw_networkx_labels(
                graph, pos,
                labels={node: graph.nodes[node]['label'] for node in graph.nodes()}
            )
            
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
        elif format == 'graphml':
            nx.write_graphml(graph, output_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved graph in {format} format to {output_path}")
    
    def get_graph_stats(self, graph: nx.DiGraph) -> Dict:
        """
        Get statistics about the knowledge graph.
        
        Args:
            graph: NetworkX DiGraph to analyze
            
        Returns:
            Dictionary containing graph statistics
        """
        if not graph:
            return {
                'nodes': 0,
                'edges': 0,
                'node_types': [],
                'edge_types': []
            }
        
        # Get unique node and edge types
        node_types = set(nx.get_node_attributes(graph, 'type').values())
        edge_types = set(nx.get_edge_attributes(graph, 'type').values())
        
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'node_types': list(node_types),
            'edge_types': list(edge_types),
            'density': nx.density(graph),
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'connected_components': nx.number_weakly_connected_components(graph)
        } 