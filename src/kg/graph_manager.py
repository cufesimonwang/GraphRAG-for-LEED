import networkx as nx
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
import matplotlib.pyplot as plt
from pyvis.network import Network
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages the knowledge graph construction and operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the graph manager."""
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()
        self.graph = nx.DiGraph()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _load_prompts(self) -> Dict:
        """Load prompts from prompts.yaml file."""
        prompts_path = self.config.get("paths", {}).get("prompts_file", "config/prompts.yaml")
        try:
            with open(prompts_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Error loading prompts from {prompts_path}: {e}")
            return self._get_default_prompts()

    def _get_default_prompts(self) -> Dict:
        """Return default prompts if prompts.yaml is not found."""
        return getattr(self, 'prompts', {}).get('defaults', {})

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_config = self.config.get("logging", {})
        logging.basicConfig(
            level=log_config.get("level", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_config.get("log_file", "graph_manager.log")
        )
    
    def _get_prompt(self, prompt_type: str, prompt_name: str, **kwargs) -> str:
        """Get a prompt from the prompts configuration."""
        try:
            prompt = self.prompts[prompt_type][prompt_name]
            return prompt.format(**kwargs)
        except KeyError:
            logger.warning(f"Prompt {prompt_type}.{prompt_name} not found, using default")
            return self._get_default_prompts().get(prompt_type, "")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return self._get_default_prompts().get(prompt_type, "")
    
    def build_graph(self, entities: List[Dict], relations: List[Dict]) -> None:
        """Build the knowledge graph from entities and relations."""
        try:
            # Add entities as nodes
            for entity in entities:
                self.graph.add_node(
                    entity['id'],
                    text=entity['text'],
                    type=entity['type'],
                    source_text=entity.get('source_text', '')
                )
            
            # Add relations as edges
            for relation in relations:
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    type=relation['type'],
                    source_text=relation.get('source_text', '')
                )
            
            logger.info(f"Graph built with {len(entities)} nodes and {len(relations)} edges")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def save_graph(self, output_dir: str = None) -> None:
        """Save the graph in various formats."""
        try:
            if output_dir is None:
                output_dir = self.config['paths']['output_dir']
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            graph_data = {
                'entities': [
                    {
                        'id': node,
                        'text': data['text'],
                        'type': data['type'],
                        'source_text': data.get('source_text', '')
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                'relations': [
                    {
                        'source': u,
                        'target': v,
                        'type': data['type'],
                        'source_text': data.get('source_text', '')
                    }
                    for u, v, data in self.graph.edges(data=True)
                ]
            }
            
            json_path = output_dir / f"{self.config['output']['graph_prefix']}.json"
            with open(json_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            # Save as GraphML
            graphml_path = output_dir / f"{self.config['output']['graph_prefix']}.graphml"
            nx.write_graphml(self.graph, graphml_path)
            
            # Save entities and relations separately
            entities_path = output_dir / f"{self.config['output']['graph_prefix']}_entities.json"
            relations_path = output_dir / f"{self.config['output']['graph_prefix']}_relations.json"
            
            with open(entities_path, 'w') as f:
                json.dump(graph_data['entities'], f, indent=2)
            
            with open(relations_path, 'w') as f:
                json.dump(graph_data['relations'], f, indent=2)
            
            logger.info(f"Graph saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            raise
    
    def load_graph(self, input_dir: str = None) -> None:
        """Load the graph from saved files."""
        try:
            if input_dir is None:
                input_dir = self.config['paths']['output_dir']
            input_dir = Path(input_dir)
            
            # Load from JSON
            json_path = input_dir / f"{self.config['output']['graph_prefix']}.json"
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            self.graph = nx.DiGraph()
            
            # Add entities
            for entity in graph_data['entities']:
                self.graph.add_node(
                    entity['id'],
                    text=entity['text'],
                    type=entity['type'],
                    source_text=entity.get('source_text', '')
                )
            
            # Add relations
            for relation in graph_data['relations']:
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    type=relation['type'],
                    source_text=relation.get('source_text', '')
                )
            
            logger.info(f"Graph loaded from {input_dir}")
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity information by ID."""
        try:
            if entity_id in self.graph:
                node_data = self.graph.nodes[entity_id]
                return {
                    'id': entity_id,
                    'text': node_data['text'],
                    'type': node_data['type'],
                    'source_text': node_data.get('source_text', '')
                }
            return None
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            return None
    
    def get_entity_relations(self, entity_id: str) -> List[Dict]:
        """Get all relations for an entity."""
        try:
            if entity_id not in self.graph:
                return []
            
            relations = []
            
            # Get incoming relations
            for pred in self.graph.predecessors(entity_id):
                edge_data = self.graph.edges[pred, entity_id]
                relations.append({
                    'source': pred,
                    'target': entity_id,
                    'type': edge_data['type'],
                    'direction': 'incoming',
                    'source_text': edge_data.get('source_text', '')
                })
            
            # Get outgoing relations
            for succ in self.graph.successors(entity_id):
                edge_data = self.graph.edges[entity_id, succ]
                relations.append({
                    'source': entity_id,
                    'target': succ,
                    'type': edge_data['type'],
                    'direction': 'outgoing',
                    'source_text': edge_data.get('source_text', '')
                })
            
            return relations
            
        except Exception as e:
            logger.error(f"Error getting entity relations: {e}")
            return []
    
    def get_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> nx.DiGraph:
        """Get subgraph around specified entities."""
        try:
            # Get nodes within max_hops
            nodes = set(entity_ids)
            for _ in range(max_hops):
                new_nodes = set()
                for node in nodes:
                    new_nodes.update(self.graph.predecessors(node))
                    new_nodes.update(self.graph.successors(node))
                nodes.update(new_nodes)
            
            return self.graph.subgraph(nodes)
            
        except Exception as e:
            logger.error(f"Error getting subgraph: {e}")
            return nx.DiGraph()
    
    def analyze_graph(self) -> Dict:
        """Analyze the graph structure and content."""
        try:
            analysis = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'node_types': {},
                'edge_types': {},
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                'is_directed': self.graph.is_directed(),
                'is_dag': nx.is_directed_acyclic_graph(self.graph)
            }
            
            # Count node types
            for _, data in self.graph.nodes(data=True):
                node_type = data['type']
                analysis['node_types'][node_type] = analysis['node_types'].get(node_type, 0) + 1
            
            # Count edge types
            for _, _, data in self.graph.edges(data=True):
                edge_type = data['type']
                analysis['edge_types'][edge_type] = analysis['edge_types'].get(edge_type, 0) + 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing graph: {e}")
            return {}

class GraphConstructor:
    # Default color mapping for entity types
    DEFAULT_ENTITY_COLORS = {
        'CREDIT': '#FFB6C1',  # Light pink
        'PREREQUISITE': '#98FB98',  # Light green
        'POINT': '#87CEEB',  # Sky blue
        'CATEGORY': '#DDA0DD',  # Plum
        'CONCEPT': '#F0E68C'  # Khaki
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the graph constructor.
        
        Args:
            config: Optional configuration dictionary containing graph settings
        """
        self.graph = nx.DiGraph()
        
        # Get entity colors from config or use defaults
        self.entity_colors = (
            config.get('graph', {}).get('entity_colors', {})
            if config
            else self.DEFAULT_ENTITY_COLORS
        )
        
        # Get layout seed from config or use default
        self.layout_seed = config.get('graph', {}).get('layout_seed', 42) if config else 42
    
    def build_graph(self, entities: List[Dict], relations: List[Dict]):
        """
        Build a directed graph from entities and relations.
        
        Args:
            entities: List of entity dictionaries with 'id', 'text', and 'type' fields
            relations: List of relation dictionaries with 'source', 'target', and 'type' fields
        """
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes (entities)
        for entity in entities:
            self.graph.add_node(
                entity['id'],
                text=entity['text'],
                type=entity['type'],
                color=self.entity_colors.get(entity['type'], '#F0E68C')
            )
        
        # Add edges (relations)
        for relation in relations:
            self.graph.add_edge(
                relation['source'],
                relation['target'],
                type=relation['type']
            )
        
        logger.info(f"Built graph with {len(entities)} nodes and {len(relations)} edges")
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'entity_types': self._count_entity_types(),
            'relation_types': self._count_relation_types(),
            'density': nx.density(self.graph),
            'is_directed': self.graph.is_directed(),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count the number of entities of each type."""
        type_counts = {}
        for node in self.graph.nodes(data=True):
            entity_type = node[1].get('type', 'UNKNOWN')
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _count_relation_types(self) -> Dict[str, int]:
        """Count the number of relations of each type."""
        type_counts = {}
        for edge in self.graph.edges(data=True):
            relation_type = edge[2].get('type', 'UNKNOWN')
            type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
        return type_counts
    
    def save_graph(self, output_path: Path, format: str = 'json'):
        """
        Save the graph in the specified format.
        
        Args:
            output_path: Path to save the graph
            format: Output format ('json', 'html', 'png', or 'graphml')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Save as JSON with node and edge data
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'text': data.get('text', ''),
                        'type': data.get('type', 'UNKNOWN')
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'type': data.get('type', 'UNKNOWN')
                    }
                    for source, target, data in self.graph.edges(data=True)
                ]
            }
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
        elif format == 'html':
            # Create interactive visualization using pyvis
            net = Network(height='750px', width='100%', directed=True)
            
            # Add nodes
            for node, data in self.graph.nodes(data=True):
                net.add_node(
                    node,
                    label=data.get('text', ''),
                    title=f"Type: {data.get('type', 'UNKNOWN')}",
                    color=data.get('color', '#F0E68C')
                )
            
            # Add edges
            for source, target, data in self.graph.edges(data=True):
                net.add_edge(
                    source,
                    target,
                    title=data.get('type', 'UNKNOWN')
                )
            
            net.save_graph(str(output_path))
            
        elif format == 'png':
            # Create static visualization using matplotlib with reproducible layout
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(self.graph, seed=self.layout_seed)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                node_color=[data.get('color', '#F0E68C') for _, data in self.graph.nodes(data=True)],
                node_size=1000
            )
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos)
            
            # Draw labels
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels={node: data.get('text', '') for node, data in self.graph.nodes(data=True)}
            )
            
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
        elif format == 'graphml':
            # Save as GraphML format
            nx.write_graphml(self.graph, output_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved graph to {output_path} in {format} format")
    
    def get_subgraph(
        self, 
        node_id: str, 
        max_depth: int = 2,
        direction: Literal["forward", "backward", "both"] = "both"
    ) -> Optional[nx.DiGraph]:
        """
        Get a subgraph centered on a specific node.
        
        Args:
            node_id: ID of the central node
            max_depth: Maximum distance from the central node
            direction: Direction of expansion
                - "forward": only include successors (dependencies)
                - "backward": only include predecessors (prerequisites)
                - "both": include both (default)
            
        Returns:
            Optional[nx.DiGraph]: Subgraph or None if node not found
        """
        if node_id not in self.graph:
            return None
        
        # Get nodes within max_depth
        nodes = {node_id}
        for _ in range(max_depth):
            new_nodes = set()
            for node in nodes:
                if direction in ["backward", "both"]:
                    new_nodes.update(self.graph.predecessors(node))
                if direction in ["forward", "both"]:
                    new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
        
        return self.graph.subgraph(nodes) 