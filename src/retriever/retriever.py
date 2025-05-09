import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import yaml
import numpy as np
from sentence_transformers import CrossEncoder
from faiss import IndexFlatL2
import networkx as nx
from openai import OpenAI
from anthropic import Anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retriever that combines RAG and GraphRAG approaches."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the hybrid retriever."""
        # Load configuration
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()
        
        # Get mode and settings
        self.mode = self.config.get('mode', 'hybrid')
        self.mode_settings = self.config.get('mode_settings', {})
        
        # Initialize components based on mode
        if self.mode in ['rag', 'hybrid']:
            self._init_rag_components()
        if self.mode in ['graphrag', 'hybrid']:
            self._init_graphrag_components()
        
        # Initialize reranker if enabled
        if self.config.get('retrieval', {}).get('use_reranker', False):
            self._init_reranker()
    
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
        return self.prompts.get('defaults', {})

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_config = self.config.get("logging", {})
        logging.basicConfig(
            level=log_config.get("level", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_config.get("log_file", "hybrid_retriever.log")
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
    
    def _init_rag_components(self):
        """Initialize RAG components."""
        # Load FAISS index
        index_path = Path(self.config['paths']['faiss_index'])
        if index_path.exists():
            self.faiss_index = IndexFlatL2.load(str(index_path))
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        # Load document chunks
        chunks_path = Path(self.config['paths']['output_jsonl'])
        self.chunks = []
        with open(chunks_path, 'r') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        # Initialize embedding model
        provider = self.config['llm']['provider']
        model_name = self.config['llm']['model_name']
        
        if provider == 'openai':
            self.embedding_client = OpenAI(api_key=self.config['llm']['api_key'])
        elif provider == 'anthropic':
            self.embedding_client = Anthropic(api_key=self.config['llm']['api_key'])
        elif provider == 'local':
            # Check if model_name is in registry
            model_registry = self.config.get('model_registry', {})
            model_path = model_registry.get(model_name, model_name)
            
            # Get model settings
            model_settings = self.config.get('model_settings', {}).get('local', {})
            device = model_settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    load_in_8bit=model_settings.get('load_in_8bit', True),
                    torch_dtype=getattr(torch, model_settings.get('torch_dtype', 'float16')),
                    use_flash_attention=model_settings.get('use_flash_attention', True)
                )
                
                # Create text generation pipeline
                self.embedding_client = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_new_tokens=self.config['llm'].get('max_tokens', 1000),
                    temperature=self.config['llm'].get('temperature', 0.1),
                    batch_size=model_settings.get('max_batch_size', 4)
                )
                logger.info(f"Successfully loaded local model from {model_path}")
                
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _init_graphrag_components(self):
        """Initialize GraphRAG components."""
        # Load knowledge graph
        graph_path = Path(self.config['paths']['output_dir']) / f"{self.config['output']['graph_prefix']}.json"
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        for entity in self.graph_data['entities']:
            self.graph.add_node(entity['id'], **entity)
        for relation in self.graph_data['relations']:
            self.graph.add_edge(
                relation['source'],
                relation['target'],
                type=relation['type'],
                source_text=relation.get('source_text', '')
            )
    
    def _init_reranker(self):
        """Initialize reranker model."""
        reranker_config = self.config['retrieval']
        self.reranker = CrossEncoder(reranker_config['reranker_model'])
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant information using the configured mode."""
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        # Expand query if enabled
        if self.config['retrieval'].get('expand_query', False):
            query = self._expand_query(query)
        
        if self.mode == 'rag':
            results = self._rag_retrieve(query, top_k)
        elif self.mode == 'graphrag':
            results = self._graphrag_retrieve(query, top_k)
        else:  # hybrid
            results = self._hybrid_retrieve(query, top_k)
        
        # Rerank results if enabled
        if self.config['retrieval']['use_reranker']:
            results = self._rerank_results(query, results)
        
        return results[:top_k]
    
    def _expand_query(self, query: str) -> str:
        """Expand query using LLM."""
        try:
            prompt = self._get_prompt('retrieval', 'query_expansion', query=query)
            
            if isinstance(self.embedding_client, OpenAI):
                response = self.embedding_client.chat.completions.create(
                    model=self.config['llm']['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                expanded_query = response.choices[0].message.content.strip()
            elif isinstance(self.embedding_client, Anthropic):
                response = self.embedding_client.messages.create(
                    model=self.config['llm']['model_name'],
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                expanded_query = response.content[0].text.strip()
            elif isinstance(self.embedding_client, pipeline):
                # For local models using transformers pipeline
                response = self.embedding_client(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    num_return_sequences=1
                )
                expanded_query = response[0]['generated_text'][len(prompt):].strip()
            else:
                return query
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query
    
    def _rag_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using RAG approach."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity score
                    'source': chunk.get('source', ''),
                    'metadata': chunk.get('metadata', {})
                })
        
        return results
    
    def _graphrag_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using GraphRAG approach."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Find most similar entities
        entity_scores = []
        for node_id, node_data in self.graph.nodes(data=True):
            if 'embedding' in node_data:
                entity_embedding = node_data['embedding']
                similarity = self._cosine_similarity(query_embedding, entity_embedding)
                entity_scores.append((node_id, similarity))
        
        # Sort by similarity
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get subgraphs for top entities
        results = []
        max_hops = self.mode_settings.get('graphrag', {}).get('max_hop_distance', 3)
        
        for entity_id, score in entity_scores[:top_k]:
            subgraph = self._get_entity_subgraph(entity_id, max_hops)
            subgraph_text = self._subgraph_to_text(subgraph)
            
            results.append({
                'text': subgraph_text,
                'score': score,
                'source': f"graph_{entity_id}",
                'metadata': {
                    'entity_id': entity_id,
                    'subgraph_size': len(subgraph.nodes()),
                    'max_hops': max_hops
                }
            })
        
        return results
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using hybrid approach combining RAG and GraphRAG."""
        # Get results from both approaches
        rag_results = self._rag_retrieve(query, top_k)
        graphrag_results = self._graphrag_retrieve(query, top_k)
        
        # Get weights from config
        rag_weight = self.mode_settings.get('hybrid', {}).get('rag_weight', 0.6)
        graphrag_weight = self.mode_settings.get('hybrid', {}).get('graphrag_weight', 0.4)
        
        # Combine results based on fusion method
        fusion_method = self.mode_settings.get('hybrid', {}).get('fusion_method', 'weighted_sum')
        
        if fusion_method == 'weighted_sum':
            # Simple weighted combination
            combined_results = []
            
            # Add RAG results with weight
            for result in rag_results:
                result['score'] *= rag_weight
                result['source'] = 'rag'
                combined_results.append(result)
            
            # Add GraphRAG results with weight
            for result in graphrag_results:
                result['score'] *= graphrag_weight
                result['source'] = 'graphrag'
                combined_results.append(result)
            
            # Sort by score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            
        else:  # reciprocal_rank_fusion
            # Reciprocal Rank Fusion (RRF)
            combined_results = []
            seen_texts = set()
            
            # Process RAG results
            for i, result in enumerate(rag_results):
                if result['text'] not in seen_texts:
                    rank = i + 1
                    score = rag_weight / (rank + 60)  # RRF formula
                    result['score'] = score
                    result['source'] = 'rag'
                    combined_results.append(result)
                    seen_texts.add(result['text'])
            
            # Process GraphRAG results
            for i, result in enumerate(graphrag_results):
                if result['text'] not in seen_texts:
                    rank = i + 1
                    score = graphrag_weight / (rank + 60)  # RRF formula
                    result['score'] = score
                    result['source'] = 'graphrag'
                    combined_results.append(result)
                    seen_texts.add(result['text'])
            
            # Sort by score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder."""
        if not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, result['text']) for result in results]
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Update result scores
        for result, score in zip(results, scores):
            result['score'] = float(score)
        
        # Sort by new scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using configured model."""
        if isinstance(self.embedding_client, OpenAI):
            response = self.embedding_client.embeddings.create(
                model=self.config['mode_settings']['rag']['embedding_model'],
                input=text
            )
            return response.data[0].embedding
        elif isinstance(self.embedding_client, Anthropic):
            response = self.embedding_client.embeddings.create(
                model=self.config['mode_settings']['rag']['embedding_model'],
                input=text
            )
            return response.embedding
        else:
            # For local models, use the model's embedding layer
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state as embedding
                embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
                return embeddings.tolist()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _get_entity_subgraph(self, entity_id: str, max_hops: int = None) -> nx.DiGraph:
        """Get subgraph around an entity."""
        if max_hops is None:
            max_hops = self.mode_settings.get('graphrag', {}).get('max_hop_distance', 3)
        
        # Get all nodes within max_hops
        nodes = {entity_id}
        for _ in range(max_hops):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
        
        # Create subgraph
        return self.graph.subgraph(nodes)
    
    def _subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
        """Convert subgraph to text representation."""
        text_parts = []
        
        # Add entity information
        for node_id, node_data in subgraph.nodes(data=True):
            text_parts.append(f"Entity: {node_data.get('text', node_id)}")
            text_parts.append(f"Type: {node_data.get('type', 'unknown')}")
            if 'properties' in node_data:
                for key, value in node_data['properties'].items():
                    text_parts.append(f"{key}: {value}")
            text_parts.append("")
        
        # Add relationship information
        for source, target, edge_data in subgraph.edges(data=True):
            source_text = subgraph.nodes[source].get('text', source)
            target_text = subgraph.nodes[target].get('text', target)
            relation_type = edge_data.get('type', 'related_to')
            text_parts.append(f"{source_text} {relation_type} {target_text}")
        
        return "\n".join(text_parts)

def main():
    """Example usage of the HybridRetriever."""
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Example query
    query = "What are the requirements for LEED certification?"
    
    # Retrieve results
    results = retriever.retrieve(query)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text'][:200]}...")
        if 'metadata' in result:
            print(f"Metadata: {result['metadata']}")

if __name__ == "__main__":
    main() 