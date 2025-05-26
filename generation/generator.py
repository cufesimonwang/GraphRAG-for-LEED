import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import yaml
from openai import OpenAI
from anthropic import Anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from retrievers.retriever import HybridRetriever # Old import
from retrievers.factory import get_retriever # New import
from kg_construction.neo4j_manager import Neo4jManager # For type hinting

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generator that creates responses based on retrieved information."""
    
    def __init__(self, 
                 config: Dict, # Changed to accept loaded config dict
                 neo4j_manager: Optional[Neo4jManager] = None,
                 prompts: Optional[Dict] = None): # Accept loaded prompts
        """Initialize the response generator."""
        self.config = config # Use pre-loaded config
        if prompts:
            self.prompts = prompts
        else:
            self.prompts = self._load_prompts_from_file() # Load if not provided
        
        self._setup_logging()
        
        # Initialize retriever using the factory
        # The factory will internally use the config to decide which retriever to instantiate
        # and will pass neo4j_manager to those that need it.
        self.retriever = get_retriever(self.config, neo4j_manager) 
        
        # Initialize LLM
        self._init_llm()

    def _load_prompts_from_file(self) -> Dict: # Renamed from _load_prompts
        """Load prompts from prompts.yaml file specified in config."""
        prompts_path_str = self.config.get("paths", {}).get("prompts_file", "config/prompts.yaml")
        prompts_path = Path(prompts_path_str)
        try:
            with open(prompts_path, "r") as f:
                loaded_prompts = yaml.safe_load(f)
                if not loaded_prompts:
                    logger.warning(f"Prompts file '{prompts_path}' is empty. Using default prompts.")
                    return self._get_default_prompts() # Ensure this default makes sense or remove
                return loaded_prompts
        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_path}. Using default prompts.")
            return self._get_default_prompts()
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_path}: {e}. Using default prompts.")
            return self._get_default_prompts()

    # Removed _load_config as config is now passed in __init__
    # def _load_config(self, config_path: str) -> Dict:
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
            filename=log_config.get("log_file", "generator.log")
        )
    
    def _init_llm(self):
        """Initialize the language model."""
        provider = self.config['llm']['provider']
        model_name = self.config['llm']['model_name']
        
        if provider == 'openai':
            self.llm = OpenAI(api_key=self.config['llm']['api_key'])
        elif provider == 'anthropic':
            self.llm = Anthropic(api_key=self.config['llm']['api_key'])
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
                self.llm = pipeline(
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
    
    def _get_prompt(self, prompt_type: str, prompt_name: str, **kwargs) -> str:
        """Get a prompt from the prompts configuration."""
        # Ensure prompts are loaded
        if not self.prompts or prompt_type not in self.prompts or prompt_name not in self.prompts[prompt_type]:
            logger.error(f"Prompt {prompt_type}.{prompt_name} not found in loaded prompts. Falling back to default or empty.")
            # Attempt to get from default prompts if structure exists there
            default_prompts = self._get_default_prompts()
            prompt_template = default_prompts.get(prompt_type, {}).get(prompt_name, "")
            if not prompt_template:
                 logger.error(f"Prompt {prompt_type}.{prompt_name} also not found in default prompts.")
                 return f"ERROR: Prompt '{prompt_type}.{prompt_name}' not found. Query: {kwargs.get('query', '')}" # Fallback
            logger.warning(f"Using default prompt for {prompt_type}.{prompt_name}.")
            return prompt_template.format(**kwargs)
        
        try:
            # Assuming self.prompts is already the loaded dictionary from YAML
            prompt_template = self.prompts[prompt_type][prompt_name]
            return prompt_template.format(**kwargs)
        except KeyError: # Should be caught by the initial check, but as a safeguard
            logger.error(f"Prompt {prompt_type}.{prompt_name} definitively not found. Query: {kwargs.get('query', '')}")
            return f"ERROR: Prompt '{prompt_type}.{prompt_name}' not found. Query: {kwargs.get('query', '')}" # Fallback
        except Exception as e:
            logger.error(f"Error formatting prompt {prompt_type}.{prompt_name}: {e}")
            return f"ERROR formatting prompt. Query: {kwargs.get('query', '')}" # Fallback
    
    def _format_context(self, retrieved_data: Dict[str, List[Any]]) -> Dict[str, str]:
        """
        Formats retrieved chunks and KG paths into strings for the LLM context.
        """
        formatted_chunks_str = "No relevant document excerpts found.\n"
        if retrieved_data.get('chunks'):
            chunk_parts = []
            for i, chunk in enumerate(retrieved_data['chunks']):
                chunk_id = chunk.get('chunk_id', f"chunk_{i}")
                doc_id = chunk.get('doc_id', 'N/A')
                page = chunk.get('page', 'N/A')
                section = chunk.get('section', 'N/A')
                text = chunk.get('text', '')
                score = chunk.get('score', 0.0)
                chunk_parts.append(
                    f"Chunk ID: {chunk_id}\n"
                    f"Source Document: {doc_id}\n"
                    f"Page: {page}\n"
                    f"Section: {section}\n"
                    f"Text: {text}\n"
                    f"Score: {score:.4f}\n"
                )
            formatted_chunks_str = "\n---\n".join(chunk_parts)

        formatted_kg_paths_str = "No relevant knowledge graph paths found.\n"
        if retrieved_data.get('kg_paths'):
            kg_path_parts = []
            for i, path_info in enumerate(retrieved_data['kg_paths']):
                # Assuming path_info has 'text' (natural language path) and 'score'
                # and potentially 'nodes' for a summary
                path_id = path_info.get('id', f"kgpath_{i}") # If paths have inherent IDs
                path_text = path_info.get('text', 'N/A')
                score = path_info.get('score', 0.0)
                
                nodes_summary_parts = []
                if 'nodes' in path_info and isinstance(path_info['nodes'], list):
                    for node in path_info['nodes']:
                        nodes_summary_parts.append(f"{node.get('name', 'Unknown Node')} ({node.get('type', 'UnknownType')})")
                nodes_summary = " -> ".join(nodes_summary_parts) if nodes_summary_parts else "N/A"

                kg_path_parts.append(
                    f"KG Path ID: {path_id}\n"
                    f"Path: {path_text}\n" # This is the natural language version from GraphRetriever
                    f"Score: {score:.4f}\n"
                    f"Nodes Summary: {nodes_summary}\n"
                )
            formatted_kg_paths_str = "\n---\n".join(kg_path_parts)
            
        return {
            "formatted_chunks": formatted_chunks_str,
            "formatted_kg_paths": formatted_kg_paths_str
        }
    
    def generate(self, query: str, top_k: int = None) -> Dict:
        """
        Generate a response based on the query and retrieved information.
        
        Args:
            query: The user's query
            top_k: Number of results to retrieve (default from config retriever settings)
            
        Returns:
            Dictionary containing:
            - answer: Generated response text
            - sources: List of source identifiers (chunk_id, kg_path_id)
            - metadata: Additional information
        """
        # Retrieve relevant information using the configured retriever
        # The retriever instance (e.g., HybridRetriever) now returns a dict 
        # like {"chunks": [...], "kg_paths": [...], "metadata": ...}
        retrieved_data_dict = self.retriever.retrieve(query, top_k=top_k) 
        
        # Format context for the LLM prompt
        formatted_contexts = self._format_context(retrieved_data_dict)
        
        # Get generation prompt
        prompt = self._get_prompt(
            'generation',
            'response_generation', # This prompt was updated in prompts.yaml
            query=query,
            formatted_chunks=formatted_contexts['formatted_chunks'],
            formatted_kg_paths=formatted_contexts['formatted_kg_paths']
        )
        
        # Generate response
        try:
            if isinstance(self.llm, OpenAI):
                response = self.llm.chat.completions.create(
                    model=self.config['llm']['model_name'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate information about LEED certification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens']
                )
                response_text = response.choices[0].message.content.strip()
                
            elif isinstance(self.llm, Anthropic):
                response = self.llm.messages.create(
                    model=self.config['llm']['model_name'],
                    max_tokens=self.config['llm']['max_tokens'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.content[0].text.strip()
                
            elif isinstance(self.llm, pipeline):
                # For local models using transformers pipeline
                response = self.llm(
                    prompt,
                    max_new_tokens=self.config['llm']['max_tokens'],
                    temperature=self.config['llm']['temperature'],
                    do_sample=True,
                    num_return_sequences=1
                )
                response_text = response[0]['generated_text'][len(prompt):].strip()
            else:
                raise ValueError("Unsupported LLM type")
            
            # Collect source IDs from the retrieved data that was passed to the LLM
            source_ids = []
            for i, chunk in enumerate(retrieved_data_dict.get('chunks', [])):
                source_ids.append(chunk.get('chunk_id', f"chunk_fallback_{i}"))
            for i, path in enumerate(retrieved_data_dict.get('kg_paths', [])):
                # Assuming paths might have an 'id' field, or generate one
                source_ids.append(path.get('id', f"kgpath_fallback_{i}"))

            # The LLM might also cite sources in its response_text. 
            # Parsing these out is an advanced step. For now, we list what was provided.
            
            return {
                'answer': response_text, # Changed from 'response' to 'answer'
                'sources': source_ids,   # List of chunk_ids and kg_path_ids
                'metadata': {
                    'num_context_chunks': len(retrieved_data_dict.get('chunks', [])),
                    'num_context_kg_paths': len(retrieved_data_dict.get('kg_paths', [])),
                    'retriever_metadata': retrieved_data_dict.get('metadata', {}), # Metadata from the retriever
                    'query': query
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return {
                'answer': "I apologize, but I encountered an error while generating the response.",
                'sources': [],
                'metadata': {
                    'error': str(e),
                    'query': query
                }
            }

def main():
    """Example usage of the ResponseGenerator, demonstrating initialization and use."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Construct absolute paths for config files relative to this script's location
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent 
    config_file_path = root_dir / "config" / "config.yaml"
    prompts_file_path = root_dir / "config" / "prompts.yaml"

    if not config_file_path.exists():
        logger.error(f"Main test: Configuration file not found at {config_file_path}. Please ensure it exists.")
        return
    if not prompts_file_path.exists():
        logger.error(f"Main test: Prompts file not found at {prompts_file_path}. Please ensure it exists.")
        return

    loaded_config = {}
    with open(config_file_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    loaded_prompts = {}
    with open(prompts_file_path, 'r', encoding='utf-8') as f:
        loaded_prompts = yaml.safe_load(f)
    
    # The ResponseGenerator's __init__ expects prompts to be part of the main config dict
    # or passed as a separate argument. If prompts are globally accessible or loaded by each class,
    # this might not be needed. For this refactor, assuming prompts are passed via config.
    # If ResponseGenerator loads its own prompts using its _load_prompts_from_file,
    # then it just needs the main 'config' dictionary.
    # The current __init__ of ResponseGenerator allows prompts to be passed or loaded.
    # For simplicity in main, we'll ensure it's loaded if not part of the main config dict already.
    # if 'prompts' not in loaded_config: # Assuming prompts are separate and need to be added
    #    loaded_config['prompts'] = loaded_prompts # This line is removed as __init__ handles it.

    neo4j_manager_instance: Optional[Neo4jManager] = None
    # Determine if Neo4jManager is needed based on the retriever strategy in config
    # The get_retriever factory will handle this logic internally if neo4j_manager is passed to it.
    # So, we initialize Neo4jManager if any strategy potentially needing it is active.
    # A simpler approach: always try to init Neo4jManager if config exists, 
    # and pass it to ResponseGenerator. get_retriever will then pass it to sub-retrievers as needed.
    
    if loaded_config.get("neo4j"): # Check if neo4j config section exists
        try:
            # Pass the full config path to Neo4jManager, as it loads its own section
            neo4j_manager_instance = Neo4jManager(config_path=str(config_file_path))
            logger.info("Main test: Neo4jManager initialized.")
        except Exception as e:
            logger.error(f"Main test: Failed to initialize Neo4jManager: {e}. Graph/Hybrid retriever strategies might fail.")
            # Depending on the default strategy, this might be a fatal error for the test.
    else:
        logger.warning("Main test: Neo4j configuration not found in config.yaml. Graph/Hybrid retrievers will not work.")

    try:
        # Initialize generator with the loaded config dictionary, Neo4jManager, and loaded prompts.
        # The ResponseGenerator's __init__ will use get_retriever, which in turn uses the config
        # to decide which retriever to instantiate and if neo4j_manager is needed for that retriever.
        generator = ResponseGenerator(config=loaded_config, 
                                      neo4j_manager=neo4j_manager_instance, 
                                      prompts=loaded_prompts) # Pass loaded prompts
        
        queries_to_test = [
            "What are the acoustic performance requirements for LEED v4 schools?",
            "Tell me about daylighting credits.",
            "Maximum LTV for composite wood materials in LEED projects."
        ]
        
        for query in queries_to_test:
            logger.info(f"\n--- Main test: Generating response for query: '{query}' ---")
            result = generator.generate(query) # top_k will use retriever's default
            
            print("\nAnswer:")
            print(result.get('answer'))
            
            print("\nSources (IDs of context items provided to LLM):")
            if result.get('sources'):
                for source_id in result['sources']:
                    print(f"- {source_id}")
            else:
                print("No source IDs listed (or retriever found no context).")
            
            print("\nMetadata:")
            # Metadata can be verbose, so only print selected parts or a summary if too long
            meta = result.get('metadata', {})
            print(f"  Query: {meta.get('query')}")
            print(f"  Context Chunks: {meta.get('num_context_chunks')}")
            print(f"  Context KG Paths: {meta.get('num_context_kg_paths')}")
            if meta.get('retriever_metadata'):
                print(f"  Retriever Strategy: {meta['retriever_metadata'].get('retrieval_strategy')}")
                if meta['retriever_metadata'].get('retrieval_strategy') == 'HybridRetriever':
                    print(f"    Vector Retriever Model: {meta['retriever_metadata'].get('vector_retriever_metadata',{}).get('embedding_model')}")
                    print(f"    Graph Retriever Entities: {meta['retriever_metadata'].get('graph_retriever_metadata',{}).get('parsed_entities')}")


    except Exception as e:
        logger.error(f"Error in ResponseGenerator main test execution: {e}", exc_info=True)
    finally:
        if neo4j_manager_instance:
            neo4j_manager_instance.close()
            logger.info("Main test: Neo4jManager connection closed.")


if __name__ == "__main__":
    main()