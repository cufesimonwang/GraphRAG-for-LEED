from transformers import pipeline
from typing import List, Dict, Tuple, Optional, Union
import json
import ast
import yaml
from pathlib import Path
import logging
import re
from slugify import slugify
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from anthropic import Anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..content_extractor import ContentExtractor

logger = logging.getLogger(__name__)

class KnowledgeGraphExtractor:
    """Extracts entities and relationships from text using LLMs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the knowledge graph extractor."""
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self.content_extractor = ContentExtractor(config_path)
        self._setup_logging()
        
        # Load LLM configuration
        self.llm_config = self.config.get('llm', {})
        self.mode = self.config.get('mode', 'graphrag')  # Options: 'rag', 'graphrag', 'hybrid'
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Load relation mapping
        self.relation_map = self.config.get('extraction', {}).get('relation_types', [
            'requires', 'contributes_to', 'is_part_of', 'has_requirement',
            'related_to', 'depends_on', 'influences', 'affects', 'supports', 'enables'
        ])
    
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
                prompts = yaml.safe_load(f)
                if not prompts:
                    logger.warning(f"Empty prompts file: {prompts_path}")
                    return self._get_default_prompts()
                return prompts
        except FileNotFoundError:
            logger.warning(f"Prompts file not found: {prompts_path}")
            return self._get_default_prompts()
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_path}: {e}")
            return self._get_default_prompts()

    def _get_default_prompts(self) -> Dict:
        """Return default prompts if prompts.yaml is not found."""
        return {
            'kg_extraction': {
                'relation_extraction': """
                Extract knowledge graph triples from the following text. 
                Each triple should have a subject, predicate, and object.
                Return the triples as a JSON array of objects with 'subject', 'predicate', and 'object' fields.
                Text: {text}
                """,
                'entity_type_inference': """
                Infer the type of the following entity based on its context.
                Entity: {text}
                Context: {context}
                Return the type as a string.
                """,
                'relation_type_inference': """
                Infer the type of relationship between these entities based on the context.
                Source: {source}
                Target: {target}
                Context: {context}
                Return the relationship type as a string.
                """
            }
        }

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_config = self.config.get("logging", {})
        logging.basicConfig(
            level=log_config.get("level", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_config.get("log_file", "kg_extractor.log")
        )

    def _init_llm_client(self):
        """Initialize the LLM client based on configuration."""
        provider = self.llm_config.get("provider")
        model_name = self.llm_config.get("model_name")
        
        if provider == "openai":
            self.client = OpenAI(api_key=self.llm_config["api_key"])
        elif provider == "anthropic":
            self.client = Anthropic(api_key=self.llm_config["api_key"])
        elif provider == "local":
            # Check if model_name is in registry
            model_registry = self.config.get('model_registry', {})
            model_path = model_registry.get(model_name, model_name)  # Use direct path if not in registry
            
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
                self.client = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_new_tokens=self.llm_config.get('max_tokens', 1000),
                    temperature=self.llm_config.get('temperature', 0.1),
                    batch_size=model_settings.get('max_batch_size', 4)
                )
                logger.info(f"Successfully loaded local model from {model_path}")
                
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM with retry logic."""
        try:
            if isinstance(self.client, OpenAI):
                response = self.client.chat.completions.create(
                    model=self.llm_config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config.get("temperature", 0.3),
                    max_tokens=self.llm_config.get("max_tokens", 1024),
                )
                return response.choices[0].message.content.strip()
                
            elif isinstance(self.client, Anthropic):
                response = self.client.messages.create(
                    model=self.llm_config["model_name"],
                    max_tokens=self.llm_config.get("max_tokens", 1024),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            elif isinstance(self.client, pipeline):
                # For local models using transformers pipeline
                response = self.client(
                    prompt,
                    max_new_tokens=self.llm_config.get("max_tokens", 1024),
                    temperature=self.llm_config.get("temperature", 0.3),
                    do_sample=True,
                    num_return_sequences=1
                )
                # Extract generated text, removing the input prompt
                generated_text = response[0]['generated_text']
                return generated_text[len(prompt):].strip()
                
            else:
                raise ValueError(f"Unsupported LLM client type: {type(self.client)}")
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise
    
    def _get_prompt(self, prompt_type: str, prompt_name: str, **kwargs) -> str:
        """Get a prompt from the prompts configuration."""
        try:
            prompt = self.prompts[prompt_type][prompt_name]
            return prompt.format(**kwargs)
        except KeyError:
            logger.warning(f"Prompt {prompt_type}.{prompt_name} not found, using default")
            return self._get_default_prompts().get(prompt_type, {}).get(prompt_name, "")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return self._get_default_prompts().get(prompt_type, {}).get(prompt_name, "")
    
    def _extract_triples(self, text: str) -> List[Dict]:
        """Extract triples from text using LLM."""
        prompt = self._get_prompt("kg_extraction", "relation_extraction", text=text)
        response = self._get_llm_response(prompt)
        
        try:
            triples = json.loads(response)
            if not isinstance(triples, list):
                triples = [triples]
            return triples
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _extract_entities_from_triples(self, triples: List[Dict]) -> List[Dict]:
        """Extract and deduplicate entities from triples."""
        entities = {}
        
        for triple in triples:
            # Process subject
            subject = triple.get('subject', '')
            if subject:
                entity_id = self._generate_entity_id(subject)
                if entity_id not in entities:
                    entity_type = self._infer_entity_type(subject, "")
                    if entity_type == 'CONCEPT':
                        logger.debug(f"Using CONCEPT as fallback type for: {subject}")
                    entities[entity_id] = {
                        'id': entity_id,
                        'text': subject,
                        'type': entity_type,
                        'source_text': triple.get('source_text', '')
                    }
            
            # Process object
            obj = triple.get('object', '')
            if obj:
                entity_id = self._generate_entity_id(obj)
                if entity_id not in entities:
                    entity_type = self._infer_entity_type(obj, "")
                    if entity_type == 'CONCEPT':
                        logger.debug(f"Using CONCEPT as fallback type for: {obj}")
                    entities[entity_id] = {
                        'id': entity_id,
                        'text': obj,
                        'type': entity_type,
                        'source_text': triple.get('source_text', '')
                    }
        
        return list(entities.values())
    
    def _generate_entity_id(self, text: str) -> str:
        """Generate a stable entity ID from text."""
        # Convert to slug format while preserving semantic differences
        base_id = text.lower().replace(' ', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')
        return base_id
    
    def _infer_entity_type(self, text: str, context: str = "") -> str:
        """Infer the type of an entity using LLM."""
        prompt = self._get_prompt("kg_extraction", "entity_type_inference", text=text, context=context)
        response = self._get_llm_response(prompt)
        return response.strip()
    
    def _infer_relation_type(self, source: str, target: str, context: str = "") -> str:
        """Infer the type of relationship between entities using LLM."""
        prompt = self._get_prompt("kg_extraction", "relation_type_inference", source=source, target=target, context=context)
        response = self._get_llm_response(prompt)
        return self._normalize_relation(response.strip())
    
    def _normalize_relation(self, relation: str) -> str:
        """Normalize relation using the relation map."""
        relation = relation.lower().replace(' ', '_')
        return relation if relation in self.relation_map else 'related_to'
    
    def process(self, text: str, return_triples: bool = False) -> Union[
        Tuple[List[Dict], List[Dict]], Tuple[List[Dict], List[Dict], List[Dict]]
    ]:
        """Process text to extract entities and relations."""
        # Extract triples
        triples = self._extract_triples(text)
        
        # Extract entities
        entities = self._extract_entities_from_triples(triples)
        
        # Extract relations
        relations = []
        for triple in triples:
            subject_id = self._generate_entity_id(triple['subject'])
            object_id = self._generate_entity_id(triple['object'])
            relation_type = self._infer_relation_type(triple['subject'], triple['object'])
            
            relations.append({
                'source': subject_id,
                'target': object_id,
                'type': relation_type,
                'source_text': triple.get('source_text', text)
            })
        
        if return_triples:
            return entities, relations, triples
        return entities, relations
    
    def extract_triplet_dict(self, text: str) -> List[Dict]:
        """
        Extract raw triplets from text (for testing).
        
        Args:
            text: Input text
            
        Returns:
            List of triple dictionaries
        """
        return self._extract_triples(text)
    
    def process_file(self, file_path: Path) -> Dict:
        """
        Process a file and extract knowledge graph elements.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing:
            - entities: List of extracted entities
            - relations: List of extracted relations
            - chunks: List of text chunks (for RAG/hybrid mode)
            - metadata: File metadata
        """
        # Process file using content extractor
        result = self.content_extractor.process_file(file_path, output_format="jsonl")
        
        # Extract knowledge graph elements
        entities, relations = self.process(result["text"])
        
        # Add relationships from diagrams if any
        if result["relationships"]:
            diagram_entities, diagram_relations = self.process("\n".join(result["relationships"]))
            entities.extend(diagram_entities)
            relations.extend(diagram_relations)
        
        return {
            "entities": entities,
            "relations": relations,
            "chunks": result["chunks"],
            "metadata": result["metadata"]
        } 