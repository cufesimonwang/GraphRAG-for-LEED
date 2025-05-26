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


logger = logging.getLogger(__name__)

class KnowledgeGraphExtractor:
    """Extracts entities and relationships from text using LLMs, designed for KGGen principles."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the knowledge graph extractor."""
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()
        
        # Load LLM configuration
        self.llm_config = self.config.get('llm', {})
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Load relation mapping for normalization (optional, can be part of prompt or post-processing)
        self.relation_map = self.config.get('extraction', {}).get('relation_types', [
            'REQUIRES', 'CONTRIBUTES_TO', 'IS_PART_OF', 'HAS_REQUIREMENT',
            'RELATED_TO', 'DEPENDS_ON', 'INFLUENCES', 'AFFECTS', 'SUPPORTS', 'ENABLES', 
            'IS_A', 'HAS_PROPERTY', 'MEASURES', 'APPLIES_TO', 'DEFINES', 'INCLUDES' 
            # Extended based on typical KG relations and previous context
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
        # Ensure prompts_file path is correctly specified in config.yaml under 'paths'
        prompts_file_path_str = self.config.get("paths", {}).get("prompts_file", "config/prompts.yaml")
        prompts_file_path = Path(prompts_file_path_str)
        try:
            with open(prompts_file_path, "r") as f:
                prompts_data = yaml.safe_load(f)
                if not prompts_data:
                    logger.warning(f"Prompts file is empty: {prompts_file_path}")
                    # Fallback to a very basic default if absolutely necessary, though this should be configured properly.
                    return self._get_emergency_default_prompts()
                logger.info(f"Successfully loaded prompts from {prompts_file_path}")
                return prompts_data
        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_file_path}. Please check the path.")
            raise  # Or return default prompts and log a critical error
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_file_path}: {e}")
            raise # Or return default prompts

    def _get_emergency_default_prompts(self) -> Dict:
        """Provides a fallback prompt if the YAML file is missing/empty. This is not ideal."""
        logger.warning("Using emergency default prompts. Please configure prompts.yaml correctly.")
        return {
            'kg_extraction': {
                'relation_extraction': {
                    'default': """
                    Extract knowledge graph triples from the following text.
                    For each triple, identify the head entity (subject), the relation (predicate), and the tail entity (object).
                    For each head and tail entity, provide its name and a likely type.
                    The relation should be a concise verb phrase.
                    Format the output as a JSON list of objects. Each object should have 'head_entity' (with 'name' and 'type'), 'relation', and 'tail_entity' (with 'name' and 'type').
                    Text: {text}
                    """
                },
                'entity_type_inference': { # Kept for potential use if LLM struggles with direct typing
                    'default': "Infer the type of the following entity: {text}. Context: {context}. Type should be one of: Concept, Process, Material, Standard, Organization, Location, Metric, Requirement, Credit, Prerequisite, System, Equipment, Role, Document, Rating_System_Version, Space_Type, Performance_Criteria, Environmental_Impact, Mitigation_Strategy, Verification_Method, Threshold_Value, Certification_Level, Design_Phase, Construction_Phase, Operation_Phase, Policy, Regulation, Guideline, Best_Practice, Case_Study, Software_Tool, Data_Source, Professional_Credential, Educational_Resource, Event, Project, Building_Component, Utility, Resource, Waste_Stream, Pollutant, Health_Impact, Economic_Impact, Social_Impact, Life_Cycle_Stage, Performance_Indicator. Return only the type as a string."
                },
                 'relation_type_inference': { # Kept for potential use
                    'default': "Infer the type of relationship between '{source}' and '{target}' given context: {context}. Relation type should be a concise verb phrase, ideally from this list: REQUIRES, CONTRIBUTES_TO, IS_PART_OF, HAS_REQUIREMENT, RELATED_TO, DEPENDS_ON, INFLUENCES, AFFECTS, SUPPORTS, ENABLES, IS_A, HAS_PROPERTY, MEASURES, APPLIES_TO, DEFINES, INCLUDES. Return only the relation type as a string."
                }
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
            # Prompts are expected to be nested, e.g., self.prompts['kg_extraction']['relation_extraction']['default']
            # or self.prompts['kg_extraction']['relation_extraction'] if 'default' is not a sub-key.
            # Adjusting to access prompts more robustly.
            prompt_template_obj = self.prompts.get(prompt_type, {}).get(prompt_name, {})
            if isinstance(prompt_template_obj, dict): # If there are sub-prompts like 'default', 'example'
                prompt_template = prompt_template_obj.get('default', "")
            elif isinstance(prompt_template_obj, str): # If the key directly holds the prompt string
                prompt_template = prompt_template_obj
            else:
                prompt_template = ""

            if not prompt_template:
                 logger.error(f"Prompt template for {prompt_type}.{prompt_name} is empty or not found. Using emergency default.")
                 # Attempt to use emergency default if main loading failed or was incomplete
                 if hasattr(self, '_get_emergency_default_prompts'):
                     emergency_prompts = self._get_emergency_default_prompts()
                     prompt_template_obj = emergency_prompts.get(prompt_type, {}).get(prompt_name, {})
                     if isinstance(prompt_template_obj, dict):
                         prompt_template = prompt_template_obj.get('default', "")
                     elif isinstance(prompt_template_obj, str):
                         prompt_template = prompt_template_obj
                 if not prompt_template: # Still no prompt
                    raise KeyError(f"Prompt {prompt_type}.{prompt_name} ultimately not found.")

            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Prompt key not found: {e}. Check your prompts.yaml structure and config.")
            # Fallback to a very generic instruction if a specific prompt is missing.
            return f"Extract information based on: {kwargs.get('text', '')}" 
        except Exception as e:
            logger.error(f"Error formatting prompt {prompt_type}.{prompt_name}: {e}")
            return f"Error in prompt: {kwargs.get('text', '')}" # Basic fallback

    def _extract_triples(self, text: str) -> List[Dict]:
        """
        Extracts structured triples from text using LLM based on the 'relation_extraction' prompt.
        The prompt should instruct the LLM to return a JSON list of objects like:
        [{"head_entity": {"name": "...", "type": "..."}, "relation": "...", "tail_entity": {"name": "...", "type": "..."}}]
        """
        prompt = self._get_prompt("kg_extraction", "relation_extraction", text=text)
        llm_response_str = self._get_llm_response(prompt)
        
        try:
            # Attempt to find JSON array within the response if the LLM adds extra text
            match = re.search(r'\[.*\]', llm_response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = llm_response_str # Assume the whole response is JSON

            extracted_data = json.loads(json_str)
            
            if not isinstance(extracted_data, list):
                logger.warning(f"LLM response was not a list as expected, but type {type(extracted_data)}. Response: {llm_response_str}")
                # Attempt to wrap it in a list if it's a single dictionary object
                if isinstance(extracted_data, dict) and "head_entity" in extracted_data: 
                    return [extracted_data]
                return []

            # Basic validation of structure for each item in the list
            valid_triples = []
            for item in extracted_data:
                if isinstance(item, dict) and \
                   "head_entity" in item and isinstance(item["head_entity"], dict) and "name" in item["head_entity"] and \
                   "relation" in item and \
                   "tail_entity" in item and isinstance(item["tail_entity"], dict) and "name" in item["tail_entity"]:
                    valid_triples.append(item)
                else:
                    logger.warning(f"Skipping invalid triple structure from LLM: {item}")
            return valid_triples

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON response for triples: {e}. Response was: {llm_response_str}")
            return []
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during triple extraction: {e}. Response was: {llm_response_str}")
            return []

    def _infer_entity_type(self, entity_name: str, context: str = "") -> str:
        """
        Infer the type of an entity using LLM.
        This is a fallback if the main relation_extraction prompt doesn't provide types.
        """
        prompt = self._get_prompt("kg_extraction", "entity_type_inference", text=entity_name, context=context)
        response = self._get_llm_response(prompt)
        # Ensure response is a single type string, and normalize
        inferred_type = response.strip().upper().replace(" ", "_")
        logger.debug(f"Inferred type for '{entity_name}': {inferred_type}")
        return inferred_type
    
    def _normalize_relation(self, relation_name: str, context: str = "") -> str:
        """
        Normalize relation type to a standard form, potentially using LLM if needed or simple mapping.
        """
        # Simple normalization: uppercase and replace spaces
        norm_relation = relation_name.strip().upper().replace(" ", "_")
        
        # Optional: Check against self.relation_map or use LLM for more advanced normalization/mapping
        # For now, simple normalization is used. If LLM provides good relation verbs, this might be sufficient.
        # Example of mapping (if self.relation_map contains preferred forms):
        # for standard_rel in self.relation_map:
        #     if norm_relation == standard_rel or norm_relation in standard_rel or standard_rel in norm_relation: # simple matching
        #         return standard_rel
        
        # Fallback: If relation type inference prompt is defined and needed:
        # prompt = self._get_prompt("kg_extraction", "relation_type_inference", relation=relation_name, context=context)
        # norm_relation = self._get_llm_response(prompt).strip().upper().replace(" ", "_")

        logger.debug(f"Normalized relation '{relation_name}' to '{norm_relation}'")
        return norm_relation

    def process_chunk(self, chunk_data: Dict, chunk_idx: int) -> List[Dict]:
        """
        Processes a single text chunk to extract structured knowledge graph triples.

        Args:
            chunk_data: A dictionary representing one line from the JSONL file 
                        (e.g., {"doc_id": "leed-v4", "section": "...", "text": "...", "page": 32}).
            chunk_idx: The index of the chunk, used for generating a unique chunk_id.

        Returns:
            A list of dictionaries, where each dictionary represents a structured triple
            ready for Neo4j integration.
            Format: {
                "head_name": "...", "head_type": "...", 
                "relation_type": "...", 
                "tail_name": "...", "tail_type": "...",
                "doc_id": "...", "chunk_id": "..."
            }
        """
        text_to_process = chunk_data.get("text")
        doc_id = chunk_data.get("doc_id", "unknown_doc")
        page_num = chunk_data.get("page", 0)
        section_info = chunk_data.get("section", "unknown_section") # For context if needed

        # Generate a unique chunk_id
        # Using chunk_idx (overall index from the JSONL file) for simplicity here.
        # If idx is per page/doc, adjust accordingly.
        # Let's assume chunk_idx is the global index of the chunk being processed.
        current_chunk_id = f"{doc_id}_page{page_num}_sec{slugify(section_info[:20])}_chunk{chunk_idx}"


        if not text_to_process or not text_to_process.strip():
            logger.info(f"Skipping empty text in chunk_id: {current_chunk_id}")
            return []

        logger.info(f"Processing chunk_id: {current_chunk_id}")
        raw_triples = self._extract_triples(text_to_process)
        processed_triples = []

        for raw_triple in raw_triples:
            head_entity_info = raw_triple.get("head_entity", {})
            tail_entity_info = raw_triple.get("tail_entity", {})
            relation_str = raw_triple.get("relation")

            head_name = head_entity_info.get("name", "").strip()
            # LLM might provide type directly, otherwise infer or default
            head_type = head_entity_info.get("type", "").strip() 
            if not head_type and head_name: # If LLM didn't provide type, try to infer
                 head_type = self._infer_entity_type(head_name, context=text_to_process)
            head_type = head_type.upper().replace(" ", "_") if head_type else "THING" # Default type

            tail_name = tail_entity_info.get("name", "").strip()
            tail_type = tail_entity_info.get("type", "").strip()
            if not tail_type and tail_name:
                tail_type = self._infer_entity_type(tail_name, context=text_to_process)
            tail_type = tail_type.upper().replace(" ", "_") if tail_type else "THING"

            if not head_name or not relation_str or not tail_name:
                logger.warning(f"Skipping incomplete triple from LLM for chunk {current_chunk_id}: {raw_triple}")
                continue

            # Normalize relation type
            relation_type = self._normalize_relation(relation_str, context=text_to_process)
            
            # Ensure relation type is not empty after normalization
            if not relation_type:
                logger.warning(f"Skipping triple with empty relation type after normalization for chunk {current_chunk_id}: {raw_triple}")
                continue

            processed_triples.append({
                "head_name": head_name,
                "head_type": head_type,
                "relation_type": relation_type,
                "tail_name": tail_name,
                "tail_type": tail_type,
                "doc_id": doc_id,
                "chunk_id": current_chunk_id,
                "source_text_chunk": text_to_process # Optional: include for traceability
            })
            
        logger.info(f"Extracted {len(processed_triples)} structured triples from chunk_id: {current_chunk_id}")
        return processed_triples