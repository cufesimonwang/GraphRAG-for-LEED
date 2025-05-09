import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import yaml
from openai import OpenAI
from anthropic import Anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ..retriever.retriever import HybridRetriever

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generator that creates responses based on retrieved information."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the response generator."""
        # Load configuration
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()
        
        # Initialize retriever
        self.retriever = HybridRetriever(config_path)
        
        # Initialize LLM
        self._init_llm()
    
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
        try:
            prompt = self.prompts[prompt_type][prompt_name]
            return prompt.format(**kwargs)
        except KeyError:
            logger.warning(f"Prompt {prompt_type}.{prompt_name} not found, using default")
            return self._get_default_prompts().get(prompt_type, "")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return self._get_default_prompts().get(prompt_type, "")
    
    def _format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into context for the LLM."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"Source {i} (Score: {result['score']:.3f}):")
            context_parts.append(result['text'])
            if 'metadata' in result:
                context_parts.append(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate(self, query: str, top_k: int = None) -> Dict:
        """
        Generate a response based on the query and retrieved information.
        
        Args:
            query: The user's query
            top_k: Number of results to retrieve (default from config)
            
        Returns:
            Dictionary containing:
            - response: Generated response text
            - sources: List of sources used
            - metadata: Additional information
        """
        # Retrieve relevant information
        results = self.retriever.retrieve(query, top_k)
        
        # Format context from results
        context = self._format_context(results)
        
        # Get generation prompt
        prompt = self._get_prompt(
            'generation',
            'response_generation',
            query=query,
            context=context
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
            
            # Prepare sources and metadata
            sources = []
            for result in results:
                source = {
                    'text': result['text'][:200] + "...",  # Truncate long texts
                    'score': result['score'],
                    'source': result['source']
                }
                if 'metadata' in result:
                    source['metadata'] = result['metadata']
                sources.append(source)
            
            return {
                'response': response_text,
                'sources': sources,
                'metadata': {
                    'num_sources': len(sources),
                    'mode': self.retriever.mode,
                    'query': query
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I encountered an error while generating the response.",
                'sources': [],
                'metadata': {
                    'error': str(e),
                    'query': query
                }
            }

def main():
    """Example usage of the ResponseGenerator."""
    # Initialize generator
    generator = ResponseGenerator()
    
    # Example query
    query = "What are the requirements for LEED certification?"
    
    # Generate response
    result = generator.generate(query)
    
    # Print response
    print("\nResponse:")
    print(result['response'])
    
    print("\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\nSource {i}:")
        print(f"Score: {source['score']:.3f}")
        print(f"Source: {source['source']}")
        print(f"Text: {source['text']}")
        if 'metadata' in source:
            print(f"Metadata: {source['metadata']}")

if __name__ == "__main__":
    main() 