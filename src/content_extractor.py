import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yaml
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from datetime import datetime
from pdfminer.high_level import extract_text as pdf_extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from anthropic import Anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

logger = logging.getLogger(__name__)

class ContentExtractor:
    """Unified content extractor for both RAG and GraphRAG applications."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the content extractor.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()
        
        # Initialize text splitter for RAG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunking', {}).get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunking', {}).get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vision handler if enabled
        if self.config.get('vision', {}).get('enabled', False):
            self.vision_handler = VisionModelHandler(self.config)
        else:
            self.vision_handler = None
        
        # Get debug settings
        self.debug_mode = self.config.get('debug', {}).get('enabled', False)
        self.debug_dir = Path(self.config.get('debug', {}).get('output_dir', './debug'))
        if self.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data directories
        self._init_directories()
        
        # Initialize LLM clients
        self._init_llm_clients()

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
            filename=log_config.get("log_file", "content_extractor.log")
        )

    def _init_directories(self):
        """Initialize directory structure for data processing."""
        # Get paths from config
        paths = self.config.get('paths', {})
        
        # Create main data directories
        self.data_dir = Path(paths.get('data_dir', 'data'))
        self.raw_dir = Path(paths.get('raw_dir', self.data_dir / 'raw'))
        self.processed_dir = Path(paths.get('processed_dir', self.data_dir / 'processed'))
        self.output_dir = Path(paths.get('output_dir', self.data_dir / 'output'))
        
        # Create subdirectories for different processing types
        self.rag_dir = Path(paths.get('rag_dir', self.processed_dir / 'rag'))
        self.graphrag_dir = Path(paths.get('graphrag_dir', self.processed_dir / 'graphrag'))
        
        # Create all directories
        directories = [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.output_dir,
            self.rag_dir,
            self.graphrag_dir,
            Path(paths.get('embeddings_dir', self.output_dir / 'embeddings')),
            Path(paths.get('logs_dir', 'logs')),
            Path(paths.get('debug_dir', 'debug'))
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Store output paths
        self.rag_output = Path(paths.get('rag_output', self.rag_dir / 'combined.jsonl'))
        self.graphrag_output = Path(paths.get('graphrag_output', self.graphrag_dir / 'combined.json'))
        self.faiss_index = Path(paths.get('faiss_index', self.output_dir / 'faiss.index'))

    def _init_llm_clients(self):
        """Initialize LLM clients for vision tasks."""
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            self.vision_client = OpenAI(api_key=self.config['llm']['api_key'])
        elif provider == 'anthropic':
            self.vision_client = Anthropic(api_key=self.config['llm']['api_key'])
        else:
            raise ValueError(f"Unsupported LLM provider for vision tasks: {provider}")

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

    def process_file(self, file_path: Path, mode: str = "hybrid") -> Dict:
        """
        Process a file and extract content for RAG and/or GraphRAG.
        
        Args:
            file_path: Path to the file to process
            mode: Processing mode ("rag", "graphrag", or "hybrid")
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{file_path.stem}_{timestamp}"
        
        # Extract content based on file type
        content = self._extract_content(file_path)
        
        # Process for RAG if needed
        if mode in ["rag", "hybrid"]:
            rag_content = self._process_for_rag(content, file_path)
            rag_output = self.rag_dir / f"{output_base}_rag.jsonl"
            self._save_jsonl(rag_content, rag_output)
            logger.info(f"Saved RAG content to {rag_output}")
        
        # Process for GraphRAG if needed
        if mode in ["graphrag", "hybrid"]:
            graphrag_content = self._process_for_graphrag(content, file_path)
            graphrag_output = self.graphrag_dir / f"{output_base}_graphrag.json"
            self._save_json(graphrag_content, graphrag_output)
            logger.info(f"Saved GraphRAG content to {graphrag_output}")
        
        return {
            'file_path': str(file_path),
            'mode': mode,
            'rag_output': str(rag_output) if mode in ["rag", "hybrid"] else None,
            'graphrag_output': str(graphrag_output) if mode in ["graphrag", "hybrid"] else None,
            'metadata': {
                'timestamp': timestamp,
                'file_type': file_path.suffix[1:],
                'file_size': file_path.stat().st_size
            }
        }

    def process_directory(self, directory_path: Path, output_format: str = "jsonl") -> List[Dict]:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory containing files to process
            output_format: Format of output ("jsonl" or "raw")
            
        Returns:
            List[Dict]: List of processed file results
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        results = []
        supported_extensions = {'.txt', '.pdf', '.docx', '.xlsx', '.csv'}
        
        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.process_file(file_path, "hybrid")
                    results.append(result)
                    logger.info(f"Successfully processed {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        return results

    def save_results(self, results: List[Dict], output_path: Path):
        """Save processing results to a JSONL file.
        
        Args:
            results: List of processing results
            output_path: Path to save the JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_path}")

    def extract_text(self, file_path: Path) -> str:
        """Extract text content from a file."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self._extract_from_txt(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.xlsx':
                return self._extract_from_excel(file_path)
            elif file_extension == '.csv':
                return self._extract_from_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    def _extract_content(self, file_path: Path) -> Dict:
        """Extract content from a file based on its type."""
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_type == '.txt':
            return self._extract_from_txt(file_path)
        elif file_type in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif file_type in ['.xlsx', '.xls']:
            return self._extract_from_excel(file_path)
        elif file_type == '.csv':
            return self._extract_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_from_txt(self, file_path: Path) -> Dict:
        """Extract text from a TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return {
            'text': [{
                'page': 1,
                'text': text
            }],
            'images': [],
            'metadata': {
                'file_type': 'txt',
                'encoding': 'utf-8'
            }
        }

    def _extract_from_pdf(self, file_path: Path) -> Dict:
        """Extract content from a PDF file."""
        content = {
            'text': [],
            'images': [],
            'metadata': {}
        }
        
        try:
            # Open PDF
            doc = fitz.open(file_path)
            
            # Extract metadata
            content['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'page_count': len(doc)
            }
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    content['text'].append({
                        'page': page_num + 1,
                        'text': text
                    })
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to base64
                    image_b64 = base64.b64encode(image_bytes).decode()
                    
                    # Get image position
                    image_rect = page.get_image_rects(xref)[0]
                    
                    content['images'].append({
                        'page': page_num + 1,
                        'index': img_index,
                        'data': image_b64,
                        'format': base_image["ext"],
                        'position': {
                            'x0': image_rect.x0,
                            'y0': image_rect.y0,
                            'x1': image_rect.x1,
                            'y1': image_rect.y1
                        }
                    })
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF {file_path}: {e}")
            raise

    def _extract_from_docx(self, file_path: Path) -> Dict:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        return {
            'text': [{'page': page_num + 1, 'text': paragraph.text} for page_num, paragraph in enumerate(doc.paragraphs)],
            'images': [],
            'metadata': {
                'file_type': 'docx',
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        }

    def _extract_from_excel(self, file_path: Path) -> Dict:
        """Extract data from Excel file and convert to text."""
        try:
            excel_file = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_parts.append(f"Sheet: {sheet_name}\n{df.to_string()}")
            
            return {
                'text': [{'page': 1, 'text': "\n\n".join(text_parts)}],
                'images': [],
                'metadata': {
                    'file_type': 'xlsx',
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {e}")
            return {
                'text': [{'page': 1, 'text': ''}],
                'images': [],
                'metadata': {
                    'file_type': 'xlsx',
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }

    def _extract_from_csv(self, file_path: Path) -> Dict:
        """Extract data from CSV file and convert to text."""
        try:
            df = pd.read_csv(file_path)
            return {
                'text': [{'page': 1, 'text': df.to_string()}],
                'images': [],
                'metadata': {
                    'file_type': 'csv',
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return {
                'text': [{'page': 1, 'text': ''}],
                'images': [],
                'metadata': {
                    'file_type': 'csv',
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }

    def _process_for_rag(self, content: Dict, file_path: Path) -> List[Dict]:
        """Process content for RAG by chunking text and preparing for embedding."""
        chunks = []
        chunk_size = self.config.get('extraction', {}).get('chunk_size', 1000)
        chunk_overlap = self.config.get('extraction', {}).get('chunk_overlap', 200)
        
        # Process text chunks
        for text_item in content['text']:
            text = text_item['text']
            page = text_item['page']
            
            # Split into chunks
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end > len(text):
                    end = len(text)
                else:
                    # Try to find a good break point
                    break_chars = ['. ', '! ', '? ', '\n']
                    for char in break_chars:
                        pos = text.rfind(char, start, end)
                        if pos != -1:
                            end = pos + len(char)
                            break
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(file_path),
                            'page': page,
                            'chunk_index': len(chunks),
                            'start_char': start,
                            'end_char': end
                        }
                    })
                
                start = end - chunk_overlap
        
        return chunks

    def _process_for_graphrag(self, content: Dict, file_path: Path) -> Dict:
        """Process content for GraphRAG by extracting entities and relationships."""
        # TODO: Implement GraphRAG processing
        # This should extract entities and relationships from the content
        # and prepare it for knowledge graph construction
        raise NotImplementedError("GraphRAG processing not implemented yet")

    def _save_jsonl(self, data: List[Dict], output_path: Path):
        """Save data as JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _save_json(self, data: Dict, output_path: Path):
        """Save data as JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

class VisionModelHandler:
    """Handler for different vision models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['vision']['model']
        self.max_tokens = self._get_model_max_tokens()
        
        # Initialize appropriate client based on model
        if 'gpt-4-vision' in self.model_name:
            openai.api_key = config['openai']['api_key']
            self.client = 'openai'
        elif 'claude' in self.model_name:
            self.client = Anthropic(api_key=config['anthropic']['api_key'])
        else:
            raise ValueError(f"Unsupported vision model: {self.model_name}")
        
        logger.info(f"Using vision model: {self.model_name} ({self.client})")
    
    def _get_model_max_tokens(self) -> int:
        """Get max tokens for the selected model."""
        for model in self.config['vision']['available_models']:
            if model['name'] == self.model_name:
                return model['max_tokens']
        return 500  # Default value
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def extract_relationships(self, image: Image.Image, prompt: str) -> Optional[str]:
        """Extract relationships from an image using the configured vision model."""
        try:
            base64_image = self._image_to_base64(image)
            
            if self.client == 'openai':
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
                
            elif isinstance(self.client, Anthropic):
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image
                                    }
                                }
                            ]
                        }
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in vision model extraction: {str(e)}")
            return None
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8') 