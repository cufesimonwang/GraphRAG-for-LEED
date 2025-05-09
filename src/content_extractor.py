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
            chunk_size=self.config.get('chunking', {}).get('chunk_size', 500),
            chunk_overlap=self.config.get('chunking', {}).get('chunk_overlap', 100),
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ": ",
                ", ",
                " ",
                ""
            ]
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
        
        logger.info(f"Step 1: Extracting raw content from {file_path}")
        # Extract content based on file type
        content = self._extract_content(file_path)
        logger.info(f"Raw content extraction complete. Found {len(content['text'])} text chunks")
        
        # Process for RAG if needed
        if mode in ["rag", "hybrid"]:
            logger.info("Step 2: Processing content for RAG")
            rag_content = self._process_for_rag(content, file_path)
            rag_output = self.rag_dir / f"{output_base}_rag.jsonl"
            self._save_jsonl(rag_content, rag_output)
            logger.info(f"Saved RAG content to {rag_output}")
        
        # Process for GraphRAG if needed
        if mode in ["graphrag", "hybrid"]:
            logger.info("Step 3: Processing content for GraphRAG")
            # First chunk the text into smaller pieces
            chunked_content = self._chunk_text_for_graphrag(content)
            logger.info(f"Text chunking complete. Created {len(chunked_content['text'])} chunks")
            
            # Then extract entities and relations
            graphrag_content = self._process_for_graphrag(chunked_content, file_path)
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

    def _chunk_text_for_graphrag(self, content: Dict) -> Dict:
        """Chunk text into smaller pieces for GraphRAG processing.
        
        Args:
            content: Dictionary containing extracted content
            
        Returns:
            Dictionary with chunked text
        """
        chunked_content = {
            'text': [],
            'metadata': content.get('metadata', {})
        }
        
        # Get chunking parameters from config
        chunk_size = self.config.get('chunking', {}).get('chunk_size', 1000)
        chunk_overlap = self.config.get('chunking', {}).get('chunk_overlap', 200)
        
        logger.info(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
        
        # Process each text item
        for text_item in content['text']:
            text = text_item['text']
            page = text_item['page']
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Add each chunk with metadata
            for i, chunk in enumerate(chunks):
                chunked_content['text'].append({
                    'text': chunk,
                    'page': page,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
            
            logger.debug(f"Split page {page} into {len(chunks)} chunks")
        
        return chunked_content

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
            logger.info(f"Opened PDF with {len(doc)} pages")
            
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
                    logger.debug(f"Extracted text from page {page_num + 1}")
                
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
                    logger.debug(f"Extracted image {img_index + 1} from page {page_num + 1}")
            
            logger.info(f"PDF extraction complete. Extracted {len(content['text'])} text chunks and {len(content['images'])} images")
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
        logger.info("Starting RAG processing")
        chunks = []
        chunk_size = self.config.get('extraction', {}).get('chunk_size', 500)
        chunk_overlap = self.config.get('extraction', {}).get('chunk_overlap', 100)
        
        logger.info(f"Using chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Process text chunks
        total_text_items = len(content['text'])
        logger.info(f"Processing {total_text_items} text items")
        
        for i, text_item in enumerate(content['text'], 1):
            text = text_item['text']
            page = text_item['page']
            
            logger.info(f"Processing text item {i}/{total_text_items} from page {page}")
            
            if not text.strip():
                logger.debug(f"Skipping empty text item {i}")
                continue
            
            try:
                # Split into chunks using the text splitter
                text_chunks = self.text_splitter.split_text(text)
                logger.info(f"Split text item {i} into {len(text_chunks)} chunks")
                
                for j, chunk in enumerate(text_chunks):
                    if not chunk.strip():
                        logger.debug(f"Skipping empty chunk {j} from text item {i}")
                        continue
                        
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(file_path),
                            'page': page,
                            'chunk_index': j,
                            'total_chunks': len(text_chunks),
                            'text_item_index': i,
                            'total_text_items': total_text_items
                        }
                    })
                    logger.debug(f"Added chunk {j+1}/{len(text_chunks)} from text item {i}")
                
            except Exception as e:
                logger.error(f"Error processing text item {i}: {e}")
                continue
        
        logger.info(f"RAG processing complete. Created {len(chunks)} total chunks")
        return chunks

    def _process_for_graphrag(self, content: Dict, file_path: Path) -> Dict:
        """Process content for GraphRAG by extracting entities and relationships."""
        # Initialize result structure
        result = {
            'entities': [],
            'relations': [],
            'metadata': content.get('metadata', {})
        }
        
        total_chunks = len(content['text'])
        logger.info(f"Starting GraphRAG processing with {total_chunks} text chunks")
        
        # Get LLM configuration
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'openai')
        model_name = llm_config.get('model_name', 'gpt-4-turbo-preview')
        api_key = llm_config.get('api_key')
        
        logger.info(f"Using LLM provider: {provider}, model: {model_name}")
        
        # Initialize LLM client based on provider
        if provider == 'openai':
            client = OpenAI(api_key=api_key)
        elif provider == 'anthropic':
            client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Process each text chunk
        for i, text_item in enumerate(content['text'], 1):
            text = text_item['text']
            page = text_item['page']
            chunk_index = text_item.get('chunk_index', i)
            total_chunks_in_page = text_item.get('total_chunks', 1)
            
            logger.info(f"Processing chunk {i}/{total_chunks} (chunk {chunk_index}/{total_chunks_in_page} from page {page})")
            
            # Skip empty chunks
            if not text.strip():
                logger.debug(f"Skipping empty chunk {i}")
                continue
            
            # Extract entities and relations using LLM
            try:
                logger.debug(f"Making API call to {provider} for chunk {i}")
                
                if provider == 'openai':
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting entities and relationships from text. Extract entities and their relationships from the following text. Format the output as JSON with 'entities' and 'relations' arrays. Each entity should have a 'type' and 'name' field. Each relation should have 'source', 'target', and 'type' fields."},
                            {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{text}"}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    extraction = json.loads(response.choices[0].message.content)
                
                elif provider == 'anthropic':
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=1024,
                        messages=[
                            {
                                "role": "user",
                                "content": f"You are an expert at extracting entities and relationships from text. Extract entities and their relationships from the following text. Format the output as JSON with 'entities' and 'relations' arrays. Each entity should have a 'type' and 'name' field. Each relation should have 'source', 'target', and 'type' fields.\n\nText to analyze:\n{text}"
                            }
                        ],
                        temperature=0.1
                    )
                    extraction = json.loads(response.content[0].text)
                
                # Add source information to entities and relations
                entities_count = len(extraction.get('entities', []))
                relations_count = len(extraction.get('relations', []))
                logger.info(f"Extracted {entities_count} entities and {relations_count} relations from chunk {i}")
                
                for entity in extraction.get('entities', []):
                    entity['source'] = {
                        'file': str(file_path),
                        'page': page,
                        'chunk_index': chunk_index
                    }
                    result['entities'].append(entity)
                
                for relation in extraction.get('relations', []):
                    relation['source'] = {
                        'file': str(file_path),
                        'page': page,
                        'chunk_index': chunk_index
                    }
                    result['relations'].append(relation)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response for chunk {i}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error extracting entities and relations from chunk {i}: {e}")
                continue
        
        logger.info(f"Completed GraphRAG processing. Total entities: {len(result['entities'])}, Total relations: {len(result['relations'])}")
        return result

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