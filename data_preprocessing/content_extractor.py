import logging
import os
import json
import re
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
        self.jsonl_output_dir = Path(paths.get('jsonl_output_dir', self.processed_dir / 'jsonl_output'))
        
        # Create all directories
        directories = [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.output_dir,
            self.rag_dir,
            self.graphrag_dir,
            self.jsonl_output_dir,
            Path(paths.get('embeddings_dir', self.output_dir / 'embeddings')),
            Path(paths.get('logs_dir', 'logs')),
            Path(paths.get('debug_dir', 'debug'))
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Store output paths
        self.rag_output_path_template = str(self.rag_dir / "{output_base}_rag.jsonl") # Keep for now
        self.jsonl_output_path_template = str(self.jsonl_output_dir / "{file_stem}.jsonl")
        self.graphrag_output_path_template = str(self.graphrag_dir / "{output_base}_graphrag.json") # Keep for now
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

        # Create output filename base (used for some outputs, not the new JSONL)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_ts = f"{file_path.stem}_{timestamp}" # For outputs that need unique names per run
        
        logger.info(f"Step 1: Extracting content from {file_path} with section detection.")
        # _extract_content now returns {'text_blocks': [...], 'images': ..., 'metadata': ...}
        extracted_data = self._extract_content(file_path)
        logger.info(f"Raw content extraction complete. Found {len(extracted_data.get('text_blocks',[]))} text blocks.")

        # Initialize return dict
        processing_results = {
            'file_path': str(file_path),
            'mode': mode,
            'jsonl_output': None,
            'graphrag_output': None,
            'metadata': extracted_data.get('metadata', {}) # Use metadata from extraction
        }
        processing_results['metadata']['processing_timestamp'] = timestamp

        # Step 2: Process for JSONL output (Primary requirement)
        # This replaces the old RAG-specific processing for this output type.
        logger.info("Step 2: Processing content for JSONL output.")
        jsonl_formatted_objects = self._process_for_jsonl_output(extracted_data, file_path)
        
        # Save the JSONL output
        jsonl_output_path_str = self.jsonl_output_path_template.format(file_stem=file_path.stem)
        jsonl_output_path = Path(jsonl_output_path_str)
        self._save_jsonl(jsonl_formatted_objects, jsonl_output_path)
        logger.info(f"Saved JSONL output to {jsonl_output_path}")
        processing_results['jsonl_output'] = str(jsonl_output_path)

        # Step 3: Process for GraphRAG if needed
        if mode in ["graphrag", "hybrid"]:
            logger.info("Step 3: Preparing and processing content for GraphRAG.")
            # GraphRAG needs chunked text.
            # _chunk_text_for_graphrag now expects {'text_blocks': ...} from _extract_content
            # Its output is {'text': [{'text': chunk, 'page':..., 'original_block_section':...}, ...], 'metadata':...}
            graphrag_input_chunks = self._chunk_text_for_graphrag(extracted_data)
            
            # Ensure the metadata from original extraction is passed to _process_for_graphrag
            # graphrag_input_chunks already contains metadata from extracted_data
            
            if graphrag_input_chunks.get('text'): # Check if any text chunks were actually generated
                graphrag_extracted_elements = self._process_for_graphrag(graphrag_input_chunks, file_path)
                
                # Save GraphRAG output (e.g., entities and relations)
                graphrag_output_file_str = self.graphrag_output_path_template.format(output_base=output_base_ts)
                graphrag_output_path = Path(graphrag_output_file_str)
                self._save_json(graphrag_extracted_elements, graphrag_output_path) # Assuming _save_json saves dict as json
                logger.info(f"Saved GraphRAG elements to {graphrag_output_path}")
                processing_results['graphrag_output'] = str(graphrag_output_path)
            else:
                logger.warning(f"No chunks generated for GraphRAG from {file_path.name}. Skipping GraphRAG processing.")
        
        # Decision: The old _process_for_rag (renamed/refactored to _process_for_jsonl_output)
        # and its direct output (self.rag_dir / f"{output_base}_rag.jsonl")
        # is now handled by the JSONL output saved to self.jsonl_output_dir.
        # The specific old RAG output path self.rag_output_path_template is no longer used in this method.

        return processing_results

    def _chunk_text_for_graphrag(self, content: Dict) -> Dict:
        """Chunk text into smaller pieces for GraphRAG processing.
        
        Args:
            content: Dictionary containing extracted content, expected to have 'text_blocks'
            
        Returns:
            Dictionary with chunked text suitable for GraphRAG (list of dicts with 'text', 'page', 'chunk_index')
        """
        chunked_output_for_graphrag = {
            'text': [], # This will be a list of {'text': chunk_text, 'page': page_num, 'original_block_section': section, ...}
            'metadata': content.get('metadata', {})
        }
        
        text_blocks = content.get('text_blocks', [])
        if not text_blocks:
            logger.warning("No 'text_blocks' found in content for _chunk_text_for_graphrag")
            return chunked_output_for_graphrag

        # Get chunking parameters from config (can be specific for GraphRAG if needed)
        # Using general chunking config for now
        chunk_size = self.config.get('chunking', {}).get('chunk_size', 1000) 
        chunk_overlap = self.config.get('chunking', {}).get('chunk_overlap', 200)
        
        logger.info(f"Chunking {len(text_blocks)} text blocks for GraphRAG with size {chunk_size} and overlap {chunk_overlap}")
        
        for block_idx, block_item in enumerate(text_blocks):
            raw_text = block_item.get('raw_text', '')
            page_num = block_item.get('page_number', 0)
            section = block_item.get('detected_section', 'Unknown Section')

            if not raw_text.strip():
                logger.debug(f"Skipping empty raw_text in block {block_idx} for GraphRAG chunking.")
                continue
            
            # Split raw_text from the block into chunks
            chunks = self.text_splitter.split_text(raw_text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    chunked_output_for_graphrag['text'].append({
                        'text': chunk_text,
                        'page': page_num,
                        'original_block_section': section, # Carry over section info
                        'original_block_index': block_idx, 
                        'chunk_index_within_block': chunk_idx,
                        'total_chunks_in_block': len(chunks)
                    })
            
            logger.debug(f"Split block {block_idx} (page {page_num}, section '{section}') into {len(chunks)} chunks for GraphRAG")
        
        logger.info(f"Total chunks created for GraphRAG: {len(chunked_output_for_graphrag['text'])}")
        return chunked_output_for_graphrag

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

    def _is_section_title(self, line_text: str, line_font_size: float, avg_font_size: float, font_flags: int) -> bool:
        """Heuristics to identify if a line is a section title."""
        # Heuristic 1: Text is ALL CAPS (and not too short)
        if line_text.isupper() and len(line_text.split()) > 1 and len(line_text) > 5:
            logger.debug(f"Potential title (ALL CAPS): {line_text}")
            return True
        
        # Heuristic 2: Specific keywords (case-insensitive)
        section_keywords = ["SECTION", "PART", "CHAPTER", "APPENDIX", "MODULE", "UNIT"]
        if any(keyword in line_text.upper() for keyword in section_keywords) and len(line_text.split()) < 10:
             # Check if it matches common patterns like "SECTION 1.2" or "Part A"
            if re.match(r"^(SECTION|PART|CHAPTER|APPENDIX|MODULE|UNIT)\s+([A-Z0-9.-]+[:\s]?)", line_text.upper()):
                logger.debug(f"Potential title (Keyword Match): {line_text}")
                return True

        # Heuristic 3: Font size significantly larger than average
        # (Ensure avg_font_size is not zero to avoid division by zero or skewed ratios if page is empty)
        if avg_font_size > 0 and line_font_size > avg_font_size * 1.4:  # e.g., 40% larger
            logger.debug(f"Potential title (Font Size): {line_text} (Size: {line_font_size}, Avg: {avg_font_size})")
            return True

        # Heuristic 4: Bold font (font_flags & 2^4) - less reliable on its own, might combine
        # PyMuPDF font flags: 2^0=superscript, 2^1=italic, 2^2=serifed, 2^3=monospaced, 2^4=bold
        # if font_flags & 16: # Bold
        #     logger.debug(f"Potential title (Bold): {line_text}")
        #     return True
            
        return False

    def _extract_from_pdf(self, file_path: Path) -> Dict:
        """
        Extract content from a PDF file, attempting to identify sections.
        Returns a dictionary with 'text_blocks' (list of dicts with raw_text, page_number, detected_section),
        'images', and 'metadata'.
        """
        processed_text_blocks = []
        images_data = []
        metadata_dict = {}
        current_section_title = "Default Section" # Default if no title found initially

        try:
            doc = fitz.open(file_path)
            logger.info(f"Opened PDF {file_path} with {len(doc)} pages")

            metadata_dict = {
                'title': doc.metadata.get('title', file_path.stem),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'page_count': len(doc)
            }

            for page_num, page in enumerate(doc):
                page_content_blocks = []
                try:
                    # Extract text blocks with detailed information
                    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                except Exception as e:
                    logger.warning(f"Could not extract text blocks using 'dict' for page {page_num + 1} in {file_path}. Falling back to 'text'. Error: {e}")
                    # Fallback to simple text extraction for this page
                    simple_text = page.get_text()
                    if simple_text.strip():
                        processed_text_blocks.append({
                            'raw_text': simple_text,
                            'page_number': page_num + 1,
                            'detected_section': current_section_title
                        })
                    continue # Move to next page

                # Calculate average font size for the page (heuristic for title detection)
                font_sizes_on_page = []
                for block in blocks:
                    if block['type'] == 0: # Text block
                        for line in block['lines']:
                            for span in line['spans']:
                                font_sizes_on_page.append(span['size'])
                avg_font_size = sum(font_sizes_on_page) / len(font_sizes_on_page) if font_sizes_on_page else 0
                
                page_text_buffer = "" # Buffer to accumulate text under the current section on this page

                for block in blocks:
                    if block['type'] == 0:  # Text block
                        for line in block['lines']:
                            line_text_parts = []
                            # Assuming all spans in a line have similar font characteristics for title detection
                            line_font_size = line['spans'][0]['size'] if line['spans'] else avg_font_size
                            line_font_flags = line['spans'][0]['flags'] if line['spans'] else 0
                            
                            for span in line['spans']:
                                line_text_parts.append(span['text'])
                            line_text = "".join(line_text_parts).strip()

                            if not line_text:
                                continue

                            if self._is_section_title(line_text, line_font_size, avg_font_size, line_font_flags):
                                # If there's buffered text under the previous section, save it
                                if page_text_buffer.strip():
                                    page_content_blocks.append({
                                        'raw_text': page_text_buffer.strip(),
                                        'page_number': page_num + 1,
                                        'detected_section': current_section_title
                                    })
                                    page_text_buffer = "" # Reset buffer
                                current_section_title = line_text # Update current section title
                                # Also add the title itself as a block, associated with itself as section
                                page_content_blocks.append({
                                    'raw_text': current_section_title,
                                    'page_number': page_num + 1,
                                    'detected_section': current_section_title
                                })
                            else:
                                page_text_buffer += line_text + "\n" # Accumulate normal text
                
                # Add any remaining buffered text from the page
                if page_text_buffer.strip():
                    page_content_blocks.append({
                        'raw_text': page_text_buffer.strip(),
                        'page_number': page_num + 1,
                        'detected_section': current_section_title
                    })
                
                processed_text_blocks.extend(page_content_blocks)
                logger.debug(f"Extracted {len(page_content_blocks)} text blocks from page {page_num + 1}")

                # Extract images (keeping existing image extraction logic)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode()
                    
                    # Attempt to get image position robustly
                    img_rects = page.get_image_rects(xref)
                    image_rect_dict = {}
                    if img_rects:
                        image_rect = img_rects[0] # Take the first rectangle
                        image_rect_dict = {'x0': image_rect.x0, 'y0': image_rect.y0, 'x1': image_rect.x1, 'y1': image_rect.y1}

                    images_data.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'data': image_b64,
                        'format': base_image["ext"],
                        'position': image_rect_dict
                    })
                    logger.debug(f"Extracted image {img_index + 1} from page {page_num + 1}")
            
            doc.close()
            logger.info(f"PDF extraction complete for {file_path}. Extracted {len(processed_text_blocks)} text blocks and {len(images_data)} images.")
            return {
                'text_blocks': processed_text_blocks, # Changed key name
                'images': images_data,
                'metadata': metadata_dict
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF {file_path}: {e}")
            # Ensure doc is closed if opened
            if 'doc' in locals() and doc:
                doc.close()
            # Return empty structure on error to prevent downstream issues
            return {
                'text_blocks': [],
                'images': [],
                'metadata': {'title': file_path.stem, 'page_count': 0, 'error': str(e)}
            }

    def _extract_from_docx(self, file_path: Path) -> Dict:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        # For docx, we don't have easy page numbers or section detection like PDF.
        # We'll treat each paragraph as a block and assign a default section.
        # Page numbers are not directly available in python-docx.
        text_blocks = []
        current_section = "Default Section" # Could be improved with heuristics later
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                 # Simple heuristic: if paragraph style suggests a heading
                if paragraph.style.name.startswith('Heading'):
                    current_section = paragraph.text.strip()
                text_blocks.append({
                    'raw_text': paragraph.text,
                    'page_number': i + 1, # Placeholder, not a real page number
                    'detected_section': current_section
                })
        return {
            'text_blocks': text_blocks,
            'images': [], # Image extraction from docx can be added if needed
            'metadata': {
                'file_type': 'docx',
                'title': file_path.stem,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        }

    def _extract_from_excel(self, file_path: Path) -> Dict:
        """Extract data from Excel file and convert to text."""
        try:
            excel_file = pd.ExcelFile(file_path)
            text_blocks = []
            current_section = "Default Section" # Default for excel sheets
            
            for i, sheet_name in enumerate(excel_file.sheet_names):
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Consider each sheet a "section" for simplicity
                current_section = f"Sheet: {sheet_name}"
                sheet_text = df.to_string()
                if sheet_text.strip():
                    text_blocks.append({
                        'raw_text': sheet_text,
                        'page_number': i + 1, # Sheet number as page number
                        'detected_section': current_section
                    })
            
            return {
                'text_blocks': text_blocks,
                'images': [],
                'metadata': {
                    'file_type': 'xlsx',
                    'title': file_path.stem,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {e}")
            return {
                'text_blocks': [],
                'images': [],
                'metadata': {
                    'file_type': 'xlsx',
                    'title': file_path.stem,
                    'error': str(e),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }

    def _extract_from_csv(self, file_path: Path) -> Dict:
        """Extract data from CSV file and convert to text."""
        try:
            df = pd.read_csv(file_path)
            csv_text = df.to_string()
            text_blocks = []
            if csv_text.strip():
                text_blocks.append({
                    'raw_text': csv_text,
                    'page_number': 1, # CSVs are single page
                    'detected_section': "CSV Data" # Default section for CSV
                })
            return {
                'text_blocks': text_blocks,
                'images': [],
                'metadata': {
                    'file_type': 'csv',
                    'title': file_path.stem,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return {
                'text_blocks': [],
                'images': [],
                'metadata': {
                    'file_type': 'csv',
                    'title': file_path.stem,
                    'error': str(e),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }

    def _process_for_jsonl_output(self, extracted_content: Dict, file_path: Path) -> List[Dict]:
        """
        Process extracted text blocks for JSONL output, chunking text and formatting.
        """
        logger.info(f"Starting JSONL processing for {file_path.name}")
        output_json_objects = []
        doc_id = file_path.stem

        # The key for text blocks is now 'text_blocks'
        text_blocks = extracted_content.get('text_blocks', [])
        if not text_blocks:
            logger.warning(f"No text blocks found in extracted content for {file_path.name}")
            return []

        total_text_blocks = len(text_blocks)
        logger.info(f"Processing {total_text_blocks} text blocks for {doc_id}")

        for i, block in enumerate(text_blocks):
            raw_text = block.get('raw_text', '')
            page_num = block.get('page_number', 0)
            section = block.get('detected_section', 'Unknown Section')

            if not raw_text.strip():
                logger.debug(f"Skipping empty text block {i+1}/{total_text_blocks} from page {page_num}")
                continue
            
            try:
                text_chunks = self.text_splitter.split_text(raw_text)
                logger.debug(f"Split block {i+1} (page {page_num}, section '{section}') into {len(text_chunks)} chunks")
                
                for chunk_index, chunk_text in enumerate(text_chunks):
                    if not chunk_text.strip():
                        logger.debug(f"Skipping empty chunk {chunk_index} from block {i+1}")
                        continue
                    
                    output_json_objects.append({
                        "doc_id": doc_id,
                        "section": section,
                        "text": chunk_text,
                        "page": page_num 
                    })
                
            except Exception as e:
                logger.error(f"Error processing text block {i+1} from page {page_num} for {doc_id}: {e}")
                continue
        
        logger.info(f"JSONL processing complete for {doc_id}. Created {len(output_json_objects)} JSON objects.")
        return output_json_objects

    def _process_for_graphrag(self, content: Dict, file_path: Path) -> Dict:
        """Process content for GraphRAG by extracting entities and relationships."""
        # Initialize result structure
        result = {
            'entities': [],
            'relations': [],
            'metadata': content.get('metadata', {}) # Ensure this metadata comes from _extract_content
        }
        
        # The 'content' for GraphRAG now needs to be adapted from 'text_blocks' if that's the primary source
        # Or it might use a different chunking strategy.
        # For now, assuming 'content' passed here is already chunked appropriately for GraphRAG
        # This part might need adjustment based on how _chunk_text_for_graphrag is called and what it expects.
        # If _chunk_text_for_graphrag expects {'text': [{'text': ..., 'page': ...}]}, we need to adapt.
        
        # Let's assume 'content' here refers to the output of _chunk_text_for_graphrag,
        # which itself would need to be adapted to take 'text_blocks' as input.
        
        text_chunks_for_graphrag = content.get('text', []) # Assuming this is already prepared list of text strings or dicts
        if not text_chunks_for_graphrag:
            logger.warning(f"No text chunks available for GraphRAG processing for {file_path.name}")
            return result

        total_chunks = len(text_chunks_for_graphrag)
        logger.info(f"Starting GraphRAG processing with {total_chunks} text chunks for {file_path.name}")
        
        # Get LLM configuration
        llm_config = self.config.get('llm', {})
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'openai')
        model_name = llm_config.get('model_name', 'gpt-4-turbo-preview')
        api_key = llm_config.get('api_key')
        
        logger.info(f"Using LLM provider: {provider}, model: {model_name} for GraphRAG")
        
        # Initialize LLM client based on provider - This should use self.vision_client or a similar dedicated client
        # For simplicity, reusing the vision_client initialization logic if applicable,
        # or a new client specific for GraphRAG's LLM needs to be initialized.
        # Let's assume self.vision_client is suitable or a similar client is available as self.graphrag_llm_client
        
        # Re-initialize a client here for clarity, or ensure one is available.
        # This part of the code might be simplified if _init_llm_clients already sets up a general purpose LLM client.
        # For now, let's assume a client is available (e.g., self.vision_client can be used or a new one for text gen)
        # If not, this would be:
        if provider == 'openai':
            graph_llm_client = OpenAI(api_key=api_key)
        elif provider == 'anthropic':
            graph_llm_client = Anthropic(api_key=api_key)
        else:
            # Potentially add local model support here if different from vision
            raise ValueError(f"Unsupported LLM provider for GraphRAG: {provider}")

        for i, text_item_dict in enumerate(text_chunks_for_graphrag, 1):
            # Assuming text_item_dict is {'text': chunk_text, 'page': page_num, ...}
            text_for_extraction = text_item_dict.get('text', '')
            page_num_of_chunk = text_item_dict.get('page', 0) # Get page number from the chunk itself
            original_chunk_index = text_item_dict.get('chunk_index', 'N/A') # if available

            if not text_for_extraction.strip():
                logger.debug(f"Skipping empty chunk {i}/{total_chunks} for GraphRAG.")
                continue
            
            logger.info(f"Processing chunk {i}/{total_chunks} for GraphRAG (Page: {page_num_of_chunk}, Chunk Index: {original_chunk_index})")

            try:
                logger.debug(f"Making API call to {provider} for GraphRAG chunk {i}")
                
                if provider == 'openai':
                    response = graph_llm_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting entities and relationships from text. Format the output as JSON with 'entities' and 'relations' arrays. Each entity should have a 'type' and 'name' field. Each relation should have 'source', 'target', and 'type' fields."}, # Ensure prompt is good
                            {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{text_for_extraction}"}
                        ],
                        response_format={"type": "json_object"}, # Requires compatible OpenAI model
                        temperature=0.1 # Low temperature for factual extraction
                    )
                    extraction = json.loads(response.choices[0].message.content)
                
                elif provider == 'anthropic':
                    response = graph_llm_client.messages.create(
                        model=model_name, # Ensure this is a Claude model name
                        max_tokens=1024, # Or other appropriate value
                        messages=[
                            {
                                "role": "user",
                                "content": f"You are an expert at extracting entities and relationships from text. Extract entities and their relationships from the following text. Format the output as JSON with 'entities' and 'relations' arrays. Each entity should have a 'type' and 'name' field. Each relation should have 'source', 'target', and 'type' fields.\n\nText to analyze:\n{text_for_extraction}"
                            }
                        ],
                        temperature=0.1
                    )
                    extraction = json.loads(response.content[0].text) # Anthropic's response structure

                # Add source information to entities and relations
                extracted_entities = extraction.get('entities', [])
                extracted_relations = extraction.get('relations', [])
                logger.info(f"Extracted {len(extracted_entities)} entities and {len(extracted_relations)} relations from GraphRAG chunk {i}")
                
                for entity in extracted_entities:
                    entity['source_info'] = { # Changed from 'source' to 'source_info' to avoid conflict if 'source' is a field in the entity itself
                        'file': str(file_path),
                        'page': page_num_of_chunk, 
                        'original_chunk_index': original_chunk_index 
                    }
                    result['entities'].append(entity)
                
                for relation in extracted_relations:
                    relation['source_info'] = {
                        'file': str(file_path),
                        'page': page_num_of_chunk,
                        'original_chunk_index': original_chunk_index
                    }
                    result['relations'].append(relation)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response for GraphRAG chunk {i} from {file_path.name}: {e}. Response: {response.choices[0].message.content if provider == 'openai' else response.content[0].text}")
                continue
            except Exception as e:
                logger.error(f"Error extracting entities and relations from GraphRAG chunk {i} from {file_path.name}: {e}")
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