# GraphRAG for LEED

A knowledge graph-based RAG system for LEED documentation.

## Project Structure

### Knowledge Graph Module (`src/kg/`)

The knowledge graph module is organized into several components:

1. `kg_builder.py` - High-level orchestrator

   - Coordinates the entire knowledge graph construction pipeline
   - Manages file processing and output generation
   - Handles configuration and logging
   - Integrates with content extraction and graph construction

2. `kg_extractor.py` - Core extraction logic

   - Uses LLMs to extract entities and relationships from text
   - Handles entity type inference
   - Manages relation extraction and normalization
   - Supports multiple LLM providers (OpenAI, Anthropic, local models)

3. `graph_manager.py` - Graph operations and management

   - Handles graph construction and manipulation
   - Provides graph analysis and statistics
   - Manages graph persistence (save/load)
   - Offers subgraph extraction and entity relation queries

4. `graph_visualizer.py` - Graph visualization utilities

   - Handles graph visualization in various formats (JSON, HTML, PNG, GraphML)
   - Manages graph layout and styling
   - Provides visualization configuration options

5. `__init__.py` - Package initialization
   - Exports main classes
   - Sets up package-level configuration

## Usage

```python
from src.kg import KnowledgeGraphBuilder

# Initialize the builder
builder = KnowledgeGraphBuilder(config_path="config/config.yaml")

# Build the knowledge graph
result = builder.build_knowledge_graph()

# Access the results
entities = result['entities']
relations = result['relations']
chunks = result['chunks']  # For RAG/hybrid mode
```

## Configuration

The system is configured through YAML files:

- `config/config.yaml` - Main configuration
- `config/prompts.yaml` - LLM prompts for extraction

## Output Formats

The knowledge graph can be saved in multiple formats:

- JSON - For data exchange
- HTML - Interactive visualization
- PNG - Static visualization
- GraphML - For graph analysis tools

## Features

- Knowledge graph construction from LEED documentation
- Hybrid retrieval combining graph-based and vector-based approaches
- Support for multiple document formats (PDF, DOCX, TXT)
- Configurable LLM integration (OpenAI, Anthropic, Local models)
- Interactive visualization of knowledge graphs
- Comprehensive test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/cufesimonwang/GraphRAG-for-LEED.git
cd GraphRAG-for-LEED

# Install the package
pip install -e .
```

## Configuration

1. Copy the example config file:

```bash
cp config/config.yaml.example config/config.yaml
```

2. Update the configuration with your settings:

- Add your OpenAI API key
- Configure the processing mode (rag, graphrag, or hybrid)
- Set up paths and other parameters

## Usage

```bash
# Process files using the GraphRAG pipeline
python src/main.py --config config.yaml --mode graphrag --input ./data/raw --output ./data/processed
```

## Project Structure

```
GraphRAG-for-LEED/
├── config/             # Configuration files
├── data/              # Data directories
│   ├── raw/          # Input files
│   ├── processed/    # Intermediate files
│   └── output/       # Final outputs
├── src/              # Source code
│   ├── kg/          # Knowledge graph components
│   ├── retriever/   # Retrieval components
│   └── generator/   # Generation components
├── tests/            # Test files
├── notebooks/        # Jupyter notebooks
└── docs/            # Documentation
```

## Development

```bash
# Run tests
pytest tests/

# Install development dependencies
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- LEED documentation and standards
- OpenAI for GPT models
- NetworkX for graph operations
- LangChain for RAG components

## Author

Simon Wang

- Email: cufesimonwang@gmail.com

## CLI Usage

The system can be run using the following command:

```bash
# Process a single file
python -m src.main --input "path/to/your/file.pdf" --output "path/to/output" --mode "hybrid"

# Process all files in a directory
python -m src.main --input "path/to/input/directory" --output "path/to/output" --mode "hybrid"
```

### Command Line Arguments

- `--input`: Path to input file or directory (required)
- `--output`: Path to output directory (optional)
- `--mode`: Processing mode (optional, default: "hybrid")
  - `rag`: Only perform RAG processing
  - `graphrag`: Only build knowledge graph
  - `hybrid`: Perform both RAG and knowledge graph processing
- `--config`: Path to config file (optional, default: "config/config.yaml")
- `--log-level`: Logging level (optional, default: "INFO")
  - Choices: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

### Example

```bash
# Process a single PDF file
python -m src.main --input "data/raw/GA01.pdf" --output "data/processed" --mode "hybrid"

# Process all PDFs in a directory
python -m src.main --input "data/raw" --output "data/processed" --mode "hybrid"
```
