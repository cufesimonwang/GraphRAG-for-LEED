# GraphRAG-for-LEED

A GraphRAG (Graph-based Retrieval Augmented Generation) system for processing and analyzing LEED (Leadership in Energy and Environmental Design) documentation.

## Overview

This project implements a GraphRAG system specifically designed for LEED documentation. It combines the power of knowledge graphs with traditional RAG (Retrieval Augmented Generation) to provide more accurate and contextually relevant information retrieval and generation.

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
