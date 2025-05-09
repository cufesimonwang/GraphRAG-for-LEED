# GraphRAG for LEED

A knowledge graph-based Retrieval Augmented Generation (GraphRAG) system for LEED certification materials. This system processes LEED documentation to build a comprehensive knowledge graph, enabling intelligent question answering about LEED requirements and processes.

## 🌟 Features

### Content Processing

- **Multi-format Support**: Process PDFs, Excel files, CSV, DOCX, and TXT files
- **Vision-based Extraction**: Extract relationships from diagrams using GPT-4-Vision
- **Text & Table Processing**: Extract structured information from text and tables
- **Configurable Prompts**: Customize extraction and inference prompts via YAML

### Knowledge Graph Construction

- **Entity Extraction**: Identify LEED credits, prerequisites, and concepts
- **Relation Extraction**: Extract relationships between entities using LLMs
- **Graph Management**: Build, visualize, and export knowledge graphs
- **Multiple Export Formats**: JSON, JSONL, CSV, TTL, GraphML, HTML, and PNG visualizations
- **Configurable Colors**: Customize entity colors and themes

### Retrieval & Generation

- **Hybrid Retrieval**: Combine RAG and GraphRAG approaches for optimal results
- **Context-aware Generation**: Generate responses with proper context
- **Source Attribution**: Include source information in responses
- **Diagnostic Mode**: Enable raw triple extraction for debugging

## 📁 Project Structure

```
.
├── src/
│   ├── kg/                    # Knowledge Graph components
│   │   ├── kg_builder.py     # Main KG builder
│   │   ├── kg_extractor.py   # Entity/relation extraction
│   │   └── graph_manager.py  # Graph operations
│   ├── retriever/            # Retrieval components
│   │   └── retriever.py      # Hybrid retriever implementation
│   ├── generator/            # Generation components
│   │   └── generator.py      # Response generator
│   └── content_extractor.py  # Content extraction
├── config/                   # Configuration files
│   ├── config.yaml          # Main configuration
│   └── prompts.yaml         # Prompt templates
├── data/                    # Data directories
│   ├── raw/                # Original input files
│   ├── processed/          # Intermediate processed files
│   │   ├── rag/           # RAG processed files
│   │   └── graphrag/      # GraphRAG processed files
│   └── output/            # Final output files
├── models/                 # Local model storage
├── logs/                  # Log files
└── debug/                # Debug outputs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for GPT-4 and GPT-4-Vision)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/GraphRAG-for-LEED.git
cd GraphRAG-for-LEED
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the system:
   - Copy `config.example.yaml` to `config.yaml`
   - Add your OpenAI API key
   - Adjust other settings as needed
   - Customize prompts in `prompts.yaml` if desired

### Usage

1. Place your LEED documents in the `data/raw/` directory:

```bash
cp /path/to/your/leed/documents/* data/raw/
```

2. Process the documents:

```bash
python src/content_extractor.py
```

3. Build the knowledge graph:

```bash
python src/kg/kg_builder.py
```

4. Run the QA system:

```bash
python src/main.py --query "What are the requirements for LEED Energy and Atmosphere credits?"
```

## ⚙️ Configuration

The system is configured through two main YAML files:

### config.yaml

```yaml
# Mode Configuration
mode: "hybrid" # Options: "rag", "graphrag", "hybrid"

# Paths Configuration
paths:
  data_dir: "./data"
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"
  output_dir: "./data/output"
  rag_dir: "./data/processed/rag"
  graphrag_dir: "./data/processed/graphrag"
  model_dir: "./models"
  logs_dir: "./logs"

# LLM Configuration
llm:
  provider: "openai"
  model_name: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 1000
```

### prompts.yaml

```yaml
# Knowledge Graph Extraction Prompts
kg_extraction:
  default_prompt: |
    Extract entities and their relationships from the text...
  structured_triple_prompt: |
    Extract entities and their relationships in the form of structured triples...

# Entity Type Inference Prompts
entity_inference:
  default_prompt: |
    Analyze the entity and determine its type...

# Relation Type Inference Prompts
relation_inference:
  default_prompt: |
    Analyze the relationship between entities...
```

## 🔧 Development

### Adding New Features

1. **New Content Types**:

   - Add support in `content_extractor.py`
   - Update configuration in `config.yaml`

2. **Custom Prompts**:

   - Add new prompts to `prompts.yaml`
   - Update prompt handling in relevant modules

3. **Graph Visualization**:
   - Customize colors in `config.yaml`
   - Add new export formats in `graph_manager.py`

### Testing

Run tests with:

```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{graphrag_leed,
  author = {Your Name},
  title = {GraphRAG for LEED},
  year = {2024},
  url = {https://github.com/yourusername/GraphRAG-for-LEED}
}
```
