import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.content_extractor import ContentExtractor
from src.kg.kg_extractor import KnowledgeGraphExtractor
from src.kg.graph_constructor import GraphConstructor

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "subject": "LEED Credit",
        "predicate": "requires",
        "object": "Energy Performance"
    }

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    LEED Credit requires Energy Performance.
    Energy Performance contributes to Sustainable Design.
    Sustainable Design is part of Green Building.
    """

@pytest.fixture
def config_path():
    """Path to test configuration file."""
    return "config/test_config.yaml"

def test_content_extractor_process_file(tmp_path):
    """Test content extractor's process_file method."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Initialize content extractor with mock config
    with patch('src.content_extractor.ContentExtractor._load_config') as mock_load_config:
        mock_load_config.return_value = {
            'paths': {'data_dir': str(tmp_path)},
            'llm': {'provider': 'openai', 'model_name': 'gpt-4'}
        }
        extractor = ContentExtractor("config.yaml")
        
        # Process the file
        result = extractor.process_file(test_file, mode="hybrid")
        
        # Verify the result
        assert result['file_path'] == str(test_file)
        assert result['mode'] == "hybrid"
        assert 'metadata' in result

def test_kg_extractor_process(sample_text, mock_llm_response):
    """Test knowledge graph extractor's process method."""
    # Initialize KG extractor with mock config
    with patch('src.kg.kg_extractor.KnowledgeGraphExtractor._load_config') as mock_load_config, \
         patch('src.kg.kg_extractor.KnowledgeGraphExtractor._get_llm_response') as mock_llm:
        
        mock_load_config.return_value = {
            'llm': {'provider': 'openai', 'model_name': 'gpt-4'},
            'extraction': {'relation_types': ['requires', 'contributes_to']}
        }
        mock_llm.return_value = json.dumps([mock_llm_response])
        
        extractor = KnowledgeGraphExtractor("config.yaml")
        
        # Process the text
        entities, relations = extractor.process(sample_text)
        
        # Verify the results
        assert len(entities) > 0
        assert len(relations) > 0
        assert all('id' in entity for entity in entities)
        assert all('type' in entity for entity in entities)
        assert all('source' in relation for relation in relations)
        assert all('target' in relation for relation in relations)

def test_graph_constructor_get_graph_stats():
    """Test graph constructor's get_graph_stats method."""
    # Create a sample graph
    graph = {
        'nodes': [
            {'id': '1', 'type': 'Credit', 'label': 'LEED Credit'},
            {'id': '2', 'type': 'Requirement', 'label': 'Energy Performance'}
        ],
        'edges': [
            {'source': '1', 'target': '2', 'type': 'requires'}
        ]
    }
    
    # Initialize graph constructor with mock config
    with patch('src.kg.graph_constructor.GraphConstructor._load_config') as mock_load_config:
        mock_load_config.return_value = {}
        constructor = GraphConstructor("config.yaml")
        
        # Get graph statistics
        stats = constructor.get_graph_stats(graph)
        
        # Verify the statistics
        assert stats['nodes'] == 2
        assert stats['edges'] == 1
        assert len(stats['node_types']) == 2
        assert len(stats['edge_types']) == 1
        assert 'Credit' in stats['node_types']
        assert 'requires' in stats['edge_types']

if __name__ == '__main__':
    pytest.main([__file__]) 