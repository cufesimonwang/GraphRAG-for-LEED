import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch

from data_preprocessing.content_extractor import ContentExtractor # Updated import
from kg_construction.kg_extractor import KnowledgeGraphExtractor # Updated import
# from src.kg.graph_constructor import GraphConstructor # Commented out

@pytest.fixture
def mock_llm_response(): # This fixture might need to be adapted or replaced for the new KGE
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
    test_file.write_text("Test content for content extractor.") # More descriptive content
    
    # Initialize content extractor with mock config
    with patch('data_preprocessing.content_extractor.ContentExtractor._load_config') as mock_load_config: # Updated patch path
        mock_load_config.return_value = {
            'paths': {
                'data_dir': str(tmp_path), # Not directly used by process_file if file_path is absolute
                'processed_dir': str(tmp_path / 'processed'), 
                'jsonl_output_dir': str(tmp_path / 'processed' / 'jsonl_output'), 
                # Add other paths if _init_directories is called and uses them
            },
            'llm': {'provider': 'openai', 'model_name': 'gpt-4-turbo-preview', 'api_key': 'test_key'}, # Ensure all expected LLM keys are present
            'chunking': {'chunk_size': 100, 'chunk_overlap': 10},
            'vision': {'enabled': False} # Assuming vision is not tested here
        }
        # ContentExtractor now takes the config dict directly
        extractor = ContentExtractor(config_path="dummy_config_not_directly_used_due_to_mock.yaml") # Path can be dummy
        
        # Process the file
        result = extractor.process_file(test_file, mode="hybrid") # Assuming mode is still relevant
        
        # Verify the result based on new ContentExtractor output
        assert result['file_path'] == str(test_file)
        assert result['mode'] == "hybrid"
        assert 'metadata' in result
        assert "jsonl_output" in result # New key for jsonl output path
        if result["jsonl_output"]: # It might be None if no text blocks were processed
            assert result["jsonl_output"].endswith(".jsonl")
        # Add more specific assertions if graphrag_output is expected and generated
        # assert "graphrag_output" in result 

def test_kg_extractor_process_chunk(sample_text): # mock_llm_response fixture removed, will define mock inside
    """Test knowledge graph extractor's process_chunk method."""
    
    sample_chunk_data = {
        "doc_id": "test_doc_1",
        "section": "Test Section A",
        "text": sample_text, 
        "page": 1
    }
    chunk_idx = 0

    # Define the mock LLM output for the _extract_triples call
    mock_llm_output_for_extract_triples = [
        {
            "head_entity": {"name": "LEED Credit", "type": "Concept"},
            "relation": "requires",
            "tail_entity": {"name": "Energy Performance", "type": "Concept"}
        },
        {
            "head_entity": {"name": "Energy Performance", "type": "Concept"},
            "relation": "contributes_to",
            "tail_entity": {"name": "Sustainable Design", "type": "Concept"}
        }
    ]

    # Initialize KG extractor with mock config
    # Patch _load_config and _get_llm_response for KnowledgeGraphExtractor
    with patch('kg_construction.kg_extractor.KnowledgeGraphExtractor._load_config') as mock_load_config, \
         patch('kg_construction.kg_extractor.KnowledgeGraphExtractor._get_llm_response') as mock_get_llm_response:
        
        mock_load_config.return_value = { # Config KGE expects
            'llm': {'provider': 'openai', 'model_name': 'gpt-4-turbo-preview', 'api_key': 'test_key'}, # Ensure all expected LLM keys
            'extraction': {'relation_types': ['REQUIRES', 'CONTRIBUTES_TO', 'IS_PART_OF']}, # Uppercase as per KGE
            'paths': {'prompts_file': 'dummy_prompts.yaml'} # Path to prompts, even if prompts are also mocked or simple
        }
        # Mock for _extract_triples call
        mock_get_llm_response.return_value = json.dumps(mock_llm_output_for_extract_triples)
        
        # KGE now takes config_path, but _load_config is mocked, so path can be dummy
        extractor = KnowledgeGraphExtractor(config_path="dummy_config_not_directly_used_due_to_mock.yaml")
        
        # Process the chunk
        triples = extractor.process_chunk(sample_chunk_data, chunk_idx)
        
        # Verify the results
        assert isinstance(triples, list)
        if triples: 
            assert len(triples) == 2 # Based on mock_llm_output_for_extract_triples
            first_triple = triples[0]
            assert "head_name" in first_triple
            assert first_triple["head_name"] == "LEED Credit"
            assert "relation_type" in first_triple
            assert first_triple["relation_type"] == "REQUIRES" # KGE normalizes relation
            assert "tail_name" in first_triple
            assert first_triple["tail_name"] == "Energy Performance"
            assert "doc_id" in first_triple
            assert first_triple["doc_id"] == "test_doc_1"
            assert "chunk_id" in first_triple # Check if chunk_id is generated correctly
            assert "test_doc_1" in first_triple["chunk_id"] and "chunk0" in first_triple["chunk_id"]

# TODO: Refactor or remove this test as GraphConstructor is outdated.
# def test_graph_constructor_get_graph_stats():
#     """Test graph constructor's get_graph_stats method."""
#     # Create a sample graph
#     graph = {
#         'nodes': [
#             {'id': '1', 'type': 'Credit', 'label': 'LEED Credit'},
#             {'id': '2', 'type': 'Requirement', 'label': 'Energy Performance'}
#         ],
#         'edges': [
#             {'source': '1', 'target': '2', 'type': 'requires'}
#         ]
#     }
    
#     # Initialize graph constructor with mock config
#     with patch('src.kg.graph_constructor.GraphConstructor._load_config') as mock_load_config:
#         mock_load_config.return_value = {}
#         constructor = GraphConstructor("config.yaml")
        
#         # Get graph statistics
#         stats = constructor.get_graph_stats(graph)
        
#         # Verify the statistics
#         assert stats['nodes'] == 2
#         assert stats['edges'] == 1
#         assert len(stats['node_types']) == 2
#         assert len(stats['edge_types']) == 1
#         assert 'Credit' in stats['node_types']
#         assert 'requires' in stats['edge_types']

if __name__ == '__main__':
    pytest.main([__file__])