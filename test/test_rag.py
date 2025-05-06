import pytest
import json
import os
from unittest.mock import patch, MagicMock, call, ANY
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Set environment variables for testing (avoid relying on actual .env)
# It's crucial that OPENAI_API_KEY is set for the Agent SDK, even if mocked
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["ELASTICSEARCH_URL"] = "http://mock-es:9200"
os.environ["ELASTICSEARCH_API_KEY"] = "mock-api-key"

# Import components under test AFTER setting env vars if they load config at import time
# We might need to adjust imports based on actual project structure/init files
# Assuming src is in the python path or tests are run from root
try:
    from src.ingest import generate_actions # Assuming ingest.py is structured to allow this import
    from src.rag_tool import (
        vector_product_search,
        filtered_product_search,
        Product,
        ProductSearchResults,
        product_agent # Needed for API test mocking target
    )
    from src.api import app # Import FastAPI app
    from agents import Runner # Import Runner to mock it
except ImportError as e:
    pytest.fail(f"Failed to import necessary modules. Ensure src is in PYTHONPATH or adjust imports: {e}", pytrace=False)


# --- Fixtures ---

@pytest.fixture(scope="module")
def test_client():
    """FastAPI test client fixture."""
    client = TestClient(app)
    return client

@pytest.fixture
def mock_embedding_model():
    """Fixture to mock the FastEmbed TextEmbedding model."""
    mock_model = MagicMock()
    # Simulate the embed method returning a generator yielding a list-like object
    mock_model.embed.return_value = iter([[0.1] * 768]) # Return a dummy 768-dim vector
    return mock_model

@pytest.fixture
def mock_es_client():
    """Fixture to mock the Elasticsearch client."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True # Simulate successful connection
    return mock_client

@pytest.fixture
def sample_jl_data_path(tmp_path):
    """Create a temporary sample JSON Lines data file."""
    data = [
        {"title": "Test Diapers", "text": "Soft and comfy.", "metadata": {"brand": "TestBrand", "category": "Diapers", "price": 20.5}},
        {"title": "Test Toy", "text": "Fun building blocks.", "metadata": {"brand": "ToyCo", "category": "Toy", "price": 15.0}},
        "this is not valid json", # Malformed line
        {"title": "Another Toy", "text": "Another fun toy.", "metadata": {"brand": "ToyCo", "category": "Toy", "price": 18.0}},
    ]
    file_path = tmp_path / "sample_data.jl"
    with open(file_path, "w") as f:
        for item in data:
            if isinstance(item, dict):
                f.write(json.dumps(item) + "\n")
            else:
                f.write(item + "\n") # Write invalid line as is
    return file_path

# --- Loader Tests (Testing generate_actions from ingest.py) ---

# Patch the embedding model used *within the ingest module* if necessary
# This depends on how embedding_model is initialized/scoped in ingest.py
@patch('src.ingest.TextEmbedding', new_callable=MagicMock)
def test_generate_actions_success(mock_fastembed_class, sample_jl_data_path, mock_embedding_model):
    """Test successful generation of Elasticsearch bulk actions."""
    # Configure the class mock instance
    mock_fastembed_class.return_value = mock_embedding_model

    actions = list(generate_actions(str(sample_jl_data_path), mock_embedding_model))

    # Expect actions for the 3 valid JSON lines
    assert len(actions) == 3
    mock_embedding_model.embed.assert_called()
    # Check structure of the first generated action's source doc
    first_action = actions[0]
    assert first_action["_index"] == "target_products_v1" # Assuming default index name
    assert "_source" in first_action
    source = first_action["_source"]
    assert source["title"] == "Test Diapers"
    assert "embedding" in source
    assert len(source["embedding"]) == 768 # Check embedding dimension
    assert source["metadata"]["brand"] == "TestBrand"
    assert source["metadata"]["price"] == 20.5

    # Check that embed was called 3 times (once per valid doc)
    assert mock_embedding_model.embed.call_count == 3
    # Check the text passed to embed for the first document
    expected_text = "Title: Test Diapers Brand: TestBrand Category: Diapers Description: Soft and comfy."
    # embed expects a list of strings
    mock_embedding_model.embed.assert_any_call([expected_text])


@patch('src.ingest.TextEmbedding', new_callable=MagicMock)
def test_generate_actions_skips_malformed(mock_fastembed_class, sample_jl_data_path, mock_embedding_model, caplog):
    """Test that malformed JSON lines are skipped and logged."""
    mock_fastembed_class.return_value = mock_embedding_model

    actions = list(generate_actions(str(sample_jl_data_path), mock_embedding_model))

    assert len(actions) == 3 # Only 3 valid lines
    assert "Skipping malformed JSON on line 3" in caplog.text # Check log message

def test_generate_actions_file_not_found(mock_embedding_model):
    """Test handling of non-existent data file."""
    with pytest.raises(FileNotFoundError):
        list(generate_actions("non_existent_file.jl", mock_embedding_model))

# --- Retriever Tests (Testing search functions from rag_tool.py) ---

# Patch dependencies *within the rag_tool module* where the functions are defined
@patch('src.rag_tool.client', new_callable=MagicMock)
@patch('src.rag_tool.embedding_model', new_callable=MagicMock)
def test_vector_product_search_tool(mock_model, mock_es, caplog):
    """Test the vector_product_search tool function."""
    # Mock embedding generation
    mock_model.embed.return_value = iter([[0.2] * 768])
    # Mock Elasticsearch response
    mock_es_response = {
        "hits": {
            "hits": [
                {"_score": 0.9, "_source": {"title": "Found Toy 1", "text": "Desc 1", "metadata": {"brand": "A", "category": "B", "price": 10.0}}}
            ]
        }
    }
    mock_es.search.return_value = mock_es_response

    query = "find toys"
    result = vector_product_search(query_text=query)

    # Assertions
    mock_model.embed.assert_called_once_with([query])
    mock_es.search.assert_called_once()
    args, kwargs = mock_es.search.call_args
    assert kwargs["index"] == "target_products_v1"
    assert "knn" in kwargs["body"]
    assert "filter" not in kwargs["body"]["knn"] # No filter for pure vector search
    assert kwargs["body"]["knn"]["field"] == "embedding"
    assert len(kwargs["body"]["knn"]["query_vector"]) == 768

    assert isinstance(result, ProductSearchResults)
    assert result.search_type == "semantic"
    assert len(result.results) == 1
    assert isinstance(result.results[0], Product)
    assert result.results[0].title == "Found Toy 1"
    assert result.results[0].price == 10.0
    assert "Semantic search for: 'find toys'" in caplog.text

@patch('src.rag_tool.client', new_callable=MagicMock)
@patch('src.rag_tool.embedding_model', new_callable=MagicMock)
def test_filtered_product_search_tool(mock_model, mock_es, caplog):
    """Test the filtered_product_search tool function with filters."""
    mock_model.embed.return_value = iter([[0.3] * 768])
    mock_es_response = {
        "hits": {
            "hits": [
                {"_score": 0.8, "_source": {"title": "Filtered Diaper", "text": "Desc 2", "metadata": {"brand": "Pampers", "category": "Diapers", "price": 25.0}}}
            ]
        }
    }
    mock_es.search.return_value = mock_es_response

    query = "pampers diapers"
    brand = "Pampers"
    category = "Diapers"
    result = filtered_product_search(query_text=query, brand=brand, category=category)

    # Assertions
    mock_model.embed.assert_called_once_with([query])
    mock_es.search.assert_called_once()
    args, kwargs = mock_es.search.call_args
    assert "knn" in kwargs["body"]
    assert "filter" in kwargs["body"]["knn"] # Filter should be present
    expected_filter = {
        "bool": {
            "filter": [
                {"term": {"metadata.brand": brand}},
                {"term": {"metadata.category": category}}
            ]
        }
    }
    assert kwargs["body"]["knn"]["filter"] == expected_filter

    assert isinstance(result, ProductSearchResults)
    assert result.search_type == "filtered"
    assert len(result.results) == 1
    assert result.results[0].title == "Filtered Diaper"
    assert result.results[0].brand == brand
    assert result.results[0].category == category
    assert f"Filtered search: '{query}' [Brand: {brand}, Category: {category}]" in caplog.text

@patch('src.rag_tool.client', new_callable=MagicMock)
@patch('src.rag_tool.embedding_model', new_callable=MagicMock)
def test_search_tool_error_handling(mock_model, mock_es):
    """Test error handling in search tools."""
    mock_model.embed.side_effect = Exception("Embedding failed!")
    mock_es.search.side_effect = Exception("ES Search failed!")

    # Test vector search error
    result_vector = vector_product_search(query_text="test")
    assert result_vector.search_type == "error"
    assert len(result_vector.results) == 0

    # Reset mock side effect if needed before next call
    mock_model.embed.side_effect = None
    mock_model.embed.return_value = iter([[0.3] * 768]) # Reset for next call

    # Test filtered search error
    result_filtered = filtered_product_search(query_text="test", brand="A")
    assert result_filtered.search_type == "error"
    assert len(result_filtered.results) == 0

# --- Full RAG Flow / API Tests ---

@patch('src.api.Runner.run') # Mock the Runner.run method used by the API
async def test_chat_completions_endpoint_success(mock_runner_run, test_client):
    """Test the /target_rag/chat/completions endpoint success case."""
    # Mock the agent's response
    mock_agent_response = MagicMock()
    mock_agent_response.final_output = "Agent response: Found Pampers diapers."
    mock_runner_run.return_value = mock_agent_response

    request_payload = {"question": "Find Pampers", "top_k": 3}
    response = test_client.post("/target_rag/chat/completions", json=request_payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Agent response: Found Pampers diapers."
    assert "meta" in data
    assert "latency_ms" in data["meta"]
    assert isinstance(data["meta"]["latency_ms"], float)
    assert data["meta"]["retrieved_docs"] == 3 # Currently tied to top_k

    # Check if Runner.run was called correctly
    mock_runner_run.assert_called_once_with(
        product_agent, # Check if the correct agent instance was passed
        request_payload["question"],
        config={'top_k': request_payload["top_k"]}
    )

@patch('src.api.Runner.run')
async def test_chat_completions_endpoint_agent_error(mock_runner_run, test_client):
    """Test the endpoint when the agent runner raises an error."""
    mock_runner_run.side_effect = Exception("Agent failed spectacularly!")

    request_payload = {"question": "risky query", "top_k": 5}
    response = test_client.post("/target_rag/chat/completions", json=request_payload)

    # Assertions
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Agent failed spectacularly!" in data["detail"]

def test_chat_completions_endpoint_invalid_request(test_client):
    """Test the endpoint with missing 'question' field."""
    request_payload = {"top_k": 3} # Missing 'question'
    response = test_client.post("/target_rag/chat/completions", json=request_payload)

    # Assertions
    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation
