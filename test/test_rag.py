import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv()

from fastembed import TextEmbedding
from elasticsearch import Elasticsearch, helpers

# --- Configuration ---
ES_URL = os.getenv("ELASTICSEARCH_URL")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
INDEX_NAME = os.getenv("INDEX_NAME", "target_products_v1")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/filtered_products.jl")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Validate essential configurations
if not ES_URL or not ES_API_KEY:
    logging.error("ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY must be set in the .env file.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Initialize Embedding Model ---
try:
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logging.info("Embedding model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing embedding model: {e}")
    sys.exit(1) # Exit if model fails to load


# --- Elasticsearch Connection ---
try:
    logging.info(f"Connecting to Elasticsearch at {ES_URL}")
    client = Elasticsearch(
        ES_URL,
        api_key=ES_API_KEY,
        request_timeout=60 # Increase timeout for potentially long operations
    )
    # Verify connection
    if not client.ping():
        raise ValueError("Connection failed")
    logging.info("Connected to Elasticsearch successfully.")
except Exception as e:
    logging.error(f"Error connecting to Elasticsearch: {e}")
    sys.exit(1)

# Simple pytest unit tests for embedding model and Elasticsearch connection
def test_embedding_model_initialization():
    assert embedding_model is not None

def test_embedding_embed_returns_correct_dimensions():
    emb_list = list(embedding_model.embed(['test input']))
    # Should return exactly one embedding vector
    assert len(emb_list) == 1
    emb0 = emb_list[0].tolist()
    # Embedding vector should have the configured dimension
    assert hasattr(emb0, '__len__')
    assert len(emb0) == EMBEDDING_DIM
    # All elements should be floats
    assert all(isinstance(x, float) for x in emb0)

def test_elasticsearch_ping():
    # Elasticsearch client should successfully ping the server
    assert client.ping() is True


from src.rag_service import pytest_rag_service

def test_answer_user_query_api():
    results = pytest_rag_service("test query")
    logging.info(f"Results: {results}")
    assert results is not None

