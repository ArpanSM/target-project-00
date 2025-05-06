from fastembed import TextEmbedding
from elasticsearch import Elasticsearch, helpers
import json
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ES_URL = os.getenv("ELASTICSEARCH_URL")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
INDEX_NAME = os.getenv("INDEX_NAME", "target_products_v1")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/filtered_products.jl")

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


# --- Define Index Mapping ---
index_mapping = {
    "properties": {
        "title": {"type": "text"},
        "llm_description": {"type": "text"},
        "metadata": {
            "properties": {
                "brand": {"type": "keyword"},
                "category": {"type": "keyword"},
                "upc": {"type": "keyword"},
                "price": {"type": "float"},
                "last_updated": {"type": "date"}
            }
        },
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIM,
            "index": True,
            "similarity": "dot_product" 
        }
    }
}

# --- Create Index ---
try:
    logging.info(f"Checking if index '{INDEX_NAME}' exists...")
    if client.indices.exists(index=INDEX_NAME):
        logging.warning(f"Index '{INDEX_NAME}' already exists. Deleting.")
        client.indices.delete(index=INDEX_NAME, ignore=[400, 404])
    logging.info(f"Creating index '{INDEX_NAME}' with mapping.")
    client.indices.create(index=INDEX_NAME, mappings=index_mapping)
    logging.info(f"Index '{INDEX_NAME}' created successfully.")
except Exception as e:
    logging.error(f"Error creating index '{INDEX_NAME}': {e}")
    sys.exit(1)


# --- Data Ingestion ---
def generate_actions(filepath, model):
    """Generator function to yield actions for Elasticsearch bulk helper."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                    # Clean and prepare llm_description
                    llm_desc = doc.get('llm_description', '').replace('-', '')
                    
                    # Combine relevant fields for embedding
                    text_to_embed = f"Title: {doc.get('title', '')} Brand: {doc.get('metadata', {}).get('brand', '')} Category: {doc.get('metadata', {}).get('category', '')} Description: {llm_desc}"

                    # Generate embedding
                    embedding = list(model.embed([text_to_embed]))[0].tolist()

                    source_doc = {
                        "title": doc.get("title"),
                        "llm_description": llm_desc,
                        "metadata": doc.get("metadata"),
                        "embedding": embedding
                    }
                    yield {
                        "_index": INDEX_NAME,
                        "_source": source_doc,
                        # "_id": doc.get('metadata', {}).get('upc')
                    }
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON on line {line_num}: {line.strip()}")
                except Exception as e:
                    logging.warning(f"Error processing document on line {line_num}: {e}")

    except FileNotFoundError:
        logging.error(f"Data file not found at {filepath}")
        raise # Re-raise to stop execution if file not found
    except Exception as e:
        logging.error(f"Error reading data file {filepath}: {e}")
        raise # Re-raise other file reading errors


logging.info(f"Starting data ingestion from {DATA_FILE_PATH}")
success_count = 0
fail_count = 0
try:
    # Using streaming_bulk for potentially large files
    for ok, action in helpers.streaming_bulk(
        client=client,
        actions=generate_actions(DATA_FILE_PATH, embedding_model),
        chunk_size=100, # Adjust chunk size based on document size and memory
        request_timeout=120 # Longer timeout for bulk operations
    ):
        if ok:
            success_count += 1
        else:
            fail_count += 1
            logging.warning(f"Failed to index document: {action}")
        if (success_count + fail_count) % 500 == 0: # Log progress periodically
             logging.info(f"Indexed {success_count} documents, {fail_count} failures...")

    # Refresh the index to make documents searchable
    client.indices.refresh(index=INDEX_NAME)
    logging.info(f"Data ingestion complete. Indexed {success_count} documents.")
    if fail_count > 0:
        logging.warning(f"{fail_count} documents failed to index.")

except Exception as e:
    logging.error(f"Error during bulk ingestion: {e}")

logging.info("Ingestion script finished.")