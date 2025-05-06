import asyncio
import logging
import sys
import textwrap
import os
import uuid
from elasticsearch import Elasticsearch
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
load_dotenv()

# Import Agent SDK components
from agents import Agent, Runner, Tool, function_tool, set_trace_processors

# Import guardrail components
from agents import (
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)

# --- Configuration ---
ES_URL = os.getenv("ELASTICSEARCH_URL")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")
KNN_K = int(os.getenv("KNN_K", "3"))
KNN_NUM_CANDIDATES = int(os.getenv("KNN_NUM_CANDIDATES", "20"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

from langsmith.wrappers import OpenAIAgentsTracingProcessor

# Validate essential configurations
if not ES_URL or not ES_API_KEY:
    logging.error("ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY must be set in the .env file.")
    sys.exit(1)
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY environment variable not set in .env file. Agent execution will likely fail.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Initialize Embedding Model ---
try:
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logging.info("Embedding model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing embedding model: {e}")
    sys.exit(1)

# --- Elasticsearch Connection ---
try:
    logging.info(f"Connecting to Elasticsearch at {ES_URL}")
    client = Elasticsearch(
        ES_URL,
        api_key=ES_API_KEY,
        request_timeout=30
    )
    if not client.ping():
        raise ValueError("Connection failed")
    logging.info("Connected to Elasticsearch successfully.")
except Exception as e:
    logging.error(f"Error connecting to Elasticsearch: {e}")
    sys.exit(1)

# --- Tool Input Schemas ---
class VectorSearchInput(BaseModel):
    """Input for performing vector search without metadata filters."""
    query_text: str = Field(..., description="The user's query text to search for similar products.")

metadata_fields = ["brand", "category", "upc", "price", "last_updated"]
metadata_fields_description = {
    "brand": "Brand manufacturer Ex. Pampers, Huggies, Vtech, etc.",
    "category": "Product category Ex. Diapers, Baby Wipes, Car Seats, etc.",
    "upc": "Universal Product Code Ex. 123456789012",
    "price": "Current price in USD Ex. 60, 42.99, 12.99, etc.",
    "last_updated": "Date of last price update Ex. 2025-05-05 (YYYY-MM-DD)"
}

class VectorMetadataSearchInput(BaseModel):
    """Input for performing vector search WITH metadata filters."""
    query_text: str = Field(..., description="The user's query text.")
    metadata_filters: Dict[str, Any] = Field(..., description="Dictionary of metadata fields (e.g., {'brand': 'Pampers', 'category': 'Diapers'}) to filter the search. " + str(metadata_fields_description))

# --- Pydantic Models for Structured Results ---
class Product(BaseModel):
    title: str = Field(..., description="Name of the product")
    brand: str = Field(..., description="Brand manufacturer")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Short product description from llm_description")
    price: float = Field(..., description="Current price in USD")
    upc: str = Field(..., description="Universal Product Code (UPC)")
    last_updated: str = Field(..., description="Date of last price update")
class ProductSearchResults(BaseModel):
    results: List[Product] = Field(..., description="List of matching products")
    search_type: str = Field(..., description="Type of search performed")

@function_tool
def vector_product_search(query_text: str) -> ProductSearchResults:
    """
    Search products using semantic similarity to the query text.
    Use for general searches without specific filters.
    
    Args:
        query_text: Natural language query describing desired products
    """
    logging.info(f"Semantic search for: '{query_text}'")
    try:
        query_embedding = list(embedding_model.embed([query_text]))[0].tolist()

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": KNN_K,
            "num_candidates": KNN_NUM_CANDIDATES
        }
        query_body = {"knn": knn_query}

        response = client.search(index=INDEX_NAME, body=query_body)
        # Convert results to Pydantic model
        products = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            products.append(Product(
                title=source.get("title", "N/A"),
                brand=metadata.get("brand", "N/A"),
                category=metadata.get("category", "N/A"),
                description=textwrap.shorten(source.get("text", ""), width=150),
                price=metadata.get("price", 0.0),
                upc=metadata.get("upc", "N/A")
            ))
        # print(products)
        return ProductSearchResults(
            results=products,
            search_type="semantic"
        )
    
    except Exception as e:
        logging.error(f"Search error: {e}")
        return ProductSearchResults(results=[], search_type="error")

@function_tool
def filtered_product_search(
    query_text: str, 
    brand: Optional[str] = None,
    category: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    updated_after: Optional[str] = None,
    updated_before: Optional[str] = None
) -> ProductSearchResults:
    """
    Search products using semantic similarity with advanced filters.
    
    Args:
        query_text: Natural language query describing desired products
        brand: Specific brand to filter by (optional)
        category: Product category to filter by (optional)
        price_min: Minimum price filter (optional)
        price_max: Maximum price filter (optional)
        updated_after: Return products updated after this date (YYYY-MM-DD) (optional)
        updated_before: Return products updated before this date (YYYY-MM-DD) (optional)
    """
    logging.info(f"Advanced filtered search: '{query_text}' [Brand: {brand}, Category: {category}, Price: {price_min}-{price_max}, Updated: {updated_after} to {updated_before}]")
    try:
        # Build range filters
        range_filters = []
        if price_min is not None or price_max is not None:
            price_range = {}
            if price_min: price_range["gte"] = price_min
            if price_max: price_range["lte"] = price_max
            range_filters.append({"range": {"metadata.price": price_range}})
        
        if updated_after or updated_before:
            date_range = {}
            if updated_after: date_range["gte"] = updated_after
            if updated_before: date_range["lte"] = updated_before
            range_filters.append({"range": {"metadata.last_updated": date_range}})

        # Build term filters
        term_filters = []
        if brand: term_filters.append({"term": {"metadata.brand": brand}})
        if category: term_filters.append({"term": {"metadata.category": category}})

        # Combine all filters
        filter_clauses = term_filters + range_filters
        
        query_embedding = list(embedding_model.embed([query_text]))[0].tolist()

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": KNN_K,
            "num_candidates": KNN_NUM_CANDIDATES
        }

        query_body = {
            "knn": knn_query,
            "query": {
                "bool": {
                    "filter": filter_clauses
                }
            } if filter_clauses else None
        }

        # Remove empty query if no filters
        if query_body["query"] is None:
            del query_body["query"]

        response = client.search(index=INDEX_NAME, body=query_body)
        
        # Convert results to Pydantic model
        products = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            products.append(Product(
                title=source.get("title", "N/A"),
                brand=metadata.get("brand", "N/A"),
                category=metadata.get("category", "N/A"),
                description=source.get("llm_description", ""),
                price=metadata.get("price", 0.0),
                upc=metadata.get("upc", "N/A"),
                last_updated=metadata.get("last_updated", "N/A")
            ))

        logging.info(f"Filtered search results: {products}")
        
        return ProductSearchResults(
            results=products,
            search_type="advanced_filtered"
        )
    
    except Exception as e:
        logging.error(f"Advanced filtered search error: {e}")
        return ProductSearchResults(results=[], search_type="error")

# --- Define Agent ---
product_agent = Agent(
    name="ProductExpert",
    instructions="""
    You are a Target product expert assistant. Your goal is to answer user queries based on product information retrieved by your tools.

    Follow these instructions EXACTLY:
    1.  Analyze the user query to understand the required product and any specified filters (brand, category, price).
    2.  Use the 'filtered_product_search' tool if specific filters (brand, category, price) are mentioned.
    3.  Use the 'vector_product_search' tool for general searches without specific filters OR as a fallback if 'filtered_product_search' returns no results.
    4.  The tools will return a list of products (`results` field in the output) or an empty list if nothing relevant is found.
    5.  **If the tool search results are empty:** Respond gracefully in Markdown, stating that you couldn't find relevant products matching the query. Do NOT invent information.
        Example: "I couldn't find any products matching your query for [User's Query Details]."
    6.  **If the tool returns products:** Generate a concise answer in MARKDOWN format summarizing the findings.
    7.  **Citation:** When mentioning information about a specific product (e.g., its price, description), YOU MUST cite the source using the format: `Product Title (UPC)`. Use the `title` and `upc` fields from the product data provided by the tool.
    8.  Structure your answer clearly. You can use bullet points for multiple products.
    9.  Focus on directly answering the user's query using the retrieved information.

    Example of a good cited answer format:
    "Okay, I found a few options for you:

    *   The **Pampers Diapers Model 304 (284440120605)** are available for $32.54. They are described as having easy-to-clean surfaces.
    *   Another option is the **Huggies Diapers Model 472 (952424777289)**, priced at $59.66. These are designed with busy families in mind."
    """,
    tools=[vector_product_search, filtered_product_search],
    model="gpt-4.1-mini-2025-04-14"
)

# --- Function to answer the user query from the api ---
async def answer_user_query_api(user_query: str, top_k: int = 3):
    """
    This function is used to answer the user query from the api.py file.
    It runs the product_agent with the given query and top_k setting.

    Args:
        user_query: The natural language query from the user.
        top_k: The desired number of results to retrieve (affects KNN search).

    Returns:
        The result object from Runner.run, containing the agent's final output and execution details.

    Raises:
        Exception: If the agent execution fails.
    """
    logging.info(f"Answering API query: '{user_query}' with top_k={top_k}")

    global KNN_K
    original_knn_k = KNN_K
    KNN_K = top_k

    try:
        result = await Runner.run(
            product_agent,
            user_query
        )
        logging.info(f"Agent finished successfully for query: '{user_query}'")
        return result
    except Exception as e:
        logging.error(f"Agent execution failed for query '{user_query}': {e}", exc_info=True)
        # Re-raise the exception so the API layer can handle it (e.g., return 500)
        raise
    finally:
        # Reset KNN_K to its original value after the request.
        KNN_K = original_knn_k

def pytest_rag_service(query: str):
    """
    This function is used to test the rag_service.py file.
    It runs the product_agent in a synchronous manner with the given query and top_k setting.

    Args:
        query: The natural language query from the user.
        top_k: The desired number of results to retrieve (affects KNN search).
    """
    result = Runner.run_sync(product_agent, query)
    logging.info(f"Agent finished successfully for query: '{query}'")
    return result

# --- Main Execution Block (for local testing) ---
async def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set. Exiting.")
        sys.exit(1)

    # Updated Test Queries with Price/Date Filters
    queries_to_test = [
        # Price filter tests
        {"q": "Find baby products under $30", "k": 3},
        {"q": "Show me car seats priced between $40 and $80", "k": 2},
        {"q": "List diapers over $50", "k": 3},
        
        # Date filter tests
        {"q": "What new products arrived after April 2024?", "k": 3},
        {"q": "Show me items updated between March and May 2025", "k": 4},
        
        # Combined filters
        {"q": "Find Pampers wipes under $20 updated this month", "k": 2},
        {"q": "Show Samsung TVs over $50 updated in Q2 2025", "k": 3},
        
        # Edge cases
        {"q": "Find products between $10 and $15", "k": 5},
        {"q": "Show items updated before January 2024", "k": 2}
    ]

    print("\n--- Running Enhanced Filter Tests ---")
    question_number = 1
    for item in queries_to_test:
        user_query = item["q"]
        k = item["k"]
        print(f"\n--- Query {question_number}: '{user_query}' (top_k={k}) ---")
        try:
            result = await answer_user_query_api(user_query, top_k=k)
            print("\nAgent Final Output:")
            print(result.final_output)

            output_filename = f"results/filter_test_{question_number}.txt"
            with open(output_filename, "w") as f:
                f.write(f"Query: {user_query}\nTop_k: {k}\n\n")
                f.write(result.final_output)
            print(f"Result saved to {output_filename}")
            question_number += 1
        except Exception as e:
            print(f"\nError running agent for query '{user_query}': {e}")
        print("-" * 50)

session_id = str(uuid.uuid4())

if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor(metadata={"session_id": session_id})])
    asyncio.run(main())