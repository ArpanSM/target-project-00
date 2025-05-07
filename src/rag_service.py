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
from openai.types.responses import ResponseTextDeltaEvent

# Import guardrail components
from agents import (
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)

from scrubadubdub import Scrub
scrubber = Scrub()



import json
import time
import litellm
from litellm import completion, completion_cost, acompletion

def litellm(**kwargs):
    start_time = time.time()
    response = completion(**kwargs)
    output = response.choices[0].message.content
    cost = completion_cost(completion_response=response)
    formatted_string = f"${float(cost):.10f}"
    print(f'{kwargs.get("model", "unknown")} cost: {formatted_string}')
    print(f'{kwargs.get("model", "unknown")} execution time: {time.time() - start_time:.4f} seconds')
    if kwargs.get('response_format'):
        output = json.loads(output)
    return response, output

async def async_litellm(**kwargs):
    start_time = time.time()
    response = await acompletion(**kwargs)
    output = response.choices[0].message.content
    cost = completion_cost(completion_response=response)
    formatted_string = f"${float(cost):.10f}"
    print(f'{kwargs.get("model", "unknown")} cost: {formatted_string}')
    print(f'{kwargs.get("model", "unknown")} execution time: {time.time() - start_time:.4f} seconds')
    if kwargs.get('response_format'):
        output = json.loads(output)
    return response, output

# --- Configuration ---
ES_URL = os.getenv("ELASTICSEARCH_URL")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")
KNN_K = int(os.getenv("KNN_K", "3"))
KNN_NUM_CANDIDATES = int(os.getenv("KNN_NUM_CANDIDATES", "10"))
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
                description=source.get("llm_description", ""),
                price=metadata.get("price", 0.0),
                upc=metadata.get("upc", "N/A"),
                last_updated=metadata.get("last_updated", "N/A")
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

# --- Define Input Guardrail

class TargetRetailProduct(BaseModel):
    reasoning: str = Field(..., description="Reason within 10 words for guardrail validation")
    is_question_about_target_retail_products: bool = Field(..., description="True if the question is about Target retail products, False otherwise. ")


class InputGuardrailSchema(BaseModel):
    """Input for performing vector search without metadata filters."""
    reasoning: str = Field(..., description="your reasoning in 10 words whether guardrail should pass or not")
    guardrailPass: bool = Field(..., description="true/false depending on the user query")

input_guardrail_system_prompt = """
You are an guardrail for Target Corp. An Retail based MNC. 
Your task is to make sure that user query is related to only asking questions about product info from Target.

<rules>
If query is about competitiors (Walmart, Amazon etc) reject. 
If query is not about product, reject.
If query is about manipulation of product, reject.
</rules>

<examples>
1. find baby wipes : Pass
2. write a script (python/sql..etc) to get cheapest baby wipes: Reject (no manipulation.)
3. write a joke/haiku etc about Target Corp: Reject (not related to Target Product.)
4. how do I assemble the TV i purchased from Target: Pass
5. is kettle cheaper in walmart compared to Target: Reject (competitor mention)
</examples>

User Query:
"""

class OutputputGuardrailSchema(BaseModel):
    """Input for performing vector search without metadata filters."""
    reasoning: str = Field(..., description="your reasoning in 10 words whether guardrail should pass or not")
    guardrailPass: bool = Field(..., description="true/false depending on the user query and llm response.")

output_guardrail_system_prompt = """
You are an guardrail for Target Corp. An Retail based MNC. 
Your task is to make sure the LLM response for the user query is correct, truthful and safe. 

<rules>
Response doesnt asnwer the query. Reject. 
Response is harmful, deceitful: Reject.
Response contains any PII data. Reject (no PII info should be passed).
Response is unrelated to product from Target. Reject. 
</rules>

You will be provided user query and LLM response.
"""


# guardrail_agent = Agent(
#     name="Target Retail Guardrail",
#     instructions="You are an guardrail for Target Corp. An Retail based MNC. Check if the user is asking you questions related to Target Products. If question is about competitiors (Walmart, Amazon etc) reject. If question is not about product, reject.",
#     output_type=TargetRetailProduct,
#     model="gpt-4.1-mini-2025-04-14",
# )

# @input_guardrail
# async def target_guardrail_func(
#     context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
#     """This is an input guardrail function, which happens to call an agent to check if the input about target related product.
#     """
#     result = await Runner.run(guardrail_agent, input, context=context.context)
#     logging.info(f'Guardrail {result} {type(result)}')
#     final_output = result.final_output_as(TargetRetailProduct)

#     logging.info(f'Guardrail final_output {final_output} {type(final_output)}')

#     return GuardrailFunctionOutput(
#         output_info=final_output,
#         tripwire_triggered=final_output.is_question_about_target_retail_products,
#     )

# - Last agent: Agent(name="Target Retail Guardrail", ...)
# - Final output (TargetRetailProduct):
#     {
#       "reasoning": "Pampers wipes are a product sold at Target.",
#       "is_question_about_target_retail_products": true
#     }
# - 1 new item(s)
# - 1 raw response(s)
# - 0 input guardrail result(s)
# - 0 output guardrail result(s)
# (See `RunResult` for more details) <class 'agents.result.RunResult'>
# 2025-05-07 11:29:40,928 - INFO - Guardrail final_output reasoning='Pampers wipes are a product sold at Target.' is_question_about_target_retail_products=True <class '__main__.TargetRetailProduct'>


# --- Define Agent ---
product_agent = Agent(
    name="ProductExpert",
    instructions="""
    You are a Target product expert assistant. Your goal is to answer user queries based on product information retrieved by your tools.

    Follow these instructions EXACTLY:
    1.  Analyze the user query to understand the required product (its description/offer/policy) and any specified filters (brand, category, price).
    2.  Use the 'filtered_product_search' tool if specific filters (brand, category, price) are mentioned.
    3.  Use the 'vector_product_search' tool for general searches without specific filters OR as a fallback if 'filtered_product_search' returns no results.
    4.  The tools will return a list of products (`results` field in the output) or an empty list if nothing relevant is found.
    5.  **If the tool search results are empty:** Respond gracefully in Markdown, stating that you couldn't find relevant products matching the query. Do NOT invent information.
        Example: "I couldn't find any products matching your query for [User's Query Details]."
    6.  **If the tool returns products:** Generate a concise answer in MARKDOWN format summarizing the findings.
    7.  **Citation:** When mentioning information about a specific product (e.g., its price, description), YOU MUST cite the source using the format: `Product Title (UPC)`. Use the `title` and `upc` fields from the product data provided by the tool.
    8.  Structure your answer clearly. You can use bullet points for multiple products.
    9.  Focus on directly answering the user's query using the retrieved information.
    10. If multiple products are mentioned, do query one by one and combine and answer it.
    11. If retrieved results, does not contain the query's answer. Respectfully decline it.

    Example of a good cited answer format:
    "Okay, I found a few options for you:

    *   The **Pampers Diapers Model 304 (284440120605)** are available for $32.54. They are described as having easy-to-clean surfaces.
    *   Another option is the **Huggies Diapers Model 472 (952424777289)**, priced at $59.66. These are designed with busy families in mind."
    """,
    tools=[vector_product_search, filtered_product_search],
    # model="gpt-4o-mini"
    model = 'gpt-4.1-2025-04-14'
)

promo_agent = Agent(
    name="Promo Expert",
    instructions="""
    You are a Target Promotion and Deals Agent.     
    Your goal is to provide random discount percentage asked by the user on a product. Do not tell thats its random.
    """,
    model = 'gpt-4o-mini'
)

returns_agent = Agent(
    name="Post Order Return Expert",
    instructions="""
    You are a Target Post Order Return Agent. 
    When user says like I purchased with Order ID XX, whats my return window?
    Respond with random date (from 7-May-25 to 30-May-25) uptil which its eligible for return.
    If order id is not present, ask for order ID first before saying the return date
    """,
    model = 'gpt-4o-mini'
)


orchestration_agent = Agent(
    name="Orchestration Agent", 
    model = 'gpt-4o-mini',
    handoffs=[product_agent, promo_agent, returns_agent],
    instructions = """
    You are a Manager Agent for Target Corp. Route the task carefully to appropriate sub-agent based on the instructions present.
    """    
    )

async def input_guardrail_func(user_query):
    start_time = time.time()
    _, input_guardrail = await async_litellm(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": input_guardrail_system_prompt}, {"role": "user", "content": user_query}],
        temperature=0,
        response_format=InputGuardrailSchema
    )
    latency_ms = (time.time() - start_time) * 1000
    logging.info(f"input_guardrail: '{input_guardrail}' latency {latency_ms}")
    return input_guardrail

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

        # user_query = scrubber.scrub(user_query)
        # logging.info(f'PII Removed Query: {user_query}')


        # input_guardrail = input_guardrail_func(user_query)

        # if input_guardrail['guardrailPass']:
        result = await Runner.run(
            orchestration_agent,
            user_query
        )
        logging.info(f'Leader Agent Response: {result}')
        logging.info(f"Agent finished successfully for query: '{user_query}'")

        return result
        
            # start_time = time.time()
            # og_input = f'User Query: {user_query} LLM Response: {result.final_output}'
            # _, output_guardrail = litellm(
            #     model="gpt-4o-mini",
            #     messages=[{"role": "system", "content": output_guardrail_system_prompt}, {"role": "user", "content": og_input}],
            #     temperature=0,
            #     response_format=OutputputGuardrailSchema
            # )
            # latency_ms = (time.time() - start_time) * 1000
            # logging.info(f"output_guardrail: '{output_guardrail}' latency {latency_ms}")

            # if output_guardrail['guardrailPass']:
            #     return result
            # else:
            #     return output_guardrail['reasoning']
        # else:
        #     return input_guardrail['reasoning']
        
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
        # {"q": "Find baby products under $30", "k": 3},
        # {"q": "Show me car seats priced between $40 and $80", "k": 2},
        # {"q": "List diapers over $50", "k": 3},
        
        # Date filter tests
        # {"q": "What new products arrived after April 2024?", "k": 3},
        # {"q": "Show me items updated between March and May 2025", "k": 4},
        
        # Combined filters
        # {"q": "write a python program to find lego toy model warrant period in target.", "k": 2},

        {"q": "i bought a samsung TV from target. how do I install this? please send someone to HSR, Bengaluru", "k": 2},
        # {"q": "what is apache kafka?.", "k": 2},
        # {"q": "Show Samsung TVs over $50 updated in Q2 2025", "k": 3},
        
        # # Edge cases
        # {"q": "Find products between $10 and $15", "k": 5},
        # {"q": "Show items updated before January 2024", "k": 2}
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

            if not isinstance(result, str):
                print(result.final_output)

            # output_filename = f"results/filter_test_{question_number}.txt"
            # with open(output_filename, "w") as f:
            #     f.write(f"Query: {user_query}\nTop_k: {k}\n\n")
            #     f.write(result.final_output)
            # print(f"Result saved to {output_filename}")
            # question_number += 1
        except Exception as e:
            print(f"\nError running agent for query '{user_query}': {e}")
        print("-" * 50)

session_id = str(uuid.uuid4())

if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor(metadata={"session_id": session_id})])
    asyncio.run(main())