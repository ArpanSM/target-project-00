from agents import set_trace_processors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from pydantic import BaseModel

import time
import uuid
import logging

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

from langsmith.wrappers import OpenAIAgentsTracingProcessor

# Import existing RAG components
from rag_service import answer_user_query_api

app = FastAPI(title="Target RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class ChatResponseMeta(BaseModel):
    latency_ms: float
    retrieved_docs: int

class ChatResponse(BaseModel):
    answer: str
    meta: ChatResponseMeta

# --- Endpoint Implementation ---
@app.post("/target_rag/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Process natural language queries using RAG system"""
    start_time = time.time()
    
    try:
        # --- Execute the agent via the dedicated function from rag_tool ---
        agent_result = await answer_user_query_api(
            user_query=request.question,
            top_k=request.top_k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        retrieved_count_for_meta = request.top_k

        return ChatResponse(
            answer=agent_result.final_output,
            meta=ChatResponseMeta(
                latency_ms=round(latency_ms, 2),
                retrieved_docs=retrieved_count_for_meta
            )
        )
        
    except Exception as e:
        # Log the error with traceback for better debugging
        logging.error(f"API Error processing query '{request.question}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while processing your request."
        )

if __name__ == "__main__":    
    import uvicorn
    set_trace_processors([OpenAIAgentsTracingProcessor(metadata={"session_id": str(uuid.uuid4())})])
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


# uvicorn api:app --reload

# curl -X POST "http://localhost:8000/target_rag/chat/completions" -H "Content-Type: application/json" -d "{\"question\": \"fruit juice\", \"top_k\": 5}"