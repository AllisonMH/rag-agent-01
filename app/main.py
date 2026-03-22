from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from agent.agent import RAGAgent
from models.response import RAGResponse
from observability.tracing import trace_request

app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation Agent for Question Answering",
    version="1.0.0"
)

# Initialize RAG Agent
agent = RAGAgent(
    index_path="indexes/faiss_index",
    model="gpt-4o",
    temperature=0.7,
    top_k=5
)


class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(
        ..., 
        description="The question to ask the RAG agent",
        min_length=1
    )
    top_k: int = Field(
        default=5,
        description="Number of relevant chunks to retrieve",
        ge=1,
        le=20
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "top_k": 5
            }
        }


@app.post("/query", response_model=RAGResponse)
@trace_request("query_endpoint")
async def query_agent(request: QueryRequest) -> RAGResponse:
    """
    Query the RAG agent with a question.
    
    Args:
        request: QueryRequest containing the question and parameters
    
    Returns:
        RAGResponse with answer, sources, and confidence score
    """
    try:
        # Call agent
        result = agent.query(
            question=request.question,
            top_k=request.top_k,
            stream=False,
            return_context=True
        )
        
        # Calculate confidence based on retrieval scores
        # Simple heuristic: if we have context, confidence is high
        confidence = 0.85 if result.get("context") else 0.3
        
        # Return structured response
        return RAGResponse(
            answer=result["response"],
            sources=result.get("context", []),
            confidence=confidence
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Index not found. Please run ingestion first: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Query the RAG agent with a question",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_model": agent.model,
        "index_loaded": agent.retriever.index is not None,
        "chunks_count": len(agent.retriever.chunks)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)