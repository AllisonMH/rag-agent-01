from pydantic import BaseModel, Field
from typing import List


class RAGResponse(BaseModel):
    """Model for RAG agent responses."""
    
    answer: str = Field(
        ..., 
        description="The generated answer to the user's question"
    )
    
    sources: List[str] = Field(
        default_factory=list,
        description="List of source chunks used to generate the answer"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer (0.0 to 1.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    "Machine learning is a method of data analysis...",
                    "ML algorithms build models based on sample data..."
                ],
                "confidence": 0.92
            }
        }