import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
from openai import OpenAI


class RAGRetriever:
    def __init__(self, index_path: str = "indexes/faiss_index"):
        """Initialize the RAG retriever."""
        self.index_path = index_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.index = None
        self.chunks = []
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and chunks from disk."""
        # Load FAISS index
        index_file = f"{self.index_path}.index"
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(index_file)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load chunks metadata
        chunks_file = f"{self.index_path}.pkl"
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using OpenAI's API."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        embedding = np.array([response.data[0].embedding], dtype=np.float32)
        
        # Normalize for cosine similarity (same as during indexing)
        faiss.normalize_L2(embedding)
        
        return embedding
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: The search query
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of tuples containing (chunk_text, similarity_score)
        """
        # Step 1: Embed the query
        query_embedding = self.embed_query(query)
        
        # Step 2: Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Step 3: Return top k relevant chunks with scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity_score = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity_score))
        
        return results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top k relevant chunks for a query (text only).
        
        Args:
            query: The search query
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of chunk texts
        """
        results = self.search(query, top_k)
        return [chunk for chunk, _ in results]


if __name__ == "__main__":
    # Example usage
    retriever = RAGRetriever(index_path="indexes/faiss_index")
    
    query = "What is machine learning?"
    results = retriever.search(query, top_k=5)
    
    print(f"\nQuery: {query}\n")
    print("Top 5 relevant chunks:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   {chunk[:200]}...")