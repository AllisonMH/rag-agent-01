import os
from pathlib import Path
from typing import List
import faiss
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGIngestor:
    def __init__(self, data_dir: str = "data", index_path: str = "faiss_index"):
        """Initialize the RAG ingestion pipeline."""
        self.data_dir = Path(data_dir)
        self.index_path = index_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.dimension = 1536  # Dimension for text-embedding-3-small
        self.chunks = []
        self.index = None
        
    def load_documents(self) -> List[str]:
        """Load all text files from the data directory."""
        documents = []
        for file_path in self.data_dir.glob("**/*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[str], chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[str]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
        
        self.chunks = chunks
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI's API."""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"Generated embeddings for {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """Create and populate a FAISS index."""
        # Using IndexFlatL2 for exact search (cosine similarity can be done with normalized vectors)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        print(f"Added {self.index.ntotal} vectors to FAISS index")
    
    def save_index(self):
        """Save the FAISS index and chunks to disk."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save chunks as metadata
        import pickle
        with open(f"{self.index_path}.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved index to {self.index_path}")
    
    def run_pipeline(self):
        """Execute the full ingestion pipeline."""
        print("Starting RAG ingestion pipeline...")
        
        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            print("No documents found in data directory")
            return
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Step 4: Create FAISS index
        self.create_faiss_index(embeddings)
        
        # Step 5: Save index
        self.save_index()
        
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    ingestor = RAGIngestor(data_dir="data", index_path="indexes/faiss_index")
    ingestor.run_pipeline()

