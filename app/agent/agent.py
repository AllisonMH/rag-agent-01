import os
from typing import List, Optional
from openai import OpenAI
import sys
from pathlib import Path
from observability.tracing import trace_function, trace_llm_call, trace_retrieval, TracingContext

# Add parent directory to path to import retriever
sys.path.append(str(Path(__file__).parent.parent))
from rag.retriever import RAGRetriever


class RAGAgent:
    def __init__(
        self, 
        index_path: str = "indexes/faiss_index",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            index_path: Path to the FAISS index
            model: OpenAI model to use for generation
            temperature: Temperature for response generation
            top_k: Number of chunks to retrieve
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retriever = RAGRetriever(index_path=index_path)
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Use the context information to answer the user's question accurately and concisely.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite information from the context when possible."""

    def create_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Create a prompt with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
        
        Returns:
            Formatted prompt string
        """
        # Format context
        context = "\n\n".join([
            f"Context {i+1}:\n{chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create user prompt with context and query
        user_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        return user_prompt
    
    def generate_response(
        self, 
        query: str, 
        context_chunks: List[str],
        stream: bool = False
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            stream: Whether to stream the response
        
        Returns:
            Generated response text
        """
        user_prompt = self.create_prompt(query, context_chunks)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if stream:
            return self._stream_response(messages)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
    
    def _stream_response(self, messages: List[dict]) -> str:
        """Stream response from OpenAI API."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()  # New line after streaming
        return full_response
    
    @trace_function("rag_agent.query")
    def query(
        self, 
        question: str, 
        top_k: Optional[int] = None,
        stream: bool = False,
        return_context: bool = False
    ) -> dict:
        """Main query method that combines retrieval + LLM."""
        k = top_k or self.top_k
        
        # Step 1: Retrieve context with tracing
        with TracingContext("retrieval", query_length=len(question), top_k=k):
            print(f"Retrieving top {k} relevant chunks...")
            context_chunks = self.retriever.retrieve(question, top_k=k)
            trace_retrieval(question, k, len(context_chunks))
        
        if not context_chunks:
            return {
                "response": "I couldn't find relevant information to answer your question.",
                "context": [] if return_context else None
            }
        
        # Step 2-3: Generate response with tracing
        with TracingContext("llm_generation", model=self.model):
            print(f"Generating response using {self.model}...\n")
            response = self.generate_response(question, context_chunks, stream=stream)
            trace_llm_call(self.model)
        
        # Step 4: Return response
        result = {"response": response}
        if return_context:
            result["context"] = context_chunks
        
        return result
    
    def chat(self):
        """Interactive chat loop."""
        print("RAG Agent initialized. Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print(f"\nAgent: ", end="")
                result = self.query(question, stream=True)
                print("\n" + "-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    # Example usage
    agent = RAGAgent(
        index_path="indexes/faiss_index",
        model="gpt-4o",
        temperature=0.7,
        top_k=5
    )
    
    # Option 1: Single query
    question = "What is machine learning?"
    result = agent.query(question, return_context=True)
    print(f"Question: {question}\n")
    print(f"Answer: {result['response']}\n")
    
    # Option 2: Interactive chat
    # agent.chat()