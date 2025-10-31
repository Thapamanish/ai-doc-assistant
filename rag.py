import os
from typing import cast
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class RAG:
    """Retrieval Augmented Generation for question answering using Gemini."""

    def __init__(self, vector_store, temperature: float = 0.7):
        """
        Initialize the RAG system.

        Args:
            vector_store: Vector store instance
            temperature: Model temperature for response generation
        """
        self.vector_store = vector_store
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")  # or gemini-1.5-pro
        self.temperature = temperature

    def generate_answer(self, query: str, k: int = 4) -> str:
        """
        Generate an answer to a query using RAG.

        Args:
            query: User query
            k: Number of top documents to retrieve

        Returns:
            Generated answer
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)

        if not docs:
            return "No relevant information found. Please try a different question or upload documents."

        # Format context from documents
        context = "\n\n".join(
            [
                f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}"
                for doc in docs
            ]
        )

        # Construct prompt
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided documents.
        Use only the information in the documents to answer the question. 
        If the answer cannot be found in the documents, say so directly.

        Documents:
        {context}

        Question: {query}
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )
            return cast(str, response.text)
        except Exception as e:
            return f"Error generating response: {str(e)}"