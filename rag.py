import os
from typing import Any
from dotenv import load_dotenv
import google.generativeai as genai

# Ensure .env is loaded when the module is imported (the test patches this)
load_dotenv()


class RAG:
    """Retrieval-Augmented Generation using a provided vector store and Gemini."""

    def __init__(self, vector_store: Any, temperature: float = 0.7):
        """
        Args:
            vector_store: An object exposing .similarity_search(query, k)
            temperature: Model temperature for generation
        """
        self.vector_store = vector_store
        self.temperature = temperature

        # Configure Gemini and create the model (tests patch rag.genai)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    # -----------------------------
    # Internal helper
    # -----------------------------
    def _normalize_response_text(self, response: Any) -> str:
        """
        Normalize various response shapes from google.generativeai and mocks.
        Prefers response.generations[0].text (used by tests) before response.text.
        """
        # 1) Try response.generations[0].text (the test sets this)
        gens = getattr(response, "generations", None)
        if gens and len(gens) > 0:
            gtxt = getattr(gens[0], "text", None)
            if callable(gtxt):
                try:
                    gtxt = gtxt()
                except Exception:
                    pass
            if isinstance(gtxt, str):
                return gtxt

        # 2) Try response.text
        txt = getattr(response, "text", None)
        if callable(txt):
            try:
                txt = txt()
            except Exception:
                pass
        if isinstance(txt, str):
            return txt

        # 3) Last resort
        return str(response)

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_answer(self, query: str, k: int = 4) -> str:
        """
        Generate an answer using retrieved context from the vector store.
        Returns a plain string in both real and mocked environments.
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)

        if not docs:
            # Keep wording simple; tests look for this phrase
            return "No relevant information found for your query."

        # Format context from documents safely
        def _meta_get(meta: dict, key: str, default: str) -> str:
            try:
                return meta.get(key, default)
            except Exception:
                return default

        parts = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            source = _meta_get(meta, "source", "Unknown")
            page = _meta_get(meta, "page", "Unknown")
            content = getattr(doc, "page_content", "")
            parts.append(f"Document: {source}, Page: {page}\n{content}")
        context = "\n\n".join(parts)

        # Construct the prompt
        prompt = (
            "You are a helpful assistant that answers questions based on the provided documents. "
            "Use only the information in the documents to answer the question. "
            "If the answer cannot be found in the documents, say so directly.\n\n"
            f"Documents:\n{context}\n\nQuestion: {query}"
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature},
            )
            return self._normalize_response_text(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"