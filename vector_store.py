import os
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class VectorStore:
    """Handles document embeddings and vector search using FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            # many tests patch get_sentence_embedding_dimension
            self.dimension = int(self.model.get_sentence_embedding_dimension())
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents: List[object] = []
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")

    def add_documents(self, documents: List[object]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add (objects with .page_content)
        """
        if not documents:
            return

        try:
            # Accept any object that has a page_content attribute (works with MockDocument)
            texts = [getattr(doc, "page_content", "") for doc in documents]
            embeddings = self.model.encode(texts)

            # Add documents to the store
            self.documents.extend(documents)

            # Add embeddings to the index
            if isinstance(embeddings, np.ndarray):
                arr = embeddings.astype("float32")
            else:
                arr = np.asarray(embeddings, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.size > 0:
                self.index.add(arr)
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")

    def similarity_search(self, query: str, k: int = 4):
        """
        Perform a similarity search for the query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.documents:
            return []

        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            q = np.asarray(query_embedding, dtype="float32")
            if q.ndim == 1:
                q = q.reshape(1, -1)

            # Search the index
            _D, I = self.index.search(q, k)

            # Return the top k documents
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")

    def save_index(self, path: str) -> None:
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, path)

    def load_index(self, path: str) -> None:
        """
        Load a FAISS index from disk. If the path doesn't exist, leave
        the current index unchanged (as tests expect).
        """
        if os.path.exists(path):
            self.index = faiss.read_index(path)