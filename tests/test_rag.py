import pytest
from unittest.mock import patch, MagicMock
from google import genai

from rag import RAG


class TestRAG:

    @patch('rag.genai')  # Patch the google generativeai module used in RAG
    def test_init(self, mock_genai, vector_store):
        """Test RAG initialization with Gemini API."""
        mock_genai.configure = MagicMock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        rag = RAG(vector_store)

        # Verify vector store assigned
        assert rag.vector_store == vector_store

        # Verify configure called once with an API key
        mock_genai.configure.assert_called_once()

        # Verify GenerativeModel called with valid model name
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.5-flash")

        # Verify model assigned
        assert rag.model == mock_model

    def test_generate_answer_no_docs(self, rag_instance):
        """Test generate_answer when no documents are found."""
        rag_instance.vector_store.similarity_search = MagicMock(return_value=[])

        answer = rag_instance.generate_answer("What is AI?")

        rag_instance.vector_store.similarity_search.assert_called_once_with("What is AI?", k=4)
        assert "No relevant information found" in answer

    def test_generate_answer_with_docs(self, rag_instance, sample_documents):
        """Test generate_answer with documents and Gemini API response."""
        rag_instance.vector_store.similarity_search = MagicMock(return_value=sample_documents)

        # Mock model.generate_content response
        mock_generation_response = MagicMock()
        mock_generation_response.generations = [MagicMock(text="Generated answer")]
        rag_instance.model.generate_content = MagicMock(return_value=mock_generation_response)

        query = "What is AI?"
        answer = rag_instance.generate_answer(query, k=2)

        rag_instance.vector_store.similarity_search.assert_called_once_with(query, k=2)
        rag_instance.model.generate_content.assert_called_once()

        assert answer == "Generated answer"

    @patch('dotenv.load_dotenv')
    def test_dotenv_loaded(self, mock_load_dotenv):
        import importlib
        import rag as rag_module
        importlib.reload(rag_module)

        mock_load_dotenv.assert_called_once()
