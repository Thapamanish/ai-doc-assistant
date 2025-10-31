import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Optional
import numpy as np

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG


class MockDocument:
    """Lightweight mock document for testing."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


@pytest.fixture
def sample_documents() -> List[MockDocument]:
    """Create sample documents for testing."""
    return [
        MockDocument(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        MockDocument(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"source": "test1.pdf", "page": 2}
        ),
        MockDocument(
            page_content="RAG combines retrieval with generative models.",
            metadata={"source": "test2.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def mock_pdf_file():
    """Create a temporary mock PDF file for testing."""
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    yield temp_file_path
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    with patch('vector_store.SentenceTransformer', autospec=True) as mock:
        instance = MagicMock()
        instance.get_sentence_embedding_dimension.return_value = 384
        instance.encode = MagicMock(return_value=np.array([
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
            ]))
        mock.return_value = instance
        yield mock


@pytest.fixture
def vector_store(mock_sentence_transformer) -> VectorStore:
    """Create a VectorStore instance with mocked embeddings for testing."""
    return VectorStore()


@pytest.fixture
def mock_gemini():
    """Create a mock Gemini client for testing (matches google.generativeai)."""
    with patch('rag.genai.GenerativeModel', autospec=True) as MockModel:
        model_instance = MockModel.return_value
        # Prepare a mock response object
        mock_response = MagicMock()
        # Tests in test_rag expect generations[0].text = "Generated answer"
        mock_response.generations = [MagicMock(text="Generated answer")]
        model_instance.generate_content.return_value = mock_response
        yield MockModel


@pytest.fixture
def rag_instance(vector_store, mock_gemini) -> RAG:
    """Create a RAG instance with a mocked vector store and Gemini model."""
    return RAG(vector_store)