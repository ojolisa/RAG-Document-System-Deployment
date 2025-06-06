from rag.rag_pipeline import RAGPipeline
import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestRAGPipeline:
    """Unit tests for RAG Pipeline functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content for testing"""
        return "This is test content from a PDF document. It contains sample text for testing purposes."

    @pytest.fixture
    def rag_pipeline(self, temp_dir):
        """Create RAG pipeline instance for testing"""
        pdfs_folder = os.path.join(temp_dir, "pdfs")
        os.makedirs(pdfs_folder, exist_ok=True)

        index_file = os.path.join(temp_dir, "test_index.bin")
        embeddings_file = os.path.join(temp_dir, "test_embeddings.pkl")

        return RAGPipeline(
            pdfs_folder=pdfs_folder,
            index_file=index_file,
            embeddings_file=embeddings_file
        )

    def test_init(self, temp_dir):
        """Test RAG pipeline initialization"""
        pdfs_folder = os.path.join(temp_dir, "pdfs")
        index_file = os.path.join(temp_dir, "index.bin")
        embeddings_file = os.path.join(temp_dir, "embeddings.pkl")

        pipeline = RAGPipeline(
            pdfs_folder=pdfs_folder,
            index_file=index_file,
            embeddings_file=embeddings_file
        )

        assert pipeline.pdfs_folder == pdfs_folder
        assert pipeline.index_file == index_file
        assert pipeline.embeddings_file == embeddings_file
        assert isinstance(pipeline.embedder, SentenceTransformer)

    def test_clean_text(self, rag_pipeline):
        """Test text cleaning functionality"""
        dirty_text = "  This   is  \n\n  dirty   text!@#$%^&*()  with   extra   spaces  "
        expected = "This is dirty text!() with extra spaces"

        cleaned = rag_pipeline.clean_text(dirty_text)
        assert cleaned == expected

    def test_chunk_text(self, rag_pipeline):
        """Test text chunking functionality"""
        text = " ".join([f"word{i}" for i in range(100)])  # 100 words

        chunks = rag_pipeline.chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        assert len(chunks[0].split()) <= 20

        # Test overlap
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        overlap_found = any(
            word in second_chunk_words[:10] for word in first_chunk_words[-10:])
        assert overlap_found

    @patch('rag.rag_pipeline.PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader, rag_pipeline, temp_dir):
        """Test PDF text extraction"""
        # Create a test PDF file
        test_pdf_path = os.path.join(temp_dir, "test.pdf")
        with open(test_pdf_path, 'wb') as f:
            f.write(b"fake pdf content")

        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = rag_pipeline.extract_text_from_pdf(test_pdf_path)
        assert result == "Sample PDF text content\n"

    def test_create_embeddings(self, rag_pipeline):
        """Test embeddings creation"""
        documents = [
            {"content": "This is document 1"},
            {"content": "This is document 2"}
        ]

        embeddings = rag_pipeline.create_embeddings(documents)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2  # 2 documents
        assert embeddings.shape[1] == 384  # MiniLM embedding dimension

    def test_build_faiss_index(self, rag_pipeline):
        """Test FAISS index building"""
        # Create sample embeddings
        embeddings = np.random.rand(5, 384).astype(np.float32)

        index = rag_pipeline.build_faiss_index(embeddings)

        assert index is not None
        assert index.ntotal == 5  # 5 documents indexed

    @patch('rag.rag_pipeline.os.listdir')
    @patch.object(RAGPipeline, 'extract_text_from_pdf')
    def test_load_and_process_pdfs(self, mock_extract, mock_listdir, rag_pipeline):
        """Test PDF loading and processing"""
        mock_listdir.return_value = ["test1.pdf", "test2.pdf", "not_pdf.txt"]
        mock_extract.return_value = "Sample PDF content for testing"

        documents = rag_pipeline.load_and_process_pdfs()

        assert len(documents) > 0
        for doc in documents:
            assert 'filename' in doc
            assert 'content' in doc
            assert 'chunk_id' in doc
            assert 'source' in doc

    @patch.object(RAGPipeline, 'load_index_and_embeddings')
    @patch.object(RAGPipeline, 'load_and_process_pdfs')
    @patch.object(RAGPipeline, 'create_embeddings')
    @patch.object(RAGPipeline, 'build_faiss_index')
    @patch.object(RAGPipeline, 'save_index_and_embeddings')
    def test_build_knowledge_base(self, mock_save, mock_build_index, mock_create_embeddings,
                                  mock_load_pdfs, mock_load_existing, rag_pipeline):
        """Test knowledge base building"""
        # Test when existing index doesn't exist
        mock_load_existing.return_value = False
        mock_load_pdfs.return_value = [{"content": "test"}]
        mock_create_embeddings.return_value = np.random.rand(1, 384)
        mock_build_index.return_value = Mock()

        rag_pipeline.build_knowledge_base()

        mock_load_pdfs.assert_called_once()
        mock_create_embeddings.assert_called_once()
        mock_build_index.assert_called_once()
        mock_save.assert_called_once()

    def test_search_similar_documents_no_index(self, rag_pipeline):
        """Test search when no index is built"""
        result = rag_pipeline.search_similar_documents("test query")
        assert result == []

    @patch('google.generativeai.GenerativeModel')
    def test_generate_response(self, mock_gemini, rag_pipeline):
        """Test response generation"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        rag_pipeline.model = mock_model

        context_docs = [{"content": "Context document"}]
        result = rag_pipeline.generate_response("Test query", context_docs)

        assert result == "Generated response"

    def test_query_no_index(self, rag_pipeline):
        """Test query when no knowledge base is built"""
        result = rag_pipeline.query("test question")
        assert "error" in result
        assert "Knowledge base not built" in result["error"]

    def test_get_stats_no_documents(self, rag_pipeline):
        """Test stats when no documents are loaded"""
        result = rag_pipeline.get_stats()
        assert "error" in result

    def test_get_stats_with_documents(self, rag_pipeline):
        """Test stats with loaded documents"""
        rag_pipeline.documents = [
            {"filename": "test1.pdf", "content": "content1"},
            {"filename": "test2.pdf", "content": "content2"}
        ]
        rag_pipeline.index = Mock()

        result = rag_pipeline.get_stats()

        assert result["total_documents"] == 2
        assert result["total_files"] == 2
        assert "test1.pdf" in result["files"]
        assert "test2.pdf" in result["files"]
        assert result["index_built"] is True
