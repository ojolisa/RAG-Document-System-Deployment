from api.api_client import RAGAPIClient
import pytest
import requests
import tempfile
import os
import sys
from unittest.mock import Mock, patch
import time

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestRAGAPIClient:
    """Integration tests for RAG API Client"""

    @pytest.fixture
    def mock_response(self):
        """Create mock response for requests"""
        mock_resp = Mock()
        mock_resp.json.return_value = {"status": "success"}
        mock_resp.status_code = 200
        return mock_resp

    @pytest.fixture
    def api_client(self):
        """Create API client instance"""
        return RAGAPIClient(base_url="http://localhost:8000")

    @patch('api.api_client.requests.get')
    def test_health_check(self, mock_get, api_client, mock_response):
        """Test health check functionality"""
        mock_response.json.return_value = {
            "status": "healthy",
            "rag_pipeline_loaded": True
        }
        mock_get.return_value = mock_response

        result = api_client.health_check()

        assert result["status"] == "healthy"
        assert result["rag_pipeline_loaded"] is True
        mock_get.assert_called_once_with("http://localhost:8000/health")

    @patch('api.api_client.requests.post')
    def test_upload_document(self, mock_post, api_client, mock_response):
        """Test document upload"""
        mock_response.json.return_value = {
            "message": "Document uploaded successfully",
            "document_id": 1,
            "chunks_created": 5
        }
        mock_post.return_value = mock_response

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"fake pdf content")
            temp_file_path = temp_file.name

        try:
            result = api_client.upload_document(temp_file_path)

            assert result["document_id"] == 1
            assert result["chunks_created"] == 5
            mock_post.assert_called_once()
        finally:
            os.unlink(temp_file_path)

    @patch('api.api_client.requests.post')
    def test_query_documents(self, mock_post, api_client, mock_response):
        """Test document querying"""
        mock_response.json.return_value = {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence",
            "sources": [
                {
                    "filename": "ai_guide.pdf",
                    "chunk_id": 1,
                    "score": 0.85,
                    "content_preview": "AI definition..."
                }
            ],
            "processing_time": 0.5
        }
        mock_post.return_value = mock_response

        result = api_client.query_documents("What is AI?", k=3)

        assert result["question"] == "What is AI?"
        assert result["answer"] == "AI is artificial intelligence"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["score"] == 0.85

        # Verify request payload
        mock_post.assert_called_once_with(
            "http://localhost:8000/query",
            json={"question": "What is AI?", "k": 3}
        )

    @patch('api.api_client.requests.get')
    def test_get_documents(self, mock_get, api_client, mock_response):
        """Test getting document list"""
        mock_response.json.return_value = {
            "documents": [
                {
                    "id": 1,
                    "filename": "test.pdf",
                    "original_name": "test.pdf",
                    "file_size": 1000,
                    "total_chunks": 5,
                    "status": "processed"
                }
            ],
            "total_count": 1
        }
        mock_get.return_value = mock_response

        result = api_client.get_documents(skip=0, limit=10)

        assert result["total_count"] == 1
        assert len(result["documents"]) == 1
        assert result["documents"][0]["filename"] == "test.pdf"

        mock_get.assert_called_once_with(
            "http://localhost:8000/documents",
            params={"skip": 0, "limit": 10}
        )

    @patch('api.api_client.requests.get')
    def test_get_document_details(self, mock_get, api_client, mock_response):
        """Test getting document details"""
        mock_response.json.return_value = {
            "document": {
                "id": 1,
                "filename": "test.pdf",
                "total_chunks": 5
            },
            "chunks": [
                {"chunk_id": 0, "content": "Chunk 0 content"},
                {"chunk_id": 1, "content": "Chunk 1 content"}
            ]
        }
        mock_get.return_value = mock_response

        result = api_client.get_document_details(1)

        assert result["document"]["id"] == 1
        assert len(result["chunks"]) == 2
        mock_get.assert_called_once_with("http://localhost:8000/documents/1")

    @patch('api.api_client.requests.delete')
    def test_delete_document(self, mock_delete, api_client, mock_response):
        """Test document deletion"""
        mock_response.json.return_value = {
            "message": "Document test.pdf deleted successfully"
        }
        mock_delete.return_value = mock_response

        result = api_client.delete_document(1)

        assert "deleted successfully" in result["message"]
        mock_delete.assert_called_once_with(
            "http://localhost:8000/documents/1")


class TestIntegrationScenarios:
    """End-to-end integration test scenarios"""

    @pytest.fixture
    def api_client(self):
        """Create API client for integration tests"""
        return RAGAPIClient(base_url="http://localhost:8000")

    @patch('api.api_client.requests.get')
    @patch('api.api_client.requests.post')
    def test_complete_workflow(self, mock_post, mock_get, api_client):
        """Test complete document upload and query workflow"""
        # Mock health check
        health_response = Mock()
        health_response.json.return_value = {
            "status": "healthy", "rag_pipeline_loaded": True}

        # Mock upload response
        upload_response = Mock()
        upload_response.json.return_value = {
            "document_id": 1,
            "chunks_created": 3,
            "status": "processed"
        }

        # Mock query response
        query_response = Mock()
        query_response.json.return_value = {
            "question": "Test question",
            "answer": "Test answer",
            "sources": [{"filename": "test.pdf", "score": 0.9}]
        }

        # Set up mock returns based on URL
        def mock_get_side_effect(url, **kwargs):
            if url.endswith("/health"):
                return health_response
            return Mock()

        def mock_post_side_effect(url, **kwargs):
            if url.endswith("/upload"):
                return upload_response
            elif url.endswith("/query"):
                return query_response
            return Mock()

        mock_get.side_effect = mock_get_side_effect
        mock_post.side_effect = mock_post_side_effect

        # Test workflow
        # 1. Health check
        health = api_client.health_check()
        assert health["status"] == "healthy"

        # 2. Upload document (create temp file)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"fake pdf content")
            temp_file_path = temp_file.name

        try:
            upload_result = api_client.upload_document(temp_file_path)
            assert upload_result["document_id"] == 1

            # 3. Query documents
            query_result = api_client.query_documents("Test question")
            assert query_result["answer"] == "Test answer"

        finally:
            os.unlink(temp_file_path)

    @patch('api.api_client.requests.post')
    def test_error_handling(self, mock_post, api_client):
        """Test error handling in API client"""
        # Mock error response
        error_response = Mock()
        error_response.json.return_value = {"error": "Something went wrong"}
        error_response.status_code = 400
        mock_post.return_value = error_response

        result = api_client.query_documents("Test question")
        assert "error" in result

    def test_network_timeout(self, api_client):
        """Test handling of network timeouts"""
        with patch('api.api_client.requests.get', side_effect=requests.exceptions.Timeout):
            with pytest.raises(requests.exceptions.Timeout):
                api_client.health_check()

    def test_connection_error(self, api_client):
        """Test handling of connection errors"""
        with patch('api.api_client.requests.get', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(requests.exceptions.ConnectionError):
                api_client.health_check()
