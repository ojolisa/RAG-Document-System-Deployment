from rag.rag_pipeline import RAGPipeline
from api.api import app, get_db, get_rag_pipeline, Base, Document, DocumentChunk
import pytest
import tempfile
import os
import sys
import uuid
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestAPI:
    """Unit and integration tests for API endpoints"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        # Use a unique database file for each test
        db_name = f"./test_{uuid.uuid4().hex}.db"
        
        engine = create_engine(f"sqlite:///{db_name}",
                               connect_args={"check_same_thread": False})
        TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)

        def override_get_db():
            try:
                db = TestingSessionLocal()
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        yield TestingSessionLocal

        # Cleanup - close all connections first
        engine.dispose()
        try:
            if os.path.exists(db_name):
                os.remove(db_name)
        except (FileNotFoundError, PermissionError):
            pass

    @pytest.fixture
    def client(self, temp_db):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAG pipeline for testing"""
        mock_pipeline = Mock(spec=RAGPipeline)
        mock_pipeline.extract_text_from_pdf.return_value = "Sample PDF content"
        mock_pipeline.build_knowledge_base.return_value = None
        mock_pipeline.documents = [{"source": "test.pdf", "chunk_id": 1}]
        mock_pipeline.query.return_value = {
            "answer": "Test answer",
            "sources": [
                {
                    "filename": "test.pdf",
                    "chunk_id": 1,
                    "score": 0.85,
                    "content_preview": "Sample content..."
                }
            ]
        }

        def override_get_rag_pipeline():
            return mock_pipeline

        app.dependency_overrides[get_rag_pipeline] = override_get_rag_pipeline
        yield mock_pipeline
        if get_rag_pipeline in app.dependency_overrides:
            del app.dependency_overrides[get_rag_pipeline]

    @patch('builtins.open')
    @patch('api.api.shutil.copyfileobj')
    @patch('api.api.os.path.getsize')
    @patch('api.api.os.makedirs')
    def test_upload_document_success(self, mock_makedirs, mock_getsize, mock_copy, mock_open, client, mock_rag_pipeline, temp_db):
        """Test successful document upload"""
        mock_getsize.return_value = 1000
        mock_copy.return_value = None
        mock_makedirs.return_value = None
        
        # Mock the file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Clear existing documents to avoid filename conflicts
        db = temp_db()
        db.query(Document).delete()
        db.commit()
        db.close()

        # Create a fake file
        fake_file = ("test.pdf", b"fake pdf content", "application/pdf")

        response = client.post("/upload", files={"file": fake_file})
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Document uploaded and processed successfully"
        assert "document_id" in data
        assert data["status"] == "processed"
