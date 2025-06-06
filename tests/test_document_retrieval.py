from rag.rag_pipeline import RAGPipeline
import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch
import numpy as np

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestDocumentRetrieval:
    """Tests specifically focused on document retrieval functionality"""

    @pytest.fixture
    def pipeline_with_data(self):
        """Create RAG pipeline with sample data for testing"""
        temp_dir = tempfile.mkdtemp()
        pipeline = RAGPipeline(
            pdfs_folder=os.path.join(temp_dir, "pdfs"),
            index_file=os.path.join(temp_dir, "index.bin"),
            embeddings_file=os.path.join(temp_dir, "embeddings.pkl")
        )

        # Mock documents
        pipeline.documents = [
            {
                "filename": "doc1.pdf",
                "chunk_id": 0,
                "content": "Machine learning is a subset of artificial intelligence",
                "source": "doc1.pdf"
            },
            {
                "filename": "doc1.pdf",
                "chunk_id": 1,
                "content": "Deep learning uses neural networks with multiple layers",
                "source": "doc1.pdf"
            },
            {
                "filename": "doc2.pdf",
                "chunk_id": 0,
                "content": "Natural language processing deals with text and speech",
                "source": "doc2.pdf"
            }
        ]

        # Mock embeddings and index
        pipeline.embeddings = np.random.rand(3, 384).astype(np.float32)
        pipeline.index = Mock()
        pipeline.index.search.return_value = (
            np.array([[0.9, 0.7, 0.5]]),  # scores
            np.array([[0, 1, 2]])         # indices
        )

        return pipeline

    def test_search_similar_documents_returns_correct_format(self, pipeline_with_data):
        """Test that search returns documents in correct format with scores"""
        results = pipeline_with_data.search_similar_documents(
            "machine learning", k=2)

        assert len(results) == 3  # All documents returned due to mock
        assert all(isinstance(item, tuple) and len(
            item) == 2 for item in results)

        # Check document structure
        doc, score = results[0]
        assert 'filename' in doc
        assert 'chunk_id' in doc
        assert 'content' in doc
        assert 'source' in doc
        assert isinstance(score, float)

    def test_search_similar_documents_with_different_k_values(self, pipeline_with_data):
        """Test search with different k values"""
        # Mock different k values
        def mock_search(query_embedding, k):
            if k == 1:
                return (np.array([[0.9]]), np.array([[0]]))
            elif k == 5:
                return (np.array([[0.9, 0.7, 0.5]]), np.array([[0, 1, 2]]))
            return (np.array([[0.9, 0.7]]), np.array([[0, 1]]))

        pipeline_with_data.index.search.side_effect = mock_search

        # Test k=1
        results = pipeline_with_data.search_similar_documents("test", k=1)
        assert len(results) == 1

        # Test k=5 (but only 3 docs available)
        results = pipeline_with_data.search_similar_documents("test", k=5)
        assert len(results) == 3

    def test_search_with_empty_query(self, pipeline_with_data):
        """Test search behavior with empty or whitespace query"""
        results = pipeline_with_data.search_similar_documents("", k=3)
        # Should still work and return results based on empty embedding
        assert isinstance(results, list)

    def test_search_similarity_scores_descending(self, pipeline_with_data):
        """Test that results are returned in descending similarity order"""
        # Mock scores in descending order
        pipeline_with_data.index.search.return_value = (
            np.array([[0.95, 0.85, 0.75]]),
            np.array([[2, 0, 1]])
        )

        results = pipeline_with_data.search_similar_documents("test", k=3)
        scores = [score for _, score in results]

        assert scores == [0.95, 0.85, 0.75]  # Should be in descending order

    def test_query_end_to_end_retrieval(self, pipeline_with_data):
        """Test complete query process including retrieval and response generation"""
        # Mock the generation part
        with patch.object(pipeline_with_data, 'generate_response') as mock_generate:
            mock_generate.return_value = "Generated answer based on retrieved documents"

            result = pipeline_with_data.query("What is machine learning?", k=2)

            assert 'answer' in result
            assert 'sources' in result
            assert result['answer'] == "Generated answer based on retrieved documents"
            assert len(result['sources']) == 3  # Based on mock data

    def test_retrieval_with_no_relevant_documents(self, pipeline_with_data):
        """Test behavior when no relevant documents are found"""
        # Mock no results
        pipeline_with_data.index.search.return_value = (
            np.array([[]]),  # Empty scores
            np.array([[]])   # Empty indices
        )

        results = pipeline_with_data.search_similar_documents(
            "irrelevant query")
        assert results == []

    def test_retrieval_content_preview_generation(self, pipeline_with_data):
        """Test that content previews are generated correctly"""
        result = pipeline_with_data.query("machine learning", k=1)

        for source in result['sources']:
            assert 'content_preview' in source
            # Content preview should be truncated if too long
            if len(pipeline_with_data.documents[0]['content']) > 200:
                assert source['content_preview'].endswith('...')


class TestQueryHandling:
    """Tests specifically focused on query handling functionality"""

    @pytest.fixture
    def pipeline_for_queries(self):
        """Create pipeline specifically for query testing"""
        temp_dir = tempfile.mkdtemp()
        pipeline = RAGPipeline(
            pdfs_folder=os.path.join(temp_dir, "pdfs"),
            index_file=os.path.join(temp_dir, "index.bin"),
            embeddings_file=os.path.join(temp_dir, "embeddings.pkl")
        )

        # Set up mock components
        pipeline.documents = [{"content": "test doc",
                               "filename": "test.pdf", "chunk_id": 0}]
        pipeline.embeddings = np.random.rand(1, 384)
        pipeline.index = Mock()
        pipeline.index.search.return_value = (
            np.array([[0.8]]), np.array([[0]]))

        return pipeline

    def test_query_input_validation(self, pipeline_for_queries):
        """Test query input validation and handling"""
        # Test empty query
        result = pipeline_for_queries.query("", k=5)
        assert 'answer' in result or 'error' in result

        # Test very long query
        long_query = "word " * 1000  # Very long query
        result = pipeline_for_queries.query(long_query, k=5)
        assert 'answer' in result or 'error' in result

    def test_query_k_parameter_validation(self, pipeline_for_queries):
        """Test k parameter handling in queries"""
        # Test various k values
        for k in [1, 5, 10, 100]:
            result = pipeline_for_queries.query("test query", k=k)
            assert 'sources' in result
            # Sources should not exceed available documents
            assert len(result['sources']) <= len(
                pipeline_for_queries.documents)

    def test_query_response_structure(self, pipeline_for_queries):
        """Test that query responses have correct structure"""
        with patch.object(pipeline_for_queries, 'generate_response') as mock_generate:
            mock_generate.return_value = "Test response"

            result = pipeline_for_queries.query("test question", k=3)

            # Check required fields
            assert 'answer' in result
            assert 'sources' in result
            assert isinstance(result['sources'], list)

            # Check source structure
            if result['sources']:
                source = result['sources'][0]
                required_fields = ['filename', 'chunk_id',
                                   'score', 'content_preview']
                assert all(field in source for field in required_fields)

    @patch('google.generativeai.GenerativeModel')
    def test_query_with_context_integration(self, mock_gemini, pipeline_for_queries):
        """Test that queries properly integrate retrieved context"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response based on context"
        mock_model.generate_content.return_value = mock_response
        pipeline_for_queries.model = mock_model

        result = pipeline_for_queries.query("test question", k=2)

        # Verify that generate_content was called with context
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Context:" in call_args
        assert "Question: test question" in call_args

    def test_query_error_handling_no_index(self):
        """Test query error handling when no index is built"""
        temp_dir = tempfile.mkdtemp()
        pipeline = RAGPipeline(
            pdfs_folder=os.path.join(temp_dir, "pdfs"),
            index_file=os.path.join(temp_dir, "index.bin"),
            embeddings_file=os.path.join(temp_dir, "embeddings.pkl")
        )
        # Don't build index

        result = pipeline.query("test question")
        assert 'error' in result
        assert 'Knowledge base not built' in result['error']

    def test_query_error_handling_no_documents(self, pipeline_for_queries):
        """Test query handling when no relevant documents found"""
        # Mock empty search results
        pipeline_for_queries.index.search.return_value = (
            np.array([[]]), np.array([[]]))

        result = pipeline_for_queries.query("test question")
        assert 'error' in result
        assert 'No relevant documents found' in result['error']

    @patch('google.generativeai.GenerativeModel')
    def test_query_generation_error_handling(self, mock_gemini, pipeline_for_queries):
        """Test handling of errors during response generation"""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        pipeline_for_queries.model = mock_model

        result = pipeline_for_queries.query("test question", k=1)

        # Should handle the error gracefully
        assert 'answer' in result
        assert "Error generating response" in result['answer']

    def test_concurrent_queries(self, pipeline_for_queries):
        """Test handling of multiple concurrent queries"""
        import threading
        import time

        results = []

        def query_worker(query_text):
            with patch.object(pipeline_for_queries, 'generate_response') as mock_gen:
                mock_gen.return_value = f"Response to {query_text}"
                result = pipeline_for_queries.query(query_text)
                results.append(result)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=query_worker, args=(f"query {i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All queries should complete successfully
        assert len(results) == 5
        assert all('answer' in result for result in results)
