import pytest
import sys
import os

# Test configuration and fixtures

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Pytest configuration


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests"""
    # Set environment variables for testing
    os.environ['GOOGLE_API_KEY'] = 'test_api_key'
    os.environ['GEMINI_API_KEY'] = 'test_api_key'

    yield

    # Cleanup after tests
    pass


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return """
    Artificial Intelligence and Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that focuses on 
    the development of algorithms and statistical models that enable computer 
    systems to improve their performance on a specific task through experience.
    
    Deep Learning
    Deep learning is a subset of machine learning that uses neural networks 
    with multiple layers (hence "deep") to model and understand complex patterns 
    in data.
    
    Natural Language Processing
    Natural language processing (NLP) is a field of AI that gives computers 
    the ability to understand, interpret, and manipulate human language.
    """


@pytest.fixture
def sample_query_responses():
    """Sample query responses for testing"""
    return {
        "ai_question": {
            "question": "What is artificial intelligence?",
            "answer": "Artificial intelligence (AI) is a field of computer science...",
            "sources": [
                {
                    "filename": "ai_guide.pdf",
                    "chunk_id": 0,
                    "score": 0.95,
                    "content_preview": "Artificial intelligence is..."
                }
            ]
        },
        "ml_question": {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence...",
            "sources": [
                {
                    "filename": "ml_basics.pdf",
                    "chunk_id": 1,
                    "score": 0.88,
                    "content_preview": "Machine learning focuses on..."
                }
            ]
        }
    }
