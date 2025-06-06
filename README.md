# RAG Document System

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using AI. Upload PDFs, build a searchable knowledge base, and ask questions to get AI-generated answers with source citations.

## Setup and Installation

### Prerequisites
- Python 3.11 or higher
- Git

### Local Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd RAG-Document-System
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Create required directories**
   ```powershell
   mkdir rag\pdfs
   ```

6. **Start the API server**
   ```powershell
   python api\api.py
   ```

7. **Launch the web interface** (optional)
   ```powershell
   cd frontend
   python server.py
   ```
   Access at: http://localhost:3000

## Deployment

### Deploy on Render (Cloud)

For easy cloud deployment, see the detailed [Render Deployment Guide](RENDER_DEPLOYMENT.md).

**Quick Deploy:**
1. Push your code to GitHub
2. Connect to Render
3. Set `GEMINI_API_KEY` environment variable
4. Deploy automatically with included `render.yaml`

### Docker Deployment

### Docker Installation

1. **Set environment variable**
   ```powershell
   $env:GEMINI_API_KEY="your_gemini_api_key_here"
   ```

2. **Run with Docker Compose**
   ```powershell
   docker-compose up -d
   ```

The API will be available at: http://localhost:8000

## API Usage and Testing

### Core Endpoints

#### Health Check
```http
GET /health
```
Check if the API and RAG pipeline are running properly.

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF file>
```

#### Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "Your question here",
  "k": 5
}
```

#### List Documents
```http
GET /documents?skip=0&limit=100
```

#### Get Document Details
```http
GET /documents/{document_id}
```

#### Delete Document
```http
DELETE /documents/{document_id}
```

### Using the API Client

```python
from api.api_client import RAGAPIClient

# Initialize client
client = RAGAPIClient(base_url="http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Upload a document
result = client.upload_document("path/to/your/document.pdf")
print(result)

# Query documents
response = client.query_documents("What is artificial intelligence?", k=5)
print(response["answer"])
print(response["sources"])

# List all documents
documents = client.get_documents()
print(documents)
```

### Testing

Run the test suite:
```powershell
# Install test dependencies
pip install -r tests\requirements-test.txt

# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest tests\test_api.py # Specific test file
```

### Web Interface

Access the web interface at http://localhost:3000 (if running frontend server) or use the API directly at http://localhost:8000.

Features:
- Document upload with drag-and-drop
- Real-time API connection status
- Document management (view, delete)
- Interactive querying with source citations
- Adjustable number of retrieved sources (k parameter)

## Configuration

### LLM Provider Configuration

The system currently uses Google's Gemini API as the default LLM provider. Configuration is handled through environment variables.

#### Google Gemini (Default)

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Set Environment Variable**: 
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Model Configuration**: The system uses `gemini-2.0-flash` by default. To change the model, modify `rag/rag_pipeline.py`:
   ```python
   self.model = genai.GenerativeModel('models/gemini-pro')  # or other model
   ```

#### Adding Other LLM Providers

To add support for other providers (OpenAI, Anthropic, etc.), modify the `RAGPipeline` class in `rag/rag_pipeline.py`:

**Example - OpenAI Integration:**
```python
# Add to imports
import openai

# In __init__ method
if provider == "openai":
    openai.api_key = os.getenv('OPENAI_API_KEY')
    self.client = openai.OpenAI()

# In generate_answer method
if self.provider == "openai":
    response = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### Embedding Model Configuration

The system uses `all-MiniLM-L6-v2` from Sentence Transformers for embeddings. To change:

```python
# In rag/rag_pipeline.py
self.embedder = SentenceTransformer('your-preferred-model')
```

Popular alternatives:
- `all-mpnet-base-v2` (better quality, slower)
- `all-distilroberta-v1` (balanced)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes | None |
| `FAISS_INDEX_PATH` | Path to FAISS index file | No | `rag/faiss_index.bin` |
| `EMBEDDINGS_PATH` | Path to embeddings file | No | `rag/embeddings.pkl` |
| `PDFS_FOLDER` | Path to PDFs directory | No | `rag/pdfs` |

### Database Configuration

The system uses SQLite by default. The database file is created automatically at `documents.db`. To use a different database, modify the connection string in `api/api.py`:

```python
SQLALCHEMY_DATABASE_URL = "sqlite:///./documents.db"  # SQLite
# SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@localhost/dbname"  # PostgreSQL
```

---

**Need help?** Check the logs for error messages or run the health check endpoint to diagnose issues.
