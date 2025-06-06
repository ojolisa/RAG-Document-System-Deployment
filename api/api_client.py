import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"


class RAGAPIClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url

    def upload_document(self, file_path: str):
        """Upload a PDF document to the system"""
        url = f"{self.base_url}/upload"

        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, files=files)

        return response.json()

    def query_documents(self, question: str, k: int = 5):
        """Query the document system"""
        url = f"{self.base_url}/query"

        payload = {
            "question": question,
            "k": k
        }

        response = requests.post(url, json=payload)
        return response.json()

    def get_documents(self, skip: int = 0, limit: int = 100):
        """Get list of processed documents"""
        url = f"{self.base_url}/documents"
        params = {"skip": skip, "limit": limit}

        response = requests.get(url, params=params)
        return response.json()

    def get_document_details(self, document_id: int):
        """Get detailed information about a specific document"""
        url = f"{self.base_url}/documents/{document_id}"

        response = requests.get(url)
        return response.json()

    def delete_document(self, document_id: int):
        """Delete a document"""
        url = f"{self.base_url}/documents/{document_id}"

        response = requests.delete(url)
        return response.json()

    def health_check(self):
        """Check API health"""
        url = f"{self.base_url}/health"

        response = requests.get(url)
        return response.json()


def demo_usage():
    """Demonstrate API usage"""
    client = RAGAPIClient()

    print("=== RAG API Demo ===\n")

    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   RAG Pipeline Loaded: {health['rag_pipeline_loaded']}")

    print("\n" + "="*50 + "\n")

    # Upload document (if PDF exists)
    pdf_files = ["../rag/pdfs/Llms Overview.pdf"]  # Add your PDF files here

    for pdf_file in pdf_files:
        print(f"2. Uploading document: {pdf_file}")
        result = client.upload_document(pdf_file)
        print(f"   Upload successful!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Status: {result['status']}")
        break

    print("\n" + "="*50 + "\n")

    # Wait a moment for processing
    time.sleep(2)

    # Get documents list
    print("3. Getting documents list:")
    documents = client.get_documents()
    print(f"   Total documents: {documents['total_count']}")
    for doc in documents['documents']:
        print(
            f"   - {doc['original_name']} (ID: {doc['id']}, Status: {doc['status']})")

    print("\n" + "="*50 + "\n")

    # Query documents
    questions = [
        "What are large language models?",
        "How do transformers work?",
        "What is attention mechanism?"
    ]

    print("4. Querying documents:")
    for question in questions:
        print(f"\n   Question: {question}")
        result = client.query_documents(question, k=3)
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Sources found: {len(result['sources'])}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")

    print("\n" + "="*50 + "\n")

    # Get document details
    print("5. Getting document details:")
    documents = client.get_documents()
    if documents['documents']:
        doc_id = documents['documents'][0]['id']
        details = client.get_document_details(doc_id)
        doc_info = details['document']
        print(f"   Document: {doc_info['original_name']}")
        print(f"   File size: {doc_info['file_size']} bytes")
        print(f"   Total chunks: {doc_info['total_chunks']}")
        print(f"   Upload date: {doc_info['upload_date']}")
        print(f"   Content preview: {doc_info['content_preview'][:100]}...")
        print(f"   Chunks available: {len(details['chunks'])}")


if __name__ == "__main__":
    demo_usage()
