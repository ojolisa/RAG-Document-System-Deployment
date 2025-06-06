import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
from typing import List, Dict, Tuple
import re


class RAGPipeline:
    def __init__(self, pdfs_folder: str = "pdfs", index_file: str = "faiss_index.bin", embeddings_file: str = "embeddings.pkl"):
        """
        Initialize the RAG pipeline with free open-source components

        Args:
            pdfs_folder: Path to folder containing PDF files
            index_file: Path to save/load FAISS index
            embeddings_file: Path to save/load embeddings and metadata
        """
        # Load environment variables
        load_dotenv()

        # Initialize paths - if relative paths are provided, make them relative to the rag folder
        if not os.path.isabs(pdfs_folder) and pdfs_folder == "pdfs":
            # Default case: use pdfs folder in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.pdfs_folder = os.path.join(script_dir, pdfs_folder)
        else:
            self.pdfs_folder = pdfs_folder

        if not os.path.isabs(index_file) and index_file == "faiss_index.bin":
            # Default case: use rag folder for index file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.index_file = os.path.join(script_dir, index_file)
        else:
            self.index_file = index_file

        if not os.path.isabs(embeddings_file) and embeddings_file == "embeddings.pkl":
            # Default case: use rag folder for embeddings file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.embeddings_file = os.path.join(script_dir, embeddings_file)
        else:
            self.embeddings_file = embeddings_file

        # Initialize embedding model (free)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-2.0-flash')

        # Initialize storage
        self.documents = []
        self.embeddings = None
        self.index = None

        # Create folders if they don't exist
        os.makedirs(pdfs_folder, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()

    def load_and_process_pdfs(self) -> List[Dict]:
        """Load and process all PDFs in the folder"""
        documents = []
        pdf_files = [f for f in os.listdir(
            self.pdfs_folder) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {self.pdfs_folder}")
            return documents

        print(f"Processing {len(pdf_files)} PDF files...")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdfs_folder, pdf_file)
            print(f"Processing: {pdf_file}")

            # Extract text
            text = self.extract_text_from_pdf(pdf_path)

            if not text.strip():
                print(f"No text extracted from {pdf_file}")
                continue

            # Clean text
            cleaned_text = self.clean_text(text)

            # Create chunks
            # Add each chunk as a document
            chunks = self.chunk_text(cleaned_text)
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = {
                        'filename': pdf_file,
                        'chunk_id': i,
                        'content': chunk,
                        'source': pdf_file  # Use filename instead of full path for consistency
                    }
                    documents.append(doc)

        print(f"Created {len(documents)} document chunks")
        return documents

    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Create embeddings for all document chunks"""
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build FAISS index for similarity search"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def save_index_and_embeddings(self):
        """Save FAISS index and document metadata"""
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        # Save documents and embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)

    def load_index_and_embeddings(self) -> bool:
        """Load existing FAISS index and document metadata"""
        if os.path.exists(self.index_file) and os.path.exists(self.embeddings_file):
            # Load FAISS index
            self.index = faiss.read_index(self.index_file)

            # Load documents and embeddings
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']

            return True
        return False

    def build_knowledge_base(self, force_rebuild: bool = False):
        """Build or load the knowledge base"""
        # Load existing index unless forced to rebuild
        if not force_rebuild and self.load_index_and_embeddings():
            print("Loaded existing knowledge base")
            return

        # Process PDFs
        self.documents = self.load_and_process_pdfs()
        if not self.documents:
            print("No documents processed. Please add PDF files to the pdfs folder.")
            return

        # Create embeddings
        self.embeddings = self.create_embeddings(self.documents)

        # Build FAISS index
        self.index = self.build_faiss_index(self.embeddings)

        # Save for future use
        self.save_index_and_embeddings()

    def search_similar_documents(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents using the query"""
        if self.index is None:
            print("Knowledge base not built. Call build_knowledge_base() first.")
            return []        # Create query embedding
        query_embedding = self.embedder.encode([query])

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                # Convert numpy.float32 to Python float
                results.append((self.documents[idx], float(score)))

        return results

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using Gemini API"""
        # Prepare context
        context = "\n\n".join(
            [f"Document {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])

        # Create prompt
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""

        try:
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query(self, question: str, k: int = 5) -> Dict:
        """Query the RAG system"""
        if self.index is None:
            return {"error": "Knowledge base not built. Call build_knowledge_base() first."}

        # Search for similar documents
        similar_docs = self.search_similar_documents(question, k)

        if not similar_docs:
            return {"error": "No relevant documents found"}

        # Extract just the documents for context
        # Generate response
        context_docs = [doc for doc, score in similar_docs]
        response = self.generate_response(question, context_docs)

        return {
            "answer": response,
            "sources": [
                {
                    "filename": doc['filename'],
                    "chunk_id": doc['chunk_id'],
                    # Convert numpy.float32 to Python float
                    "score": float(score),
                    "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                }
                for doc, score in similar_docs
            ]
        }

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        if not self.documents:
            return {"error": "No knowledge base loaded"}

        files = list(set([doc['filename'] for doc in self.documents]))
        return {
            "total_documents": len(self.documents),
            "total_files": len(files),
            "files": files,
            "index_built": self.index is not None
        }
