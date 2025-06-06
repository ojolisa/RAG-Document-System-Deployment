# Add the parent directory to Python path for proper imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from datetime import datetime
import shutil
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from rag.rag_pipeline import RAGPipeline
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# Database setup
SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./documents.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={
                       "check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_name = Column(String)
    file_size = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    content_preview = Column(Text)
    total_chunks = Column(Integer)
    # processed, error, processing
    status = Column(String, default="processed")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    chunk_id = Column(Integer)
    content = Column(Text)
    embedding_dimension = Column(Integer)


# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models for API


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    processing_time: Optional[float] = None


class DocumentMetadata(BaseModel):
    id: int
    filename: str
    original_name: str
    file_size: int
    upload_date: datetime
    content_preview: str
    total_chunks: int
    status: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total_count: int


# FastAPI app
app = FastAPI(
    title="RAG Document System API",
    description="REST API for document upload, processing, and querying using RAG pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment, or specify your Render frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Initialize RAG pipeline
rag_pipeline = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        # Use the correct paths for all RAG components in the rag folder
        pdfs_path = os.path.join("..", "rag", "pdfs")
        index_path = os.path.join("..", "rag", "faiss_index.bin")
        embeddings_path = os.path.join("..", "rag", "embeddings.pkl")
        
        rag_pipeline = RAGPipeline(
            pdfs_folder=pdfs_path,
            index_file=index_path,
            embeddings_file=embeddings_path
        )
        # Load existing knowledge base
        rag_pipeline.build_knowledge_base(force_rebuild=False)
    return rag_pipeline


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    # Ensure required directories exist
    os.makedirs("rag/pdfs", exist_ok=True)
    os.makedirs("rag", exist_ok=True)
    
    # Initialize RAG pipeline
    get_rag_pipeline()

# API Endpoints


@app.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload a document and process it for RAG system
    """    # Validate file type
    # Only PDF files are supported
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "Only PDF files are supported"}

    # Create unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join("..", "rag", "pdfs", safe_filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    # Extract content preview
    full_text = pipeline.extract_text_from_pdf(file_path)
    content_preview = full_text[:500] + \
        "..." if len(full_text) > 500 else full_text

    # Create database record
    db_document = Document(
        filename=safe_filename,
        original_name=file.filename,
        file_size=file_size,
        content_preview=content_preview,
        total_chunks=0,        status="processing"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Process document with RAG pipeline
    # Rebuild knowledge base to include new document
    pipeline.build_knowledge_base(force_rebuild=True)    # Count chunks for this document
    chunks_count = len(
        [doc for doc in pipeline.documents if doc['source'] == safe_filename])

    # Update database record
    db_document.total_chunks = chunks_count
    db_document.status = "processed"
    db.commit()    # Store chunk information
    for doc in pipeline.documents:
        if doc['source'] == safe_filename:
            chunk = DocumentChunk(
                document_id=db_document.id,
                chunk_id=doc['chunk_id'],
                content=doc['content'],
                embedding_dimension=384  # all-MiniLM-L6-v2 dimension
            )
            db.add(chunk)
    db.commit()

    return {
        "message": "Document uploaded and processed successfully",
        "document_id": db_document.id,
        "filename": safe_filename,
        "chunks_created": chunks_count,
        "status": "processed"
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Query the document system using RAG pipeline
    """
    import time
    start_time = time.time()

    # Perform RAG query
    result = pipeline.query(query_request.question, k=query_request.k)

    processing_time = time.time() - start_time    # Check if there was an error in the RAG pipeline
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    # Convert any numpy types to Python native types for proper serialization
    sources = convert_numpy_types(result['sources'])

    return QueryResponse(
        question=query_request.question,
        answer=result['answer'],
        sources=sources,
        processing_time=processing_time
    )


@app.get("/documents", response_model=DocumentListResponse)
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get list of processed documents with metadata
    """
    # Get total count
    total_count = db.query(Document).count()

    # Get documents with pagination
    documents = db.query(Document).offset(skip).limit(limit).all()

    # Convert to response format
    document_list = []
    for doc in documents:
        document_list.append(DocumentMetadata(
            id=doc.id,
            filename=doc.filename,
            original_name=doc.original_name,
            file_size=doc.file_size,
            upload_date=doc.upload_date,
            content_preview=doc.content_preview,
            total_chunks=doc.total_chunks,
            status=doc.status
        ))

    return DocumentListResponse(
        documents=document_list,
        total_count=total_count
    )


@app.get("/documents/{document_id}")
async def get_document_details(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific document
    """
    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return {"error": "Document not found"}

    # Get document chunks
    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id).all()

    return {
        "document": {
            "id": document.id,
            "filename": document.filename,
            "original_name": document.original_name,
            "file_size": document.file_size,
            "upload_date": document.upload_date,
            "content_preview": document.content_preview,
            "total_chunks": document.total_chunks,
            "status": document.status
        },
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            for chunk in chunks
        ]
    }


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Delete a document and its associated data
    """
    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return {"error": "Document not found"}

    # Delete physical file
    file_path = os.path.join("..", "rag", "pdfs", document.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete chunks from database
    db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id).delete()

    # Delete document from database
    db.delete(document)
    db.commit()

    # Rebuild knowledge base to remove document from RAG pipeline
    pipeline.build_knowledge_base(force_rebuild=True)

    return {"message": f"Document {document.original_name} deleted successfully"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "rag_pipeline_loaded": rag_pipeline is not None
    }


@app.get("/")
async def root():
    """
    Serve the main frontend page
    """
    return FileResponse('../frontend/index.html')


@app.get("/api")
async def api_info():
    """
    API information endpoint
    """
    return {
        "message": "RAG Document System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload a PDF document",
            "query": "POST /query - Query documents using RAG",
            "documents": "GET /documents - List all documents",
            "document_details": "GET /documents/{id} - Get document details",
            "delete_document": "DELETE /documents/{id} - Delete a document",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
