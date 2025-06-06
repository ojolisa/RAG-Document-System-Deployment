# Simple Dockerfile for RAG Document System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p rag/pdfs && \
    mkdir -p data

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "api/api.py"]
