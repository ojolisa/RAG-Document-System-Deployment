#!/bin/bash
set -e

echo "Creating necessary directories..."
mkdir -p rag/pdfs

echo "Starting FastAPI application..."
uvicorn api.api:app --host 0.0.0.0 --port $PORT
