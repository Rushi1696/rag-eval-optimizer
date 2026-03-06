"""
Simple FastAPI server for RAG system - Docker ready
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
import time
from pathlib import Path

from app.rag.pipeline import RAGPipeline
from app.config.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG API",
    description="Simple RAG system API for Docker deployment",
    version="1.0.0"
)

# Global RAG pipeline
rag_pipeline = None

# Simple request/response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    processing_time: float

def load_documents():
    """Load documents from datasets directory"""
    config = load_config()
    data_dir = Path(config["documents"]["data_dir"])
    documents = []

    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return documents

    for file_path in data_dir.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({
                "content": content,
                "metadata": {"source": str(file_path.name)}
            })
            logger.info(f"Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    return documents

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global rag_pipeline

    logger.info("Starting simple RAG API...")

    try:
        # Load config and documents
        config = load_config()
        documents = load_documents()

        if not documents:
            logger.warning("No documents found!")
            return

        # Create pipeline
        rag_pipeline = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            generator_model="google/flan-t5-base",
            chunk_strategy="fixed",
            retrieval_strategy="dense"
        )

        # Build index
        rag_pipeline.build_index(documents)
        logger.info(f"API ready! Loaded {len(documents)} documents")

    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.get("/health")
async def health():
    """Health check"""
    if rag_pipeline is None:
        return {"status": "error", "message": "Pipeline not ready"}

    return {
        "status": "healthy",
        "documents": len(rag_pipeline.indexed_documents) if rag_pipeline else 0,
        "chunks": len(rag_pipeline.indexed_chunks) if rag_pipeline else 0
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Simple query endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start_time = time.time()

    try:
        result = rag_pipeline.query(request.question)
        processing_time = time.time() - start_time

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)