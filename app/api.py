"""
FastAPI application for RAG Evaluation & Optimization Framework
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import time
from pathlib import Path
import json

from app.rag.pipeline import RAGPipeline
from app.config.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Evaluation & Optimization API",
    description="Production-ready RAG system with multiple retrieval strategies and automatic evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")
    strategy: Optional[str] = Field("hybrid", description="Retrieval strategy: dense, bm25, hybrid")
    top_k: Optional[int] = Field(5, description="Number of contexts to retrieve")

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    metrics: Dict[str, float]
    strategy: str
    processing_time: float

class EvaluateRequest(BaseModel):
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    contexts: List[str] = Field(..., description="Retrieved contexts")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer for comparison")

class EvaluateResponse(BaseModel):
    question: str
    answer: str
    metrics: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    message: str
    pipeline_loaded: bool
    documents_count: int
    chunks_count: int

class StrategyResponse(BaseModel):
    available_strategies: List[str]
    current_strategy: str

def load_documents() -> List[Dict]:
    """Load documents from datasets directory"""
    config = load_config()
    data_dir = Path(config["documents"]["data_dir"])
    file_types = config["documents"]["file_types"]

    documents = []

    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return documents

    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix[1:] in file_types:
            try:
                with open(file_path, 'r', encoding=config["documents"]["encoding"]) as f:
                    content = f.read()

                documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(file_path.relative_to(data_dir)),
                        "file_type": file_path.suffix[1:],
                        "file_size": file_path.stat().st_size
                    }
                })
                logger.info(f"Loaded document: {file_path.name}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    return documents

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG API server (without loading models)"""
    logger.info("Starting RAG API server...")
    logger.info("RAG API server started successfully! Use /initialize to load the pipeline.")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the RAG pipeline on startup"""
#     global rag_pipeline

#     logger.info("Starting RAG API server...")

#     try:
#         # Load configuration
#         config = load_config()

#         # Initialize pipeline
#         rag_pipeline = RAGPipeline(
#             embedding_model=config["retrieval"]["dense"]["embedding_model"],
#             generator_model="google/flan-t5-base",  # Could be configurable
#             chunk_strategy=config["chunking"]["strategy"],
#             retrieval_strategy=config["retrieval"]["strategy"]
#         )

#         # Load and index documents
#         logger.info("Loading documents...")
#         documents = load_documents()

#         if documents:
#             logger.info(f"Building index from {len(documents)} documents...")
#             rag_pipeline.build_index(documents)
#             logger.info("Index built successfully!")
#         else:
#             logger.warning("No documents found. Pipeline will need manual indexing.")

#         logger.info("RAG API server started successfully!")

#     except Exception as e:
#         logger.error(f"Failed to initialize RAG pipeline: {e}")
#         # Don't raise - allow server to start without pipeline
#         # raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        return HealthResponse(
            status="error",
            message="Pipeline not initialized",
            pipeline_loaded=False,
            documents_count=0,
            chunks_count=0
        )

    return HealthResponse(
        status="healthy",
        message="RAG system is operational",
        pipeline_loaded=True,
        documents_count=len(rag_pipeline.indexed_documents),
        chunks_count=len(rag_pipeline.indexed_chunks)
    )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    start_time = time.time()

    try:
        # Switch strategy if requested
        if request.strategy and request.strategy != rag_pipeline.retrieval_strategy:
            rag_pipeline.set_retrieval_strategy(request.strategy)

        # Perform query
        result = rag_pipeline.query(request.question)

        processing_time = time.time() - start_time

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            contexts=result["contexts"],
            metrics=result["metrics"],
            strategy=rag_pipeline.retrieval_strategy,
            processing_time=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_answer(request: EvaluateRequest):
    """Evaluate a RAG answer"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    start_time = time.time()

    try:
        # Evaluate using RAGAS or heuristics
        metrics = rag_pipeline.evaluate_answer(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        processing_time = time.time() - start_time

        return EvaluateResponse(
            question=request.question,
            answer=request.answer,
            metrics=metrics,
            processing_time=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/strategies", response_model=StrategyResponse)
async def get_strategies():
    """Get available retrieval strategies"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    return StrategyResponse(
        available_strategies=["dense", "bm25", "hybrid"],
        current_strategy=rag_pipeline.retrieval_strategy
    )

@app.post("/strategies/{strategy}")
async def set_strategy(strategy: str):
    """Set the retrieval strategy"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    if strategy not in ["dense", "bm25", "hybrid"]:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    try:
        rag_pipeline.set_retrieval_strategy(strategy)
        return {"message": f"Strategy changed to {strategy}", "current_strategy": strategy}
    except Exception as e:
        logger.error(f"Failed to set strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set strategy: {str(e)}")

@app.post("/initialize")
async def initialize_pipeline():
    """Initialize the RAG pipeline and build index"""
    global rag_pipeline

    if rag_pipeline is not None:
        return {"message": "Pipeline already initialized", "status": "ready"}

    try:
        logger.info("Initializing RAG pipeline...")

        # Load configuration
        config = load_config()

        # Initialize pipeline
        rag_pipeline = RAGPipeline(
            embedding_model=config["retrieval"]["dense"]["embedding_model"],
            generator_model="google/flan-t5-base",
            chunk_strategy=config["chunking"]["strategy"],
            retrieval_strategy=config["retrieval"]["strategy"]
        )

        # Load and index documents
        logger.info("Loading documents...")
        documents = load_documents()

        if documents:
            logger.info(f"Building index from {len(documents)} documents...")
            rag_pipeline.build_index(documents)
            logger.info("Index built successfully!")
            return {
                "message": "Pipeline initialized successfully",
                "status": "ready",
                "documents_loaded": len(documents),
                "chunks_created": len(rag_pipeline.indexed_chunks)
            }
        else:
            return {
                "message": "Pipeline initialized but no documents found",
                "status": "ready",
                "documents_loaded": 0,
                "chunks_created": 0
            }

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)