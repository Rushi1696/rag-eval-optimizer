# rag-eval-optimizer
# A production-ready RAG evaluation framework

from app.rag import RAGPipeline, adaptive_pipeline
from app.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from app.evaluation import RAGEvaluator
from app.optimization import StrategySelector

__version__ = "1.0.0"

__all__ = [
    "RAGPipeline",
    "adaptive_pipeline",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "RAGEvaluator",
    "StrategySelector",
]
