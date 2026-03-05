"""
Legacy Test: Day 8 - Adaptive Pipeline with Strategy Selection
Updated to use new modular structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import adaptive_pipeline, load_documents
from app.rag.chunking import FixedSizeChunker, SemanticAdaptiveChunker
from app.retrieval import DenseRetriever, BM25Retriever, HybridRetriever

def test_adaptive_pipeline():
    """Test the legacy adaptive_pipeline function with new structure"""
    
    print("\n" + "="*70)
    print("DAY 8 TEST: Adaptive Pipeline with Strategy Selection")
    print("="*70)
    
    # Load documents
    documents = load_documents("datasets")
    
    if not documents:
        print("[WARNING] No documents found in datasets/ folder")
        documents = [{
            "content": "Self attention is a mechanism in transformers. Attention mechanisms help models focus on different parts of the input.",
            "metadata": {"source": "demo", "file_type": "txt"}
        }]
    
    # Initialize chunkers
    chunkers = {
        "fixed": FixedSizeChunker(),
        "semantic": SemanticAdaptiveChunker(),
        "adaptive": FixedSizeChunker(chunk_size=512, overlap=100),
    }
    
    # Initialize retrievers
    dense = DenseRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
    )
    
    bm25_docs = [d["content"] for d in documents]
    bm25 = BM25Retriever(bm25_docs, top_k=5)
    
    hybrid = HybridRetriever(dense, bm25, alpha=0.5)
    
    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }
    
    # Run adaptive pipeline
    query = "What is self attention in transformers?"
    
    print(f"\nQuery: {query}")
    print("\nRunning adaptive pipeline...")
    
    result = adaptive_pipeline(
        query=query,
        documents=documents,
        chunkers=chunkers,
        retrievers=retrievers,
    )
    
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)
    print(f"Selected Strategy: {result['selected_strategy']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Contexts Retrieved: {len(result['contexts'])}")
    
    for i, ctx in enumerate(result['contexts'][:2], 1):
        print(f"\n  Context [{i}]:")
        print(f"  {ctx[:100]}...")

if __name__ == "__main__":
    test_adaptive_pipeline()
