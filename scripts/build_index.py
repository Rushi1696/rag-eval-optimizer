"""
Script: Build RAG Index from documents
Pre-processes documents and creates retrieval indexes for efficient querying.
"""

from app.rag import RAGPipeline, load_documents

def build_index(
    data_path: str = "datasets",
    chunk_strategy: str = "semantic",
    retrieval_strategy: str = "hybrid",
    save_path: str = None,
):
    """
    Build and optionally save RAG index
    
    Args:
        data_path: Path to documents folder
        chunk_strategy: 'fixed', 'adaptive', or 'semantic'
        retrieval_strategy: 'dense', 'bm25', or 'hybrid'
        save_path: Optional path to save index metadata
    """
    
    print("\n" + "="*70)
    print("RAG INDEX BUILDER")
    print("="*70)
    
    # Load documents
    print(f"\n[1] Loading documents from: {data_path}")
    documents = load_documents(data_path)
    
    if not documents:
        print(f"[ERROR] No documents found in {data_path}")
        return None
    
    print(f"[OK] Loaded {len(documents)} documents")
    
    # Create pipeline
    print(f"\n[2] Initializing RAG Pipeline")
    print(f"    - Chunk Strategy: {chunk_strategy}")
    print(f"    - Retrieval Strategy: {retrieval_strategy}")
    
    pipeline = RAGPipeline(
        chunk_strategy=chunk_strategy,
        retrieval_strategy=retrieval_strategy,
    )
    
    # Build index
    print(f"\n[3] Building index...")
    pipeline.build_index(documents)
    
    print(f"[OK] Index built successfully")
    print(f"    - Total chunks: {len(pipeline.indexed_chunks)}")
    
    # Save metadata if requested
    if save_path:
        import json
        metadata = {
            "num_documents": len(documents),
            "num_chunks": len(pipeline.indexed_chunks),
            "chunk_strategy": chunk_strategy,
            "retrieval_strategy": retrieval_strategy,
        }
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved to: {save_path}")
    
    return pipeline

if __name__ == "__main__":
    pipeline = build_index()
    
    if pipeline:
        print("\n" + "="*70)
        print("INDEX READY FOR QUERYING")
        print("="*70)
        print("\nNextStep: Use example_rag_pipeline.py or evaluate_strategies.py")
