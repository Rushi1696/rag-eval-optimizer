from app.pipeline import adaptive_pipeline
from app.chunking import FixedSizeChunker, SemanticAdaptiveChunker
from app.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from app.chunking import load_documents

documents = load_documents("datasets")

chunkers = {
    "fixed": FixedSizeChunker(),
    "semantic": SemanticAdaptiveChunker(),
    "adaptive": FixedSizeChunker(chunk_size=512, overlap=100),
}

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

query = "What is self attention in transformers?"

result = adaptive_pipeline(
    query=query,
    documents=documents,
    chunkers=chunkers,
    retrievers=retrievers,
)

print("\nFINAL OUTPUT")
print(result)
