from app.chunking import AdaptiveChunker, load_documents
from app.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from app.query_optimizer import (
    QueryRewriter,
    MultiQueryGenerator,
    SelfReflectionQueryAgent
)
from app.router import RetrieverRouter, QueryOptimizerRouter
from app.pipeline import run_pipeline


# -------------------------
# 1️⃣ Load documents
# -------------------------
docs = load_documents("datasets")
texts = [d["content"] for d in docs]

print("\nDocuments loaded:")
for t in texts:
    print("-", t[:60])


# -------------------------
# 2️⃣ Chunk documents
# -------------------------
chunker = AdaptiveChunker(chunk_size=128, overlap=20)
chunks = chunker.chunk(texts)

print(f"\nTotal chunks: {len(chunks)}")


# -------------------------
# 3️⃣ Build retrievers
# -------------------------
dense = DenseRetriever(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=3
)
dense.build_index(chunks)

bm25 = BM25Retriever(chunks, top_k=3)
hybrid = HybridRetriever(dense, bm25, alpha=0.5)

retriever_router = RetrieverRouter({
    "dense": dense,
    "bm25": bm25,
    "hybrid": hybrid
})


# -------------------------
# 4️⃣ Query optimizers
# -------------------------
query_router = QueryOptimizerRouter({
    "none": None,
    "rewrite": QueryRewriter(),
    "multi_query": MultiQueryGenerator(),
    "self_reflection": SelfReflectionQueryAgent()
})


# -------------------------
# 5️⃣ Run test query
# -------------------------
query = "Explain attention"

print("\nOriginal Query:", query)

results = run_pipeline(
    query=query,
    retriever=retriever_router.route("hybrid"),
    query_optimizer=query_router,
    query_strategy="rewrite"   # try: none | rewrite | multi_query | self_reflection
)

print("\nRetrieved Chunks:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r}")
