# test_day6.py

from app.pipeline import prepare_indexes, adaptive_pipeline


# -----------------------------
# Dummy chunker (for testing)
# -----------------------------
class DummyChunker:
    def chunk(self, documents):
        return documents  # no real chunking, just pass-through


# -----------------------------
# Dummy retriever (for testing)
# -----------------------------
class DummyRetriever:
    def __init__(self, name):
        self.name = name
        self.data = []

    def build_index(self, chunks):
        self.data = chunks

    def retrieve(self, query):
        if self.name == "hybrid":
            return ["Self-attention lets transformers relate tokens."]
        return ["Transformers are neural networks."]


# -----------------------------
# Input setup
# -----------------------------
documents = [
    "Self-attention allows transformers to model relationships between tokens.",
    "Transformers are widely used in NLP."
]

strategies = [
    {"chunking": "fixed", "retriever": "dense", "query_opt": "none"},
    {"chunking": "adaptive", "retriever": "hybrid", "query_opt": "rewrite"},
]

chunkers = {
    "fixed": DummyChunker(),
    "adaptive": DummyChunker(),
}

retrievers = {
    "dense": DummyRetriever("dense"),
    "hybrid": DummyRetriever("hybrid"),
}


# -----------------------------
# OFFLINE STEP
# -----------------------------
indexes = prepare_indexes(
    documents=documents,
    strategies=strategies,
    chunkers=chunkers,
    retrievers=retrievers,
)


# -----------------------------
# ONLINE STEP
# -----------------------------
result = adaptive_pipeline(
    query="What is self-attention in transformers?",
    strategies=strategies,
    indexes=indexes,
)

print("\nSelected Strategy:")
print(result["selected_strategy"])
print("Score:", result["score"])
print("Contexts:", result["contexts"])
