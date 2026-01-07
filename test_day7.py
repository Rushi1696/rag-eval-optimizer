# test_day7.py

from app.pipeline import prepare_indexes, adaptive_pipeline


# Dummy chunker
class DummyChunker:
    def chunk(self, documents):
        return documents


# Dummy retriever
class DummyRetriever:
    def __init__(self, name):
        self.name = name
        self.data = []

    def build_index(self, chunks):
        self.data = chunks

    def retrieve(self, query):
        if self.name == "hybrid":
            return [
                "Self-attention allows transformers to weigh relationships between tokens."
            ]
        return ["Transformers are neural networks."]


documents = [
    "Self-attention allows transformers to model relationships between tokens.",
    "Transformers are widely used in NLP tasks."
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

# OFFLINE
indexes = prepare_indexes(documents, strategies, chunkers, retrievers)

# ONLINE
result = adaptive_pipeline(
    query="What is self-attention in transformers?",
    strategies=strategies,
    indexes=indexes,
)

print("\nSelected Strategy:")
print(result["selected_strategy"])

print("\nGenerated Answer:")
print(result["answer"])
