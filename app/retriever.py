# app/retriever.py

from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class DenseRetriever:
    def __init__(self, embedding_model: str, top_k: int = 5):
        self.model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.index = None
        self.texts = []

    def build_index(self, documents: List[str]):
        embeddings = self.model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.texts = documents

    def retrieve(self, query: str) -> List[str]:
        if self.index is None:
            raise RuntimeError("Dense index not built")

        query_emb = self.model.encode([query]).astype("float32")
        _, indices = self.index.search(query_emb, self.top_k)
        return [self.texts[i] for i in indices[0]]


class BM25Retriever:
    def __init__(self, documents: List[str], top_k: int = 5):
        self.top_k = top_k
        self.documents = documents
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> List[str]:
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        return [self.documents[i] for i in top_idx]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, alpha: float = 0.5):
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha

    def retrieve(self, query: str) -> List[str]:
        dense_docs = self.dense.retrieve(query)
        bm25_docs = self.bm25.retrieve(query)

        scores = {}
        for d in dense_docs:
            scores[d] = scores.get(d, 0) + self.alpha
        for d in bm25_docs:
            scores[d] = scores.get(d, 0) + (1 - self.alpha)

        return sorted(scores, key=scores.get, reverse=True)
