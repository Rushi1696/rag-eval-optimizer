from typing import List, Dict
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# =========================
# 1️⃣ Dense Retriever (FAISS)
# =========================
class DenseRetriever:
    def __init__(self, embedding_model: str, top_k: int = 5):
        self.model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.index = None
        self.texts: List[str] = []

    def build_index(self, documents: List[str]) -> None:
        """
        Builds FAISS index from documents.
        Must be called before retrieval.
        """
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.texts = documents

    def retrieve(self, query: str) -> List[str]:
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")

        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        _, indices = self.index.search(query_emb, self.top_k)
        return [self.texts[i] for i in indices[0]]


# =========================
# 2️⃣ Sparse Retriever (BM25)
# =========================
class BM25Retriever:
    def __init__(self, documents: List[str], top_k: int = 5):
        self.top_k = top_k
        self.documents = documents
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str) -> List[str]:
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [self.documents[i] for i in top_indices]


# =========================
# 3️⃣ Hybrid Retriever
# =========================
class HybridRetriever:
    """
    Combines Dense + Sparse retrieval.
    alpha = 1.0 → dense only
    alpha = 0.0 → bm25 only
    """
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha
        self.top_k = dense_retriever.top_k

    def retrieve(self, query: str) -> List[str]:
        dense_docs = self.dense.retrieve(query)
        bm25_docs = self.bm25.retrieve(query)

        scores: Dict[str, float] = {}

        for doc in dense_docs:
            scores[doc] = scores.get(doc, 0.0) + self.alpha

        for doc in bm25_docs:
            scores[doc] = scores.get(doc, 0.0) + (1 - self.alpha)

        ranked_docs = sorted(
            scores.keys(),
            key=lambda d: scores[d],
            reverse=True
        )

        return ranked_docs[:self.top_k]
