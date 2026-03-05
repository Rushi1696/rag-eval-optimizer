from typing import List
from app.retrieval.dense import DenseRetriever
from app.retrieval.bm25 import BM25Retriever


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
