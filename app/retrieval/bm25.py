from typing import List
import numpy as np
from rank_bm25 import BM25Okapi


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
