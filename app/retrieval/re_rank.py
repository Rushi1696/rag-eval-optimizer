from typing import List
from sentence_transformers import CrossEncoder


class ReRankRetriever:
    """
    Re-ranking Retrieval: Uses cross-encoder to re-rank initially retrieved documents.
    Improves relevance by considering query-document pairs jointly.
    """

    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5, candidate_k: int = 20):
        self.base_retriever = base_retriever
        self.cross_encoder = CrossEncoder(model_name)
        self.top_k = top_k
        self.candidate_k = candidate_k  # Number of candidates to re-rank

    def retrieve(self, query: str) -> List[str]:
        """Retrieve and re-rank documents"""
        # Get initial candidates (more than final top_k)
        candidates = self.base_retriever.retrieve(query) if hasattr(self.base_retriever, 'retrieve') else []

        # If base retriever doesn't have retrieve method, assume it's a list
        if not candidates and isinstance(self.base_retriever, list):
            candidates = self.base_retriever

        # Limit candidates for efficiency
        candidates = candidates[:self.candidate_k]

        if not candidates:
            return []

        # Re-rank using cross-encoder
        pairs = [(query, doc) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Sort by scores (higher is better)
        scored_docs = list(zip(scores, candidates))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top-k re-ranked documents
        top_docs = [doc for _, doc in scored_docs[:self.top_k]]

        print(f"[INFO] Re-ranked {len(candidates)} candidates, selected top {len(top_docs)}")
        return top_docs
