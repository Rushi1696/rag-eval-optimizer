from typing import List
from app.retrieval.dense import DenseRetriever
from app.optimization.query_optimizer import MultiQueryGenerator


class MultiQueryRetriever:
    """
    Multi-Query Retrieval: Generates multiple query variations
    to improve recall by retrieving from different perspectives.
    """

    def __init__(self, base_retriever: DenseRetriever, query_generator: MultiQueryGenerator, top_k: int = 5):
        self.base_retriever = base_retriever
        self.query_generator = query_generator
        self.top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        """Generate multiple queries and retrieve documents for each"""
        # Generate multiple query variations
        query_variations = self.query_generator.generate(query)
        all_queries = [query] + query_variations  # Include original query

        print(f"[INFO] Multi-query: Generated {len(query_variations)} variations")

        # Retrieve documents for each query
        all_docs = []
        for q in all_queries:
            docs = self.base_retriever.retrieve(q)
            all_docs.extend(docs)

        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)

        # Return top-k unique documents
        return unique_docs[:self.top_k]
