class RetrieverRouter:
    """
    Routes queries to the correct retriever
    based on strategy.
    """

    def __init__(self, retrievers: dict):
        """
        retrievers = {
            "dense": DenseRetriever,
            "bm25": BM25Retriever,
            "hybrid": HybridRetriever
        }
        """
        self.retrievers = retrievers

    def route(self, strategy: str):
        if strategy not in self.retrievers:
            raise ValueError(
                f"Unknown retrieval strategy: {strategy}. "
                f"Available: {list(self.retrievers.keys())}"
            )

        return self.retrievers[strategy]
