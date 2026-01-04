class RetrieverRouter:
    """
    Routes queries to the correct retriever based on strategy.
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


class QueryOptimizerRouter:
    """
    Routes query through the selected query optimization strategy.
    """

    def __init__(self, optimizers: dict):
        """
        optimizers = {
            "none": None,
            "rewrite": QueryRewriter(),
            "multi_query": MultiQueryGenerator(),
            "self_reflection": SelfReflectionQueryAgent()
        }
        """
        self.optimizers = optimizers

    def optimize(self, query: str, strategy: str):
        if strategy == "none":
            return query

        if strategy not in self.optimizers:
            raise ValueError(
                f"Unknown query optimization strategy: {strategy}. "
                f"Available: {list(self.optimizers.keys())}"
            )

        optimizer = self.optimizers[strategy]

        if strategy == "multi_query":
            return optimizer.generate(query)

        if hasattr(optimizer, "rewrite"):
            return optimizer.rewrite(query)

        return optimizer.reflect(query)
