def run_pipeline(
    query: str,
    retriever,
    query_optimizer,
    query_strategy: str = "none"
):
    """
    Full RAG pipeline (without generation).

    Steps:
    1. Optimize query
    2. Retrieve documents
    """

    optimized_query = query_optimizer.optimize(query, query_strategy)

    if isinstance(optimized_query, list):
        results = []
        for q in optimized_query:
            results.extend(retriever.retrieve(q))
        return results

    return retriever.retrieve(optimized_query)
