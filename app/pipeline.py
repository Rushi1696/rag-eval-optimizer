# app/pipeline.py

from typing import Dict, List
from app.strategy_selector import StrategySelector


def heuristic_metrics(query: str, contexts: List[str]) -> Dict[str, float]:
    joined = " ".join(contexts).lower()
    query_terms = query.lower().split()

    coverage = sum(1 for t in query_terms if t in joined) / max(len(query_terms), 1)

    return {
        "retrieval_coverage": coverage,
        "faithfulness": 1.0 if contexts else 0.7,
        "context_precision": min(len(contexts) / 5, 1.0),
    }


def prepare_indexes(
    documents: List[str],
    strategies: List[Dict],
    chunkers: Dict,
    retrievers: Dict,
):
    """
    OFFLINE STEP:
    Build chunked indexes once per strategy.
    """
    indexes = {}

    for strategy in strategies:
        name = str(strategy)

        chunker = chunkers[strategy["chunking"]]
        chunks = chunker.chunk(documents)

        retriever = retrievers[strategy["retriever"]]
        retriever.build_index(chunks)

        indexes[name] = retriever

    return indexes


def adaptive_pipeline(
    query: str,
    strategies: List[Dict],
    indexes: Dict,
):
    """
    ONLINE STEP:
    Retrieve + evaluate + select best strategy.
    """
    results = []

    for strategy in strategies:
        name = str(strategy)
        retriever = indexes[name]

        contexts = retriever.retrieve(query)
        metrics = heuristic_metrics(query, contexts)

        results.append({
            "strategy": strategy,
            "metrics": metrics,
            "contexts": contexts,
        })

    selector = StrategySelector()
    decision = selector.select_best(results)

    best_strategy = decision["best_strategy"]

    best_contexts = next(
        r["contexts"] for r in results
        if r["strategy"] == best_strategy
    )

    return {
        "selected_strategy": best_strategy,
        "contexts": best_contexts,
        "score": decision["score"],
    }
