from typing import Dict, List
from app.strategy_selector import StrategySelector
from app.generator import RAGGenerator


# -------------------------
# Heuristic metrics (LLM-agnostic)
# -------------------------
def heuristic_metrics(query: str, contexts: List[str]) -> Dict[str, float]:
    joined_context = " ".join(contexts).lower()
    query_terms = query.lower().split()

    coverage = sum(
        1 for term in query_terms if term in joined_context
    ) / max(len(query_terms), 1)

    return {
        "retrieval_coverage": coverage,
        "faithfulness": 1.0 if contexts else 0.7,
        "context_precision": min(len(contexts) / 5, 1.0),
    }


# -------------------------
# OFFLINE STEP
# -------------------------
def prepare_indexes(
    documents: List[str],
    strategies: List[Dict],
    chunkers: Dict,
    retrievers: Dict,
) -> Dict[str, object]:
    """
    Build chunked indexes ONCE per strategy.
    This function should be called only during startup.
    """
    indexes = {}

    for strategy in strategies:
        strategy_key = (
            f"{strategy['chunking']}|"
            f"{strategy['retriever']}|"
            f"{strategy.get('query_opt', 'none')}"
        )

        chunker = chunkers[strategy["chunking"]]
        chunks = chunker.chunk(documents)

        retriever = retrievers[strategy["retriever"]]
        retriever.build_index(chunks)

        indexes[strategy_key] = retriever

    return indexes


# -------------------------
# ONLINE STEP
# -------------------------
def adaptive_pipeline(
    query: str,
    strategies: List[Dict],
    indexes: Dict[str, object],
) -> Dict:
    """
    Adaptive RAG pipeline:
    retrieval → evaluation → strategy selection → generation
    """
    results = []

    for strategy in strategies:
        strategy_key = (
            f"{strategy['chunking']}|"
            f"{strategy['retriever']}|"
            f"{strategy.get('query_opt', 'none')}"
        )

        retriever = indexes[strategy_key]

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

    generator = RAGGenerator()
    answer = generator.generate(query, best_contexts)

    return {
        "selected_strategy": best_strategy,
        "contexts": best_contexts,
        "answer": answer,
        "score": decision["score"],
    }
