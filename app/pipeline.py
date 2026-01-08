# app/pipeline.py

from typing import List, Dict
from app.strategy_selector import StrategySelector
from app.config import load_config


def heuristic_metrics(query: str, contexts: List[str]) -> Dict[str, float]:
    joined = " ".join(contexts).lower()
    query_terms = query.lower().split()

    coverage = sum(1 for t in query_terms if t in joined) / max(len(query_terms), 1)

    return {
        "retrieval_coverage": coverage,
        "faithfulness": 1.0 if contexts else 0.7,
        "context_precision": min(len(contexts) / 5, 1.0),
    }


def adaptive_pipeline(
    query: str,
    documents: List[Dict],
    chunkers: Dict,
    retrievers: Dict,
):
    config = load_config()

    strategy = {
        "chunking": config["chunking"]["strategy"],
        "retriever": config["retrieval"]["strategy"],
        "query_opt": "none",
    }

    print(f"[INFO] Evaluating strategy: {strategy}")

    # 1️⃣ Normalize documents → TEXT ONLY
    texts = [d["content"] for d in documents]

    # 2️⃣ Chunking
    chunker = chunkers[strategy["chunking"]]
    chunks = chunker.chunk(texts)
    print(f"[INFO] Chunks created: {len(chunks)}")

    # 3️⃣ Build indexes EXPLICITLY
    retriever = retrievers[strategy["retriever"]]

    if hasattr(retriever, "dense"):
        retriever.dense.build_index(chunks)
    elif hasattr(retriever, "build_index"):
        retriever.build_index(chunks)

    # 4️⃣ Retrieval
    contexts = retriever.retrieve(query)
    print(f"[INFO] Contexts retrieved: {len(contexts)}")

    # 5️⃣ Evaluation
    metrics = heuristic_metrics(query, contexts)

    selector = StrategySelector()
    decision = selector.select_best([
        {"strategy": strategy, "metrics": metrics}
    ])

    return {
        "selected_strategy": decision["best_strategy"],
        "contexts": contexts,
        "score": decision["score"],
    }
