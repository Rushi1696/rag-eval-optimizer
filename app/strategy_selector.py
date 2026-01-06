# app/strategy_selector.py

from typing import Dict, List
import math


class StrategySelector:
    """
    Metric-agnostic strategy selector for RAG systems.
    """

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {
            "answer_relevancy": 0.4,
            "faithfulness": 0.4,
            "context_precision": 0.1,
            "context_recall": 0.1,
            "retrieval_coverage": 0.2,
        }

    def safe_score(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        total_weight = 0.0

        for key, weight in self.weights.items():
            value = metrics.get(key)

            if value is None:
                continue

            if isinstance(value, float) and math.isnan(value):
                continue

            score += weight * value
            total_weight += weight

        return score / total_weight if total_weight else 0.0

    def select_best(self, results: List[Dict]) -> Dict:
        scored = []

        for r in results:
            s = self.safe_score(r["metrics"])
            scored.append((s, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        return {
            "best_strategy": scored[0][1]["strategy"],
            "score": scored[0][0],
            "all_results": scored,
        }
