"""
Example: Evaluating different retrieval strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGPipeline, load_documents
from app.optimization import StrategySelector

# Load documents
documents = load_documents("datasets")

if not documents:
    print("[WARNING] No documents found in datasets/ folder")
    documents = [{
        "content": "Transformers use attention mechanisms to process sequences. There are different types: dense attention, sparse attention, and multi-head attention.",
        "metadata": {"source": "demo", "file_type": "txt"}
    }]

# Test different retrieval strategies
strategies = ["dense", "bm25", "hybrid"]
query = "What is attention in transformers?"

results = []

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy.upper()}")
    print(f"{'='*60}")

    # Create pipeline with specific strategy
    pipeline = RAGPipeline(
        chunk_strategy="semantic",
        retrieval_strategy=strategy,
    )

    # Build index
    pipeline.build_index(documents)

    # Query
    result = pipeline.query(query)

    results.append({
        "strategy": {"retriever": strategy},
        "metrics": result["metrics"],
    })

    print(f"Metrics: {result['metrics']}")

# Select best strategy
selector = StrategySelector()
best = selector.select_best(results)

print(f"\n{'='*60}")
print("STRATEGY COMPARISON")
print(f"{'='*60}")
print(f"Best Strategy: {best['best_strategy']}")
print(f"Score: {best['score']:.4f}")
print(f"\nAll Results:")
for score, res in best['all_results']:
    print(f"  {res['strategy']['retriever']}: {score:.4f}")
