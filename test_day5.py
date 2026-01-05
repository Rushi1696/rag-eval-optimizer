from app.evaluator import RAGEvaluator
from app.generator import SimpleGenerator

question = "What is self-attention in transformers?"
contexts = [
    "Self-attention allows a transformer model to weigh relationships between tokens in a sequence."
]
ground_truth = "Self-attention lets transformers understand token relationships by computing attention scores."

generator = SimpleGenerator()
answer = generator.generate(question, contexts)

evaluator = RAGEvaluator()
scores = evaluator.evaluate(
    question=question,
    answer=answer,
    contexts=contexts,
    ground_truth=ground_truth
)

print("\nEvaluation Scores:")
for k, v in scores.items():
    print(f"{k}: {v}")
