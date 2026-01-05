# app/evaluator.py

from typing import List, Dict
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from langchain_huggingface import HuggingFaceEndpoint


class RAGEvaluator:
    """
    Evaluates RAG outputs using RAGAS metrics
    with Hugging Face Inference API (stable client).
    """

    def __init__(self):
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0.0,
            max_new_tokens=256,
        )

        self.metrics = [
            AnswerRelevancy(llm=llm),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            Faithfulness(llm=llm),
        ]

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str | None = None,
    ) -> Dict[str, float]:

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=self.metrics)
        return result.to_pandas().iloc[0].to_dict()
