from transformers import pipeline
from typing import List


class RAGGenerator:
    """
    LLM-based generator for RAG.
    Uses a small, CPU-friendly instruction model.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
    ):
        self.generator = pipeline(
            task="text2text-generation",
            model=model_name,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, query: str, contexts: List[str]) -> str:
        context_block = "\n".join(contexts)

        prompt = (
            "Answer the question using the given context.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        output = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        return output[0]["generated_text"].strip()
