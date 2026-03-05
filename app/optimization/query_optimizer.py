from typing import List
from transformers import pipeline


# =========================
# 1️⃣ Query Rewriter
# =========================
class QueryRewriter:
    """Rewrites a user query into a clearer, retrieval-friendly version."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.rewriter = pipeline("text2text-generation", model=model_name)

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the following question to be more specific and clear for document retrieval:\n"
            f"{query}"
        )
        output = self.rewriter(prompt, max_length=64)
        return output[0]["generated_text"]


# =========================
# 2️⃣ Multi Query Generator
# =========================
class MultiQueryGenerator:
    """Generates multiple query variations to improve recall."""

    def __init__(self, model_name: str = "google/flan-t5-base", n_queries: int = 3):
        self.generator = pipeline("text2text-generation", model=model_name)
        self.n_queries = n_queries

    def generate(self, query: str) -> List[str]:
        prompt = (
            f"Generate {self.n_queries} different search queries "
            f"that have the same meaning:\n{query}"
        )
        outputs = self.generator(prompt, max_length=128)
        queries = outputs[0]["generated_text"].split("\n")
        return [q.strip() for q in queries if q.strip()]


# =========================
# 3️⃣ Self Reflection Agent
# =========================
class SelfReflectionQueryAgent:
    """Reflects on query quality and improves it only if needed."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.agent = pipeline("text2text-generation", model=model_name)

    def reflect(self, query: str) -> str:
        prompt = (
            "Analyze the following query. "
            "If it is vague or incomplete, improve it. "
            "If it is already good, return it unchanged.\n\n"
            f"Query: {query}"
        )
        output = self.agent(prompt, max_length=64)
        return output[0]["generated_text"]
