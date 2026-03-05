from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, embedding_model: str, top_k: int = 5):
        self.model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.index = None
        self.texts = []

    def build_index(self, documents: List[str]):
        embeddings = self.model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.texts = documents

    def retrieve(self, query: str) -> List[str]:
        if self.index is None:
            raise RuntimeError("Dense index not built")

        query_emb = self.model.encode([query]).astype("float32")
        _, indices = self.index.search(query_emb, self.top_k)
        return [self.texts[i] for i in indices[0]]
