from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss


class SPLADERetriever:
    """
    SPLADE Retrieval: Sparse retrieval using learned sparse representations.
    Documents and queries are encoded into sparse vectors for efficient search.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", top_k: int = 5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.top_k = top_k
        self.index = None
        self.doc_texts = []

    def _encode_sparse(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into sparse representations"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # SPLADE uses log-sigmoid activation on token logits
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        sparse_reps = torch.log(1 + torch.relu(logits))  # SPLADE activation

        # Sum over sequence dimension to get document-level sparse vectors
        sparse_vectors = sparse_reps.sum(dim=1)  # Shape: (batch_size, vocab_size)

        return sparse_vectors

    def build_index(self, documents: List[str]):
        """Build sparse index from documents"""
        print(f"[INFO] Building SPLADE index for {len(documents)} documents...")

        # Encode all documents
        sparse_vectors = self._encode_sparse(documents)

        # Convert to dense for Faiss (SPLADE vectors are high-dimensional sparse)
        dense_vectors = sparse_vectors.numpy().astype('float32')

        # Build Faiss index
        dim = dense_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product for similarity
        faiss.normalize_L2(dense_vectors)  # Normalize for cosine similarity
        self.index.add(dense_vectors)

        self.doc_texts = documents
        print(f"[INFO] SPLADE index built with {len(documents)} documents")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents using sparse query encoding"""
        if self.index is None:
            raise RuntimeError("SPLADE index not built. Call build_index() first.")

        # Encode query
        query_vector = self._encode_sparse([query]).numpy().astype('float32')
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, self.top_k)

        # Return documents
        results = [self.doc_texts[i] for i in indices[0] if i < len(self.doc_texts)]
        return results
