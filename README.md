# Adaptive RAG Evaluation & Benchmarking Framework

## Problem
Most Retrieval-Augmented Generation (RAG) systems use static retrieval
and chunking strategies and do not measure answer quality or hallucination.

## Solution
This project implements a unified RAG system that:
- Supports multiple chunking and retrieval strategies
- Optimizes queries using agent-based techniques
- Evaluates answers using RAGAS metrics
- Dynamically adapts strategies based on evaluation scores
- Provides an offline benchmarking framework to compare strategies

## Key Features
- Fixed & semantic chunking
- Dense, BM25, and hybrid retrieval
- Query rewriting and multi-query expansion
- Faithfulness and relevance evaluation
- Online adaptive inference & offline benchmarking

## Tech Stack
- LangChain / LangGraph
- FAISS / BM25
- HuggingFace embeddings
- Open-source LLMs (Mistral / LLaMA)
- FastAPI

## Day 3 – Retrieval System (Dense, Sparse & Hybrid)

This module implements a **production-grade retrieval layer** for a RAG system.

### Implemented Retrievers

1. **Dense Retriever**
   - Uses Sentence Transformers for embeddings
   - FAISS (L2) vector index
   - Best for semantic similarity

2. **Sparse Retriever (BM25)**
   - Keyword-based lexical retrieval
   - Strong for exact term matching

3. **Hybrid Retriever**
   - Combines dense + sparse results
   - Score fusion controlled by `alpha`
   - Improves recall and robustness

### Retriever Router
A router selects the retrieval strategy dynamically:
- `dense`
- `bm25`
- `hybrid`

This allows experimentation and benchmarking without code changes.

### Why Not LangChain Built-ins?
- Full control over scoring & fusion
- Easier debugging
- Framework-agnostic design
- Research-friendly architecture

This layer is evaluation-ready and integrates cleanly with downstream RAG pipelines.
## Day 5 – RAG Evaluation (RAGAS)

This project integrates an evaluation layer using **RAGAS** to measure:

- Answer relevancy
- Context precision
- Context recall
- Faithfulness (hallucination detection)

### Note on Hugging Face Evaluation
RAGAS metrics rely on asynchronous LLM-based judgment.  
While the evaluation pipeline executes end-to-end using Hugging Face Inference API,  
current async incompatibilities may result in `NaN` metric values.

This is a known limitation when using Hugging Face endpoints for RAGAS evaluation.
The evaluation layer is structurally complete and can be switched to OpenAI/Azure
for numeric scoring when required.

**Key takeaway:**  
The system supports evaluation, detects hallucination pathways, and exposes
clear failure modes — which is critical for production RAG systems.
