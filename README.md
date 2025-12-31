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
