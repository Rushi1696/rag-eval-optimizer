# RAG Scripts

Quick start examples and utilities for the RAG Evaluation Optimizer.

## Usage

### 1. **example_rag_pipeline.py** - Modern RAGPipeline Class

The recommended way to use the system. Uses the new `RAGPipeline` class.

```bash
python scripts/example_rag_pipeline.py
```

**What it does:**
- Loads documents from `datasets/`
- Initializes RAGPipeline with semantic chunking
- Builds hybrid retrieval index
- Runs a sample query
- Displays answer, contexts, and metrics

**Best for:** Learning the modern API

---

### 2. **build_index.py** - Preprocess & Index

Build retrieval indexes before querying. Useful for production setups.

```bash
python scripts/build_index.py
```

**What it does:**
- Loads and processes documents
- Creates chunked indexes
- Initializes retrievers (dense, BM25, hybrid)
- Saves metadata

**Best for:** Preprocessing documents once, then querying multiple times

---

### 3. **evaluate_strategies.py** - Compare Retrieval Methods

Test and compare different retrieval strategies (dense, BM25, hybrid).

```bash
python scripts/evaluate_strategies.py
```

**What it does:**
- Tests each retrieval strategy separately
- Computes metrics for each
- Selects the best performing strategy
- Displays comparison results

**Best for:** Performance benchmarking

---

### 4. **test_legacy_day8.py** - Legacy Code

Old adaptive pipeline test updated to use new structure. For backward compatibility.

```bash
python scripts/test_legacy_day8.py
```

**Best for:** Testing legacy code paths

---

## New Project Structure

```
app/
├── __init__.py
├── config/                    # Configuration management
│   ├── config.py
│   └── __init__.py
├── rag/                       # Core RAG pipeline
│   ├── __init__.py
│   ├── pipeline.py           # New RAGPipeline class
│   ├── generator.py          # LLM generation
│   └── chunking.py           # Document processing
├── retrieval/                # Retrieval strategies
│   ├── __init__.py
│   ├── dense.py              # Dense vector retrieval
│   ├── bm25.py               # BM25 keyword retrieval
│   └── hybrid.py             # Hybrid retrieval
├── evaluation/               # RAG evaluation
│   ├── __init__.py
│   └── evaluator.py          # RAGAS evaluation
└── optimization/             # Query/strategy optimization
    ├── __init__.py
    ├── query_optimizer.py    # Query rewriting
    ├── router.py             # Strategy routing
    └── strategy_selector.py  # Strategy selection

scripts/                      # Example scripts
├── example_rag_pipeline.py
├── build_index.py
├── evaluate_strategies.py
└── test_legacy_day8.py
```

---

## Folder Setup

Make sure `datasets/` folder exists with sample documents:

```bash
mkdir -p datasets
# Add .txt, .pdf, .docx, .csv, or .xlsx files here
```

---

## Quick Start

1. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**
   ```bash
   # Add documents to datasets/ folder
   ```

3. **Run example:**
   ```bash
   python scripts/example_rag_pipeline.py
   ```

4. **Evaluate strategies:**
   ```bash
   python scripts/evaluate_strategies.py
   ```

---

## API Overview

### Using RAGPipeline class (Recommended)

```python
from app.rag import RAGPipeline, load_documents

# Load documents
documents = load_documents("datasets")

# Initialize pipeline
pipeline = RAGPipeline(
    chunk_strategy="semantic",
    retrieval_strategy="hybrid"
)

# Build index
pipeline.build_index(documents)

# Query
result = pipeline.query("Your question here")
print(result["answer"])
print(result["contexts"])
print(result["metrics"])

# Switch strategy
pipeline.set_retrieval_strategy("dense")

# Evaluate using RAGAS (if available)
eval_metrics = pipeline.evaluate_answer(
    question="...",
    answer="...",
    contexts=[...],
    ground_truth="..."  # optional
)
```

---

## Next Steps

- For API server → See [Day 4 - FastAPI](../docs/day4_api.md)
- For Docker deployment → See [Day 5 - Docker](../docs/day5_docker.md)
- For detailed docs → See [README.md](../README.md)
