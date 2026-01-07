# 🚀 Adaptive RAG Evaluation & Optimization Framework

An **end-to-end Retrieval-Augmented Generation (RAG) system** that dynamically selects the **best retrieval strategy** and generates answers using a **local open-source LLM**.

This project focuses on **system design, adaptability, and evaluation**, not just calling an LLM API.

---

## 📌 Key Highlights

* ✅ Multiple chunking strategies (fixed, adaptive, semantic)
* ✅ Multiple retrievers (dense, sparse, hybrid)
* ✅ Query optimization (rewrite, multi-query, reflection)
* ✅ **Adaptive strategy selection** using heuristic evaluation
* ✅ Offline indexing + online inference separation
* ✅ End-to-end answer generation using a **local LLM**
* ✅ Evaluation layer designed (RAGAS + custom heuristics)

---

## 🧠 Why this project?

Most RAG demos:

* use one retriever
* use one chunking method
* hardcode an LLM
* ignore evaluation and trade-offs

This project answers a harder question:

> **“Which RAG strategy works best for a given query?”**

The system **measures, compares, and decides** — automatically.

---

## 🏗️ Architecture Overview

```
Documents (offline)
   ↓
Chunking
   ↓
Indexing (FAISS / BM25)
   ↓
────────────────────────
User Query (online)
   ↓
Multiple Retrieval Strategies
   ↓
Heuristic Evaluation
   ↓
Strategy Selection
   ↓
Best Context
   ↓
Local LLM Generator
   ↓
Final Answer
```

---

## 📂 Project Structure

```
rag-eval-optimizer/
│
├── app/
│   ├── chunking.py              # Document chunking strategies
│   ├── retriever.py             # Dense, sparse & hybrid retrievers
│   ├── query_optimizer.py       # Query rewrite & expansion
│   ├── strategy_selector.py     # Metric-agnostic strategy selection
│   ├── generator.py             # Local LLM generator
│   └── pipeline.py              # Adaptive RAG pipeline
│
├── experiments/                 # Design validation experiments
│
├── test_day4.py                 # Query optimization sanity test
├── test_day5.py                 # Evaluation layer test
├── test_day6_pipeline.py        # Adaptive pipeline test
├── test_day7.py                 # End-to-end RAG test
│
├── config.yaml                  # Central configuration
├── environment.yml              # Conda environment
└── README.md
```

---

## 🔄 Offline vs Online Design (Important)

### Offline (once)

* Load documents
* Chunk documents
* Build retriever indexes

### Online (per query)

* Optimize query
* Retrieve contexts
* Evaluate strategies
* Select best strategy
* Generate answer

This avoids **re-chunking and re-indexing per query**, making the system scalable.

---

## 🤖 LLM Choice (Design Decision)

### Generator LLM

* **Model:** `google/flan-t5-base`
* **Why:**

  * Runs locally on CPU/GPU
  * No API keys required
  * Stable on Windows
  * Ideal for demonstrating RAG architecture

### Why not large models (e.g., Mistral-7B)?

* Require GPU infrastructure
* Increase setup complexity
* Not necessary to demonstrate system design

> Larger models are documented as **production targets**, not local development defaults.

---

## 📊 Evaluation Strategy

### Implemented

* Custom heuristic metrics:

  * retrieval coverage
  * context precision
  * faithfulness signal

### RAGAS

* Integrated as an **optional evaluation layer**
* Known limitation: requires a strong judge LLM (e.g., OpenAI)
* Metrics may return `NaN` in open-source-only setups
* **Does not block core system functionality**

> Evaluation is **decoupled** from generation.

---

## 🧪 Experiments vs Tests

### `experiments/`

* Used to **validate design ideas**
* Not for benchmarking or leaderboard scores

### `test_dayX.py`

* Learning checkpoints
* Sanity tests for each system stage
* Document project progression clearly

Not all tests are meant to be run end-to-end without infra setup — this is intentional.

---

## ▶️ How to Run (Core Demo)

```bash
conda activate rag-eval
python test_day7.py
```

Expected output:

* Selected strategy
* Generated answer from local LLM

---

## 🎯 What this project demonstrates

* System-level thinking
* Trade-off awareness
* Modular ML design
* Production-oriented RAG architecture

---

## 🧠 Interview-ready summary

> “I built an adaptive RAG system that evaluates multiple retrieval strategies, selects the best one dynamically, and generates answers using a local open-source LLM. The system separates offline indexing from online inference and includes an optional evaluation layer.”

---

## 🔮 Future Work (Optional)

* Caching & latency optimization
* API fallback for LLM generation
* Streamlit demo UI
* Production vector DB (Qdrant)
* Monitoring dashboard

---

## ✅ Project Status

* **Core system:** Complete (Day 7)
* **Enhancements:** Optional
* **UI:** Intentionally deferred

---

### 👏 Final Note

This project prioritizes **correct architecture over flashy demos**.
It is designed to **age well**, scale conceptually, and be easy to explain.

