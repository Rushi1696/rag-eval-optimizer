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

## 🔮 Future Work (Optional)

* Caching & latency optimization
* API fallback for LLM generation
* Streamlit demo UI
* Production vector DB (Qdrant)
* Monitoring dashboard


⚠️ Limitations & Areas for Improvement

While the project demonstrates a strong adaptive RAG architecture, there are several areas where it can be further improved for broader adoption and production readiness.

1. No Production-Ready UI or Dashboard

The project currently focuses on backend system design and experimentation.
There is no polished UI or interactive dashboard.

A future Streamlit-based demo could improve usability.

A UI would help visualize strategy selection, retrieved contexts, and generated answers.

2. Evaluation Is Heuristic, Not Benchmark-Driven

The current evaluation relies on heuristic metrics such as:

retrieval coverage

context precision

faithfulness signals

While useful for strategy comparison, these are not standardized benchmarks.

Notable improvements include:

Integrating RAGAS, ARES, or similar frameworks

Adding reference-based and judge-based evaluation

Using community-accepted RAG evaluation benchmarks

Evaluation is intentionally designed as optional and decoupled, but can be expanded.

3. Limited Default Model Choice

The system defaults to a small local LLM for reliability and ease of setup.

This is suitable for:

architectural validation

local development

demos

However, it is not ideal for high-quality generation at scale.

Future improvements may include support for:

Larger open-source models (LLaMA, Mistral, Mixtral)

Cloud LLM APIs (OpenAI, Anthropic, Vertex AI)

Hybrid local + API-based generation

4. Documentation Can Be Expanded

The README currently provides a high-level overview but lacks:

detailed API usage examples

sample outputs

performance benchmarks

end-to-end usage walkthroughs

Adding these would make the repository easier for external users to adopt.

🛠️ Future Work & Extensions
1. Standardized Evaluation Integration

Integrate RAGAS, ARES, or Open-RAG-Eval

Support both automatic and judge-based evaluation

Enable offline benchmarking pipelines

2. Multiple LLM Backends

Modular support for:

OpenAI / Anthropic APIs

Hugging Face Inference

Local inference via llama-cpp or text-generation-inference

Backend selection via configuration

3. Visualization & Reporting

JSON-based evaluation reports

Strategy comparison plots

HTML dashboards for analysis

Retrieval vs generation quality diagnostics

4. Packaging & Deployment

pip install support

Dockerized deployment

Example configuration presets

Cloud and on-prem deployment guides

🧠 Overall Assessment

This project is a strong foundation for adaptive RAG systems, with particular strengths in:

system architecture

strategy optimization

offline vs online separation

extensibility-first design

While additional work is needed for large-scale research or production deployment, it already serves as a solid research and prototyping framework.