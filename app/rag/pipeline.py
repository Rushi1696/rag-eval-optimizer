from typing import List, Dict, Optional
from app.rag.chunking import FixedSizeChunker, SemanticAdaptiveChunker, AdaptiveChunker
from app.rag.generator import RAGGenerator
from app.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from app.optimization.strategy_selector import StrategySelector
from app.config.config import load_config


def heuristic_metrics(query: str, contexts: List[str]) -> Dict[str, float]:
    """Compute heuristic metrics for retrieval quality"""
    joined = " ".join(contexts).lower()
    query_terms = query.lower().split()

    coverage = sum(1 for t in query_terms if t in joined) / max(len(query_terms), 1)

    return {
        "retrieval_coverage": coverage,
        "faithfulness": 1.0 if contexts else 0.7,
        "context_precision": min(len(contexts) / 5, 1.0),
    }


def adaptive_pipeline(
    query: str,
    documents: List[Dict],
    chunkers: Dict,
    retrievers: Dict,
):
    """Legacy adaptive pipeline function"""
    config = load_config()

    strategy = {
        "chunking": config["chunking"]["strategy"],
        "retriever": config["retrieval"]["strategy"],
        "query_opt": "none",
    }

    print(f"[INFO] Evaluating strategy: {strategy}")

    # 1️⃣ Normalize documents → TEXT ONLY
    texts = [d["content"] for d in documents]

    # 2️⃣ Chunking
    chunker = chunkers[strategy["chunking"]]
    chunks = chunker.chunk(texts)
    print(f"[INFO] Chunks created: {len(chunks)}")

    # 3️⃣ Build indexes EXPLICITLY
    retriever = retrievers[strategy["retriever"]]

    if hasattr(retriever, "dense"):
        retriever.dense.build_index(chunks)
    elif hasattr(retriever, "build_index"):
        retriever.build_index(chunks)

    # 4️⃣ Retrieval
    contexts = retriever.retrieve(query)
    print(f"[INFO] Contexts retrieved: {len(contexts)}")

    # 5️⃣ Evaluation
    metrics = heuristic_metrics(query, contexts)

    selector = StrategySelector()
    decision = selector.select_best([
        {"strategy": strategy, "metrics": metrics}
    ])

    return {
        "selected_strategy": decision["best_strategy"],
        "contexts": contexts,
        "score": decision["score"],
    }


class RAGPipeline:
    """
    Main RAG Pipeline class.
    Orchestrates chunking, retrieval, generation, and evaluation.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_strategy: str = "semantic",
        retrieval_strategy: str = "hybrid",
    ):
        # Initialize chunkers
        self.chunkers = {
            "fixed": FixedSizeChunker(),
            "adaptive": AdaptiveChunker(),
            "semantic": SemanticAdaptiveChunker(),
        }
        self.chunk_strategy = chunk_strategy

        # Initialize retrievers
        self.dense = DenseRetriever(embedding_model, top_k=5)
        self.bm25 = None  # Will be initialized after documents are loaded
        self.hybrid = None

        self.retrieval_strategy = retrieval_strategy
        self.current_retriever = None

        # Initialize generator
        self.generator = RAGGenerator(generator_model)

        # Initialize strategy selector
        self.selector = StrategySelector()

        # Storage for indexed documents
        self.indexed_chunks: List[str] = []
        self.indexed_documents: List[Dict] = []

    def build_index(self, documents: List[Dict]):
        """Build retrieval indexes from documents"""
        print(f"[INFO] Building index from {len(documents)} documents...")

        # Store documents
        self.indexed_documents = documents
        texts = [d["content"] for d in documents]

        # Chunk documents
        chunker = self.chunkers[self.chunk_strategy]
        self.indexed_chunks = chunker.chunk(texts)
        print(f"[INFO] Created {len(self.indexed_chunks)} chunks")

        # Build dense index
        self.dense.build_index(self.indexed_chunks)

        # Build BM25 retriever
        self.bm25 = BM25Retriever(self.indexed_chunks, top_k=5)

        # Build hybrid retriever
        self.hybrid = HybridRetriever(self.dense, self.bm25, alpha=0.5)

        # Select retriever
        retriever_map = {
            "dense": self.dense,
            "bm25": self.bm25,
            "hybrid": self.hybrid,
        }
        self.current_retriever = retriever_map[self.retrieval_strategy]
        print(f"[INFO] Using {self.retrieval_strategy} retriever")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve contexts for a query"""
        if self.current_retriever is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        contexts = self.current_retriever.retrieve(query)
        print(f"[INFO] Retrieved {len(contexts)} contexts")
        return contexts

    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate answer from query and contexts"""
        answer = self.generator.generate(query, contexts)
        return answer

    def query(self, question: str) -> Dict:
        """
        End-to-end query: retrieve → generate
        Returns: {answer, contexts, metrics}
        """
        if self.current_retriever is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Retrieve
        contexts = self.retrieve(question)

        # Generate
        answer = self.generate(question, contexts)

        # Compute metrics
        metrics = heuristic_metrics(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "metrics": metrics,
        }

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate RAG output using RAGAS metrics.
        (Requires ragas package and HuggingFace API key)
        """
        try:
            from app.evaluation.evaluator import RAGEvaluator

            evaluator = RAGEvaluator()
            metrics = evaluator.evaluate(question, answer, contexts, ground_truth)
            return metrics
        except ImportError:
            print("[WARNING] RAGAS not available. Using heuristic metrics only.")
            return heuristic_metrics(question, contexts)

    def set_retrieval_strategy(self, strategy: str):
        """Switch retrieval strategy (dense, bm25, hybrid)"""
        if strategy not in ["dense", "bm25", "hybrid"]:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.retrieval_strategy = strategy

        retriever_map = {
            "dense": self.dense,
            "bm25": self.bm25,
            "hybrid": self.hybrid,
        }
        self.current_retriever = retriever_map[strategy]
        print(f"[INFO] Switched to {strategy} retriever")

    def set_chunk_strategy(self, strategy: str):
        """Switch chunking strategy"""
        if strategy not in self.chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        self.chunk_strategy = strategy
        print(f"[INFO] Chunking strategy set to {strategy}")
        print("[INFO] Index needs to be rebuilt for changes to take effect")
