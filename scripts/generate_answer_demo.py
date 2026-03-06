"""
Generate Answer Demo
Shows how your RAG system generates answers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGPipeline, load_documents

def generate_answer_demo():
    """Demo of answer generation"""

    print("\n" + "="*70)
    print("🤖 RAG ANSWER GENERATION DEMO")
    print("="*70)

    # Load documents
    documents = load_documents("datasets")

    if not documents:
        print("[INFO] Using demo document...")
        documents = [{
            "content": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with generative AI. It first retrieves relevant documents from a knowledge base, then uses those documents as context to generate more accurate and informative answers. RAG helps reduce hallucinations in large language models by grounding responses in factual information.",
            "metadata": {"source": "demo", "file_type": "txt"}
        }]

    # Initialize pipeline
    pipeline = RAGPipeline(
        chunk_strategy="semantic",
        retrieval_strategy="hybrid",
    )

    # Build index
    print(f"[INFO] Building index from {len(documents)} documents...")
    pipeline.build_index(documents)

    # Test questions
    questions = [
        "What is RAG?",
        "How does RAG work?",
        "What are the benefits of RAG?"
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)

        # Generate answer
        result = pipeline.query(question)

        print("📝 Answer:")
        print(f"   {result['answer']}")

        print("\n📊 Metrics:")
        print(f"   Coverage: {result['metrics']['retrieval_coverage']:.2f}")
        print(f"   Faithfulness: {result['metrics']['faithfulness']:.2f}")
        print(f"   Contexts: {len(result['contexts'])}")

        print("\n" + "="*70)

if __name__ == "__main__":
    generate_answer_demo()
