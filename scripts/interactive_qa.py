"""
Interactive RAG Answer Generator
Ask questions and get answers from your RAG system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGPipeline, load_documents

def interactive_qa():
    """Interactive question-answering with the RAG system"""

    print("\n" + "="*70)
    print("🤖 INTERACTIVE RAG ANSWER GENERATOR")
    print("="*70)
    print("Your RAG system is ready to answer questions!")
    print("Type 'quit' or 'exit' to stop.")
    print("="*70)

    # Load documents
    documents = load_documents("datasets")

    if not documents:
        print("[WARNING] No documents found in datasets/ folder")
        print("[INFO] Using demo document for testing...")
        documents = [{
            "content": "Self-attention is a mechanism in transformer neural networks that allows the model to weigh the importance of different words in a sequence when processing each word. It computes attention scores between all pairs of words, enabling the model to focus on relevant context. Multi-head attention extends this by using multiple attention mechanisms in parallel.",
            "metadata": {"source": "demo", "file_type": "txt"}
        }]

    # Initialize pipeline
    print(f"\n[INFO] Initializing RAG Pipeline with {len(documents)} documents...")
    pipeline = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generator_model="google/flan-t5-base",
        chunk_strategy="semantic",
        retrieval_strategy="hybrid",
    )

    # Build index
    pipeline.build_index(documents)
    print("[INFO] Index built successfully!")

    while True:
        # Get user question
        question = input("\n❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print(f"\n🔍 Processing: '{question}'")
        print("-" * 50)

        try:
            # Get answer
            result = pipeline.query(question)

            print("📝 Answer:")
            print(f"   {result['answer']}")
            print(f"\n📊 Metrics:")
            print(f"   Retrieval Coverage: {result['metrics']['retrieval_coverage']:.2f}")
            print(f"   Faithfulness: {result['metrics']['faithfulness']:.2f}")
            print(f"   Context Precision: {result['metrics']['context_precision']:.2f}")

            print(f"\n📄 Contexts Retrieved: {len(result['contexts'])}")
            for i, ctx in enumerate(result['contexts'][:2], 1):
                print(f"\n   [{i}] {ctx[:150]}...")

        except Exception as e:
            print(f"❌ Error generating answer: {e}")

if __name__ == "__main__":
    interactive_qa()
