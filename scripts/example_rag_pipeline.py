"""
Example: Using the new RAGPipeline class
This is the modern way to use the system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGPipeline, load_documents

# Load documents from datasets folder
documents = load_documents("datasets")

if not documents:
    print("[WARNING] No documents found in datasets/ folder")
    print("[INFO] Creating a test document for demo...")
    documents = [{
        "content": "Attention is a mechanism in transformers that allows the model to weigh the importance of different words when processing sequences. Self-attention specifically allows each token to attend to all other tokens in the sequence.",
        "metadata": {"source": "demo", "file_type": "txt"}
    }]

# Initialize pipeline
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    chunk_strategy="semantic",
    retrieval_strategy="hybrid",
)

# Build index
print("\n" + "="*60)
print("Building RAG Index")
print("="*60)
pipeline.build_index(documents)

# Run query
query = "What is self attention in transformers?"
print("\n" + "="*60)
print(f"Query: {query}")
print("="*60)

result = pipeline.query(query)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Answer: {result['answer']}")
print(f"\nContexts Retrieved: {len(result['contexts'])}")
for i, ctx in enumerate(result['contexts'][:2], 1):
    print(f"\n  [{i}] {ctx[:100]}...")
print(f"\nMetrics: {result['metrics']}")
