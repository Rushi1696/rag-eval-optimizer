import os
import re
from typing import List, Dict
import PyPDF2
import docx
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

# Ensure punkt tokenizer exists
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text: str) -> str:
    """Basic text preprocessing for RAG"""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------
# Document Loader
# ---------------------------
def load_documents(
    folder_path: str,
    file_types: List[str] = None,
    recursive: bool = True
) -> List[Dict]:
    """Load documents of multiple formats for RAG"""
    if file_types is None:
        file_types = ["txt", "pdf", "docx", "csv", "xlsx"]

    documents = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext not in file_types:
                continue

            file_path = os.path.join(root, file)
            try:
                content = ""

                if ext == "txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                elif ext == "pdf":
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        content = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )

                elif ext == "docx":
                    doc = docx.Document(file_path)
                    content = "\n".join(p.text for p in doc.paragraphs)

                elif ext == "csv":
                    df = pd.read_csv(file_path)
                    content = df.to_string()

                elif ext == "xlsx":
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)

                content = clean_text(content)

                if content:
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": file_path,
                            "file_type": ext
                        }
                    })

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        if not recursive:
            break

    return documents


# ---------------------------
# Fixed Size Chunker
# ---------------------------
class FixedSizeChunker:
    """Character-based baseline chunking"""
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            start = 0
            while start < len(doc):
                end = start + self.chunk_size
                chunks.append(doc[start:end])
                start += self.chunk_size - self.overlap
        return chunks


# ---------------------------
# Token-based Chunker
# ---------------------------
class AdaptiveChunker:
    """Token-aware chunking aligned with LLM context limits"""
    def __init__(self, chunk_size=512, overlap=50, encoding_name="cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            tokens = self.encoder.encode(doc)
            start = 0
            while start < len(tokens):
                end = start + self.chunk_size
                chunk_text = self.encoder.decode(tokens[start:end])
                chunks.append(chunk_text)
                start += self.chunk_size - self.overlap
        return chunks


# ---------------------------
# Semantic + Token Chunker
# ---------------------------
class SemanticAdaptiveChunker:
    """Sentence-aware chunking with token constraints"""
    def __init__(self, max_tokens=512, encoding_name="cl100k_base"):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            sentences = sent_tokenize(doc)
            current_chunk = []
            current_tokens = 0

            for sent in sentences:
                sent_tokens = len(self.encoder.encode(sent))
                if current_tokens + sent_tokens > self.max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens

            if current_chunk:
                chunks.append(" ".join(current_chunk))

        return chunks
