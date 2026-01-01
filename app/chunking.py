import os
import re
from typing import Dict, List, Tuple

import PyPDF2
import docx
import pandas as pd


def clean_text(text:str) -> str:
    """basic text processing for RAG"""
    text= text.replace("\x00", " ") # remove null bytes
    text= re.sub("\s+", " ", text)
    return text.strip()

def load_documents(
        folder_path: str,
        file_types: List[str]= None,
        recursive: bool= True
)-> List[Dict]:

    """
    Load documents of multiple formats for RAG.

    Returns:
        List of dicts:
        {
            "content": str,
            "metadata": {
                "source": file_path,
                "file_type": extension
            }
        }
    """
    if file_types is None:
        file_types=["txt", "pdf", "docx", "csv", "xlsx"]

    dovuments= []

    for root, _, files in os.walk(folder_path):
        for file in files:
            ext= file.split(".")[-1].lower()
            if ext not in file_types:
                continue
            file_path = os.path.join(root, file)

            try:
                content= ""

                if ext =="txt":
                    with open(file_path, "r", encoding = "utf-8") as f:
                        content = f.read()

                elif ext == "pdf" and PyPDF2:
                    with open(file_path, "rb") as F:
                        reader= PyPDF2.PdfReader(f)
                        content = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )

                elif ext =="docx" and docx:
                    doc= docx.Document(file_path)
                    content = "\n".join(
                        para.text for para in doc.paragraphs
                    )

                elif ext =="csv" and pd:
                    df = pd.read_csv(file_path)
                    content = df.to_string()

                elif ext == "xlsx" and pd:
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)

                else:
                    continue
            
                content= clean_text(content)

                if content:
                    documents.append(
                        {
                            "content": content,
                            "metadata": {
                                "source": file_path,
                                "file_type": ext
                            }
                        }
                    )
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

            if not recursive:
                break
    return documents

