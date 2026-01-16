import json
from pathlib import Path

def save(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def exists(path: str) -> bool:
    return Path(path).exists()

def save_rag_documents(docs, path="artifacts/rag_documents.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)



def save_split_documents(split_docs, path="artifacts/split_documents.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in split_docs
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
