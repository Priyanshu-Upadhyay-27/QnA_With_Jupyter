import json
from pathlib import Path
from langchain_core.documents import Document

def save(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def exists(path: str) -> bool:
    return Path(path).exists()

def save_documents(documents, path: str):
    serializable = []

    for doc in documents:
        serializable.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(documents)} docs to {path}")



def load_documents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"]
        )
        for d in raw
    ]

    print(f"📂 Loaded {len(documents)} docs from {path}")
    return documents
