import torch
from transformers import AutoModel, AutoTokenizer
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from typing import List

# 1. The Custom CodeT5+ Wrapper (Crucial for 256-dim compatibility)
class CodeT5Embeddings(Embeddings):
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding"):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                embedding = self.model(**inputs).detach().cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# 2. Load Models
llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

# Replace OllamaEmbeddings with our local CodeT5 class
embeddings = CodeT5Embeddings()

# 3. Load Vector DB
# Note: Ensure collection_name matches what you used in build_index.py
db = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings,
    collection_name="notebook_rag_codet5"
)

def ask(question: str, k: int = 4, debug: bool = False) -> str:
    # This now uses the 256-dim search
    docs = db.similarity_search(question, k=k)

    if debug:
        print("\n--- DEBUG: Retrieved context ---")
        for i, d in enumerate(docs):
            cell_id = d.metadata.get('cell_id', 'Unknown')
            section = d.metadata.get('section', 'N/A')
            print(f"{i+1}. [Cell {cell_id}] | Section: {section}")

    context = "\n\n".join(
        f"[Cell {d.metadata.get('cell_id')}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
You are an expert AI assistant analyzing a Jupyter notebook for a user. Your goal is to explain code logic, data flow, and model training steps based ONLY on the retrieved notebook cells.

## Core Rules:
1. **Source-Based Only**: Answer strictly using the provided Context. If the context is insufficient, state "The provided notebook cells do not contain this information."
2. **Code & Intent**: Explain WHAT the code does (syntax) and WHY (intent/purpose). Mention specific variables (`df`, `model`) and functions (`train_test_split`).
3. **Flow Awareness**: Refer to cell order or sections (e.g., "In the Data Prep section...") if visible in metadata.
4. **No Hallucination**: Do not invent metrics, hyperparameters, or steps not shown in the context.

## Context:
{context}

## User Question:
{question}

## Answer:
""".strip()

    response = llm.invoke(prompt)
    return response.content