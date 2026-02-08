import torch
from transformers import AutoModel, AutoTokenizer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import List

# 1. Custom Class to fix the "RuntimeError: expand" dimension mismatch
class CodeT5Embeddings(Embeddings):
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding"):
        # We use CPU because Intel UHD 128MB VRAM is not enough for CUDA
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                # CodeT5+ outputs the correct 256-dim vector directly
                embedding = self.model(**inputs).detach().cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# 2. Your Updated build_index function
def build_index(documents, persist_dir="./chroma_store"):
    print(f"Loading CodeT5+ embedding model on CPU...")

    # Initialize our custom wrapper instead of HuggingFaceEmbeddings
    embeddings = CodeT5Embeddings()

    print(f"Building index for {len(documents)} documents...")

    # Chroma will now use our custom class to process documents
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="notebook_rag"
    )

    print(f"✅ Index built! {len(documents)} docs in {persist_dir}")
    return vectordb