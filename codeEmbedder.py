import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from typing import List


class CodeT5Embeddings(Embeddings):
    """
    A custom LangChain-compatible wrapper for the Salesforce CodeT5+ 110M model.
    Optimized for CPU inference on Windows laptops with limited VRAM.
    """

    def __init__(self, model_name="Salesforce/codet5p-110m-embedding"):
        # Explicitly use CPU to avoid 128MB VRAM bottlenecks
        self.device = "cpu"

        print(f"Initializing {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # Truncate to 512 to stay within model limits
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)

                # The model outputs 256-dimensional normalized embeddings
                result = self.model(**inputs)
                embedding = result.detach().cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of notebook cells or code files."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question."""
        return self._embed([text])[0]