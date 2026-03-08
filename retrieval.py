import json
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class RelationalRetriever:
    def __init__(self, persist_dir: str = "./chroma_store", doc_store_path: str = "artifacts/custom_object.json"):
        """
        Initializes the Relational Retriever.
        """
        print("🚀 Initializing Relational Retriever...")

        # 1. Load the Embedding Model
        self.embedding_function = OllamaEmbeddings(model="bge-m3")

        # 2. Connect to ChromaDB
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_function,
            collection_name="notebook_text_rag"
        )

        # 3. Load the Pure JSON Warehouse
        if not os.path.exists(doc_store_path):
            raise FileNotFoundError(f"❌ DocStore not found at {doc_store_path}")

        with open(doc_store_path, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)

        print(f"✅ Retriever Ready! Loaded {len(self.doc_store)} unique cells.")

    def retrieve(self, query: str, k: int = 3):
        print(f"\n🔎 Querying: '{query}'")

        # MMR is perfect here for diverse chunk retrieval
        results = self.vector_db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=10,
            lambda_mult=0.5
        )
        print(f"🛠️ [DEBUG] Chroma returned {len(results)} chunks.")

        final_results = []
        seen_cell_ids = set()

        for i, chunk in enumerate(results):
            # 1. The Bait: Extract the cell_id directly from the chunk's metadata
            cell_id = chunk.metadata.get("cell_id")

            if not cell_id:
                print("❌ [DEBUG] ERROR: Chroma chunk has no cell_id metadata!")
                continue

            # 2. Deduplication: Skip if we already grabbed this parent cell
            if cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(cell_id)

            # 3. The Switch: Fetch the rich, uncut payload directly from JSON
            if cell_id in self.doc_store:
                full_cell = self.doc_store[cell_id]
                final_results.append(full_cell)
            else:
                print(f"❌ [DEBUG] ERROR: cell_id {cell_id} is NOT in custom_object.json!")

        return final_results

    def format_for_llm(self, retrieved_results):
        """
        Injects the uncut code, explanations, and outputs into a beautiful prompt.
        """
        prompt_context = ""
        for i, item in enumerate(retrieved_results):
            prompt_context += f"--- FULL CONTEXT FOR CELL {item.get('cell_id')} ---\n"

            # Injecting the rich semantic data
            prompt_context += f"[INTENT / PURPOSE]:\n{item.get('purpose', 'N/A')}\n\n"
            prompt_context += f"[EXPLANATION]:\n{item.get('explanation', 'N/A')}\n\n"

            # Injecting the exact Python code
            prompt_context += f"[PYTHON CODE]:\n```python\n{item.get('source', 'No code')}\n```\n\n"

            # Injecting the newly cleaned outputs and result summaries!
            prompt_context += f"[TERMINAL OUTPUT]:\n{item.get('outputs', 'No output')}\n\n"
            prompt_context += f"[STATISTICAL RESULT]:\n{item.get('result_summary', 'N/A')}\n"
            prompt_context += "------------------------------------------\n\n"

        return prompt_context


# --- How to Run ---
if __name__ == "__main__":
    retriever = RelationalRetriever().

    # Test it with a real question!
    results = retriever.retrieve("How did we handle missing values in the dataset?")
    reference = retriever.format_for_llm(results)

    print(reference)