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

    def retrieve_debug(self, query: str, k: int = 3):
        """Diagnostic tool to expose the raw math and chunk text."""
        print(f"\n🔎 DEBUG QUERYING: '{query}'")

        # 1. Check the DB size
        total_chunks = self.vector_db._collection.count()
        print(f"📊 Total chunks in ChromaDB: {total_chunks}")

        # 2. Perform raw similarity search to expose the distance scores!
        results = self.vector_db.similarity_search_with_score(query, k=k)

        for i, (chunk, distance) in enumerate(results):
            cell_id = chunk.metadata.get("cell_id", "UNKNOWN")
            # Print the math score and a snippet of the text
            print(f"   [{i + 1}] Distance: {distance:.4f} | Cell: {cell_id} | Text: {chunk.page_content[:60]}...")

        return []  # Just returning empty for now since we are just debugging

    def retrieve(self, query: str, max_cells: int = 3):
        print(f"\n🔎 Querying: '{query}'")

        # 1. Over-fetch: Ask Chroma for 15 chunks using standard similarity search
        results = self.vector_db.similarity_search(query, k=15)

        final_results = []
        seen_cell_ids = set()

        for chunk in results:
            cell_id = chunk.metadata.get("cell_id")

            # Skip if there's no ID, or if we already grabbed this Parent Cell
            if not cell_id or cell_id in seen_cell_ids:
                continue

            # 2. Deduplication: We found a brand new unique Parent Cell!
            seen_cell_ids.add(cell_id)

            # 3. The Switch: Fetch the rich, uncut payload directly from JSON
            if cell_id in self.doc_store:
                full_cell = self.doc_store[cell_id]
                final_results.append(full_cell)
            else:
                print(f"❌ [DEBUG] ERROR: cell_id {cell_id} is NOT in custom_object.json!")

            # 4. Stop exactly when we hit our target number of UNIQUE cells
            if len(final_results) == max_cells:
                break

        print(f"🛠️ [DEBUG] Successfully grabbed {len(final_results)} unique parent cells.")
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


if __name__ == "__main__":
    retriever = RelationalRetriever()

    # Test the ACTUAL retrieve function with the over-fetch fix
    print("\n" + "=" * 50)
    print("TESTING QUERY 1: SVM Accuracy")
    print("=" * 50)
    results_1 = retriever.retrieve("What was the accuracy of the SVM model?")
    print(retriever.format_for_llm(results_1))

    print("\n" + "=" * 50)
    print("TESTING QUERY 2: Cross Validation")
    print("=" * 50)
    results_2 = retriever.retrieve("Why and where cross validation is used in the notebook?")
    print(retriever.format_for_llm(results_2))