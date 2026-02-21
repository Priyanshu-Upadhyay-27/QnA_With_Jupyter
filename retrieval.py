import json
import os
from langchain_chroma import Chroma
from codeEmbedder import CodeT5Embeddings
from langchain_ollama import OllamaEmbeddings


class RelationalRetriever:
    def __init__(self, persist_dir: str = "./chroma_store", doc_store_path: str = "artifacts/doc_store.json"):
        """
        Initializes the Relational Retriever.
        1. Loads the Vector Database (for finding the 'Anchor').
        2. Loads the DocStore (for finding the 'Full Content').
        3. Builds a fast 'Cell Index' to instantly find related documents.
        """
        print("🚀 Initializing Relational Retriever...")

        # 2. Use your custom embedding class!
        self.embedding_function = OllamaEmbeddings(model="bge-m3")

        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_function,
            collection_name="notebook_text_rag"
        )
        print(f"📊 [DEBUG] Chroma Collection Count: {self.vector_db._collection.count()}")

        # 2. Load DocStore
        if not os.path.exists(doc_store_path):
            raise FileNotFoundError(f"❌ DocStore not found at {doc_store_path}")

        with open(doc_store_path, "r") as f:
            self.doc_store = json.load(f)

        # 3. Build Reverse Index (Cell ID -> Parent IDs)
        # This makes finding the "Twin" (Other Half) O(1) instant.
        self.cell_index = {}
        self._build_fast_lookup_index()
        print(f"✅ Retriever Ready! Indexed {len(self.cell_index)} unique cells.")

    def _build_fast_lookup_index(self):
        """
        Iterates through DocStore to create a map:
        { cell_id: [parent_id_code, parent_id_explanation] }
        """
        for parent_id, doc in self.doc_store.items():
            # Get the cell_id from metadata (stored as string or int in JSON)
            cell_id = doc["metadata"].get("cell_id")

            if cell_id is not None:
                # Ensure cell_id is a string for consistent dict keys
                cell_id = str(cell_id)

                if cell_id not in self.cell_index:
                    self.cell_index[cell_id] = []
                self.cell_index[cell_id].append(parent_id)

    def retrieve(self, query: str, k: int = 3):
        print(f"\n🔎 Querying: '{query}'")

        results = self.vector_db.similarity_search(query, k=k)
        print(f"🛠️ [DEBUG] Chroma returned {len(results)} chunks.")

        final_results = []
        seen_cell_ids = set()

        for i, chunk in enumerate(results):
            parent_id = chunk.metadata.get("parent_id")
            print(f"🛠️ [DEBUG] Result {i + 1} parent_id: {parent_id}")

            if not parent_id:
                print("❌ [DEBUG] ERROR: Chroma chunk has no parent_id metadata!")
                continue

            if parent_id not in self.doc_store:
                print(f"❌ [DEBUG] ERROR: parent_id {parent_id} is NOT in doc_store.json!")
                continue

            # Get the full anchor document to find its cell_id
            anchor_doc = self.doc_store[parent_id]
            cell_id = str(anchor_doc["metadata"].get("cell_id"))

            # De-duplication: If we already processed this cell, skip it.
            if cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(cell_id)

            # Step 3: Relational Fetch (Get ALL parts of this cell)
            # We look up the cell_id in our pre-built index
            related_parent_ids = self.cell_index.get(cell_id, [])

            cell_context = {
                "cell_id": cell_id,
                "code": "",
                "explanation": "",
                "score": 1.0  # Placeholder, can use chunk.score if available
            }

            for p_id in related_parent_ids:
                full_doc = self.doc_store[p_id]

                # Simple Heuristic: If it looks like code, it's code.
                content = full_doc["page_content"]

                if "def " in content or "import " in content or "=" in content:
                    cell_context["code"] += content + "\n"
                else:
                    cell_context["explanation"] += content + "\n"

            final_results.append(cell_context)

        return final_results

    def format_for_llm(self, retrieved_results):
        """
        Converts the structured results into a string prompt for the LLM.
        """
        prompt_context = ""
        for item in retrieved_results:
            prompt_context += f"--- CONTEXT FOR CELL {item['cell_id']} ---\n"
            if item['code'].strip():
                prompt_context += f"[CODE IMPLEMENTATION]:\n{item['code']}\n"
            if item['explanation'].strip():
                prompt_context += f"[EXPLANATION]:\n{item['explanation']}\n"
            prompt_context += "------------------------------------------\n\n"

        return prompt_context


# --- How to Run ---
if __name__ == "__main__":
    retriever = RelationalRetriever()
    results = retriever.retrieve("What dataframe is used in the jupyter notebook")
    reference = retriever.format_for_llm(results)
    print(reference)