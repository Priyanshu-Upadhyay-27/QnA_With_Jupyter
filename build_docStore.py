import json
import os
from storage import load_documents, save_documents
from rag_text_splitter import split_text_documents, split_code_documents


def build_rag_artifacts():
    print("📂 Loading original documents...")
    text_docs = load_documents("artifacts/text_docs.json")
    code_docs = load_documents("artifacts/code_docs.json")

    print("✂️ Splitting and Tagging...")

    split_text_chunks = split_text_documents(text_docs)

    # This updates 'code_docs' in-place!
    split_code_chunks = split_code_documents(code_docs)

    print("🏗️ Building DocStore from tagged parents...")
    doc_store = {}

    def add_to_store(docs):
        for doc in docs:
            # We can trust 'parent_id' exists because the splitter added it
            p_id = doc.metadata["parent_id"]

            doc_store[p_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }

    add_to_store(text_docs)
    add_to_store(code_docs)

    # Save Everything
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")

    print(f"💾 Saving DocStore ({len(doc_store)} items)...")
    with open("artifacts/doc_store.json", "w") as f:
        json.dump(doc_store, f, indent=4)

    print(f"💾 Saving Vectors ({len(split_text_chunks) + len(split_code_chunks)} chunks)...")
    save_documents(split_text_chunks, "artifacts/split_text_docs.json")
    save_documents(split_code_chunks, "artifacts/split_code_docs.json")

    print("✅ Build Complete!")


build_rag_artifacts()

