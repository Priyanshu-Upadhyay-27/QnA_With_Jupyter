from langchain_chroma import Chroma  # ✅ FIXED: New import
from langchain_ollama import OllamaEmbeddings


def build_index(documents, persist_dir="./chroma_store"):
    """
    Build ChromaDB vector index from documents using Ollama bge-m3 embeddings.
    """
    # Your Ollama bge-m3 ✅ Perfect
    embeddings = OllamaEmbeddings(model="bge-m3")

    print(f"Building index for {len(documents)} documents...")

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="notebook_rag"
    )

    print(f"✅ Index built & persisted! {len(documents)} docs in {persist_dir}")
    return vectordb
