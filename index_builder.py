from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def build_index(documents, persist_dir="./chroma_store"):
    """
    Build ChromaDB vector index from documents using BGE-M3 embeddings.

    Args:
        documents: List[Document] - LangChain Document objects
        persist_dir: str - Directory to persist ChromaDB

    Returns:
        Chroma vectorstore instance
    """

    # BGE-M3 embeddings (optimized for code)
    embeddings = OllamaEmbeddings(model="bge-m3")

    print(f"Building index for {len(documents)} documents...")

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="notebook_rag"  # Named collection
    )

    print(f"✅ Index built! {len(documents)} docs stored in {persist_dir}")
    return vectordb







