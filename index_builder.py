from langchain_chroma import Chroma

CUSTOM_OBJECT_PATH = "artifacts/final_build_cell.json"
PERSIST_DIR = "./chroma_store"

CODE_COLLECTION = "code_embeddings"
TEXT_COLLECTION = "text_embeddings"


def build_collection(documents, embedding_model, collection_name: str, persist_dir: str = "./chroma_store"):
    """
    Builds a single Chroma collection.

    Args:
        documents: List[Document]
        embedding_model: Embedding wrapper (CodeT5Embeddings or OllamaEmbeddings)
        collection_name: Name of the collection
        persist_dir: Base directory for Chroma storage

    Returns:
        Chroma vector DB instance
    """

    if not documents:
        raise ValueError(f"No documents provided for collection '{collection_name}'")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    print(f"✅ Collection '{collection_name}' built with {len(documents)} documents")

    return vectordb


def load_collection(embedding_model, collection_name: str, persist_dir: str = "./chroma_store"):
    """
    Loads an existing Chroma collection.
    IMPORTANT: embedding_model must match the one used during build.
    """

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    print(f"📂 Collection '{collection_name}' loaded")

    return vectordb
