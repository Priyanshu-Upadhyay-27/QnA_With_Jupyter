from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_index(documents, persist_dir="./chroma_store"):
    print(f"Loading embedding model (this may take a minute first time)...")

    # ✅ FIX: trust_remote_code=True is REQUIRED for Salesforce models
    embeddings = HuggingFaceEmbeddings(
        model_name="Salesforce/codet5p-110m-embedding",
        model_kwargs={
            "device": "cuda",  # Use "cpu" if CUDA fails
            "trust_remote_code": True  # <--- CRITICAL FIX
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(f"Building index for {len(documents)} documents...")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="notebook_rag"
    )

    print(f"✅ Index built! {len(documents)} docs in {persist_dir}")
    return vectordb


