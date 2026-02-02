from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def build_index(documents, persist_dir="./chroma_store"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectordb.persist()
    return vectordb


# Jina v3 setup
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode('test')

print(len(embeddings))




