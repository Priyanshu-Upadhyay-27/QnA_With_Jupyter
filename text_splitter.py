from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_rag_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

from rag_document_builder import build_rag_documents
from build_cells import final_cells

rag_docs = build_rag_documents(final_cells)
rag_chunks = split_rag_documents(rag_docs)
