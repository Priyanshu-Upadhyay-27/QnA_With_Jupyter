# In this, all the pre-built user defined functions are called and used here, to make the embeddings and
# store them in a vector store
from parser import parse_notebook
from analyzer import analyze_code_cell
from explainer import explain_cell
from final_object_builder import assign_sections, build_final_object
from rag_document_builder import build_rag_documents
from text_splitter import split_rag_documents
from index_builder import build_index
from storage import (
    save,
    load,
    exists,
    save_rag_documents,
    save_split_documents
)
from langchain_core.documents import Document


NOTEBOOK = "SVM Training and EDA.ipynb"

PARSED = "artifacts/parsed_cells.json"
ANALYZED = "artifacts/analyzed_cells.json"
EXPLAINED = "artifacts/explained_cells.json"
FINAL = "artifacts/final_cells.json"
RAG_DOCS = "artifacts/rag_documents.json"
SPLIT_DOCS = "artifacts/split_documents.json"


def main():

    # 1. PARSE
    if exists(PARSED):
        parsed = load(PARSED)
    else:
        parsed = parse_notebook(NOTEBOOK)
        parsed = assign_sections(parsed)
        save(parsed, PARSED)

    # 2. ANALYZE
    if exists(ANALYZED):
        analyzed = load(ANALYZED)
    else:
        analyzed = [analyze_code_cell(c) for c in parsed]
        save(analyzed, ANALYZED)

    # 3. EXPLAIN (OLLAMA — expensive)
    if exists(EXPLAINED):
        explained = load(EXPLAINED)
    else:
        explained = [explain_cell(c) for c in analyzed]
        save(explained, EXPLAINED)

    # 4. FINAL OBJECT
    if exists(FINAL):
        final_cells = load(FINAL)
    else:
        final_cells = [
            build_final_object(p, a, e)
            for p, a, e in zip(parsed, analyzed, explained)
        ]
        save(final_cells, FINAL)

    # 5. RAG DOCUMENTS (persist + reload)
    if exists(RAG_DOCS):
        rag_docs_raw = load(RAG_DOCS)
        rag_docs = [
            Document(
                page_content=d["page_content"],
                metadata=d["metadata"]
            )
            for d in rag_docs_raw
        ]
    else:
        rag_docs = build_rag_documents(final_cells)
        save_rag_documents(rag_docs, RAG_DOCS)

    # 6. SPLIT DOCUMENTS (persist + reload)
    # I have removed the text splitter, since jupyter cells have less code so, each page_content of a document is small
    # enough for a llm to understand.
    rag_chunks = rag_docs

    # 7. EMBEDDINGS
    build_index(rag_chunks)

    print("Index built successfully")


if __name__ == "__main__":
    main()
