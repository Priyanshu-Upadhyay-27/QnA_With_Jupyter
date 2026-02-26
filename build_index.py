# In this, all the pre-built user defined functions are called and used here, to make the embeddings and
# store them in a vector store
from parser import parse_notebook
from analyzer import analyze_code_cell
from explainer import explain_cell
from final_object_builder import assign_sections, build_final_object
from rag_document_builder import build_rag_documents
from rag_text_splitter import split_text_documents, split_code_documents
from storage import (
    save,
    load,
    exists,
    save_documents,
    load_documents
)
from langchain_core.documents import Document


NOTEBOOK = "SVM Training and EDA.ipynb"

PARSED = "artifacts/parsed_cells.json"
ANALYZED = "artifacts/analyzed_cells.json"
EXPLAINED = "artifacts/explained_cells.json"
FINAL = "artifacts/final_cells.json"
RAG_DOCS = "artifacts/rag_documents.json"



