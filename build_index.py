# In this, all the pre-built user defined functions are called and used here, to make the embeddings and
# store them in a vector store
from parser import parse_notebook
from analyzer import analyze_code_cell
from explainer import explain_cell
from final_object_builder import assign_sections, build_final_object
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from codeEmbedder import CodeT5Embeddings
from index_builder import build_collection
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
import json
import uuid



NOTEBOOK = "SVM Training and EDA.ipynb"

PARSED = "artifacts/parsed_cells.json"
ANALYZED = "artifacts/analyzed_cells.json"
EXPLAINED = "artifacts/explained_cells.json"
FINAL = "artifacts/custom_object.json"
RAG_DOCS = "artifacts/rag_documents.json"

print("Parsing Starts...")
raw_cells = parse_notebook(NOTEBOOK)
assigned_cells = assign_sections(raw_cells)
custom_doc_object = {}
print("Building Custom Document Object...")
for cell in assigned_cells:
    if cell["type"] == "code":
        analyzed_cell = analyze_code_cell(cell)

        explained_cell = explain_cell(analyzed_cell)


        cell_obj = build_final_object(cell, analyzed_cell, explained_cell)

        custom_doc_object[cell_obj["cell_id"]] = cell_obj
print(f"💾 Saving {len(custom_doc_object)} code cells to {FINAL}...")
with open(FINAL, "w", encoding="utf-8") as f:
    json.dump(custom_doc_object, f, indent=4)

print("Printing Custom Object:")
print(json.dumps(custom_doc_object, indent=4))

print(40*"=", "✅ Custom Object formed!", 40*"=")

########################################################################################################

print("🗂️ Starting Indexing...")


code_docs, text_docs = build_rag_documents(list(custom_doc_object.values()))

print(f"🧠 Code docs: {len(code_docs)}")
print(f"📄 Text docs: {len(text_docs)}")
save_documents(code_docs, "artifacts/code_docs.json")
save_documents(text_docs, "artifacts/text_docs.json")

print("🗂️ Starting Splitting...")
split_text_docs = split_text_documents(text_docs)
split_code_docs = split_code_documents(code_docs)
save_documents(split_code_docs, "artifacts/split_code_docs.json")
save_documents(split_text_docs, "artifacts/split_text_docs.json")

print("Starting saving the split chunks in the VectorDB...")
embeddings = CodeT5Embeddings()

# Build text collection
text_db = build_collection(
    documents=split_text_docs,
    embedding_model=OllamaEmbeddings(model="bge-m3"),
    collection_name="notebook_text_rag"
)

# Build code collection
code_db = build_collection(
    documents=split_code_docs,
    embedding_model=CodeT5Embeddings(),
    collection_name="notebook_code_rag"
)

print("Vector Database created successfully and embeddings are stored")
print("Indexing completed!")
