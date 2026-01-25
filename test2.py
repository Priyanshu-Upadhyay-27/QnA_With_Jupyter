from langchain_chroma import Chroma
from rag_document_builder import final_cell_to_document, build_rag_documents
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from storage import load
from rag_document_builder import build_rag_documents
from dotenv import load_dotenv
load_dotenv()

#############
# Doc Loader
#############
final_custom_object = load("artifacts/final_cells.json")
document = build_rag_documents(final_custom_object)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma.from_documents(
    documents = document,
    embedding=embeddings,
    persist_directory="./chroma_store_test2"
)
prompt = """
You are a technical code analyzer.

Task: Analyze the provided code and respond with exactly TWO statements:

1. WHAT: Identify the SPECIFIC operation/algorithm/technique in the code
   - Name the exact library/function used (e.g., "GridSearchCV", "KMeans", "train_test_split")
   - Mention key parameters if relevant
   - Maximum 30 words

2. WHY: State the TECHNICAL reason this operation is needed
   - Focus on the prerequisite or purpose (e.g., "hyperparameter tuning", "avoid data leakage")
   - Be specific to the code's context
   - Maximum 30 words

Format:
WHAT: [specific operation from the code]
WHY: [technical reason based on the code's purpose]

Critical:
- Extract information FROM the code, don't assume patterns
- Different code types need different explanations
- Be precise, not generic
"""

