from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load models
llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

embeddings = OllamaEmbeddings(model="bge-m3")

# Load vector DB (NO rebuild)
db = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings
)


def ask(question: str, k: int = 4, debug: bool = False) -> str:
    docs = db.similarity_search(question, k=k)

    if debug:
        print("\nRetrieved context:")
        for d in docs:
            print(f"- Cell {d.metadata.get('cell_id')} | Section: {d.metadata.get('section')}")

    context = "\n\n".join(
        f"[Cell {d.metadata.get('cell_id')}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
You are an expert AI assistant analyzing a Jupyter notebook for a user. Your goal is to explain code logic, data flow, and model training steps based ONLY on the retrieved notebook cells.

## Core Rules:
1. **Source-Based Only**: Answer strictly using the provided Context. If the context is insufficient, state "The provided notebook cells do not contain this information."
2. **Code & Intent**: Explain WHAT the code does (syntax) and WHY (intent/purpose). Mention specific variables (`df`, `model`) and functions (`train_test_split`).
3. **Flow Awareness**: Refer to cell order or sections (e.g., "In the Data Prep section...") if visible in metadata.
4. **No Hallucination**: Do not invent metrics, hyperparameters, or steps not shown in the context.

## Context Structure:
Each context chunk represents a notebook cell with:
- **Intent/Purpose**: High-level goal (e.g., "Model Training")
- **Code**: The actual Python code executed
- **Explanation**: A summary of the logic
- **Metadata**: Variables used/defined, execution order

## Context:
{context}

## User Question:
{question}

## Answer:
""".strip()

    response = llm.invoke(prompt)
    return response.content
