from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load models
llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
You are answering questions about a Jupyter notebook.

Rules:
- Use ONLY the context provided.
- If the answer is not in the context, say "The notebook does not contain this information."
- Do NOT add external knowledge.
- Be concise and precise.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    response = llm.invoke(prompt)
    return response.content
