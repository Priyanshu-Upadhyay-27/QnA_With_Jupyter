# 6: creates the RAG document object which can be circulated in the RAG pipeline
from langchain_core.documents import Document


# ----------------------------
# Strict scalar enforcement
# ----------------------------
def safe_scalar(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Invalid metadata type: {type(value)}")


# ----------------------------
# Code document
# ----------------------------
def build_code_document(cell: dict) -> Document | None:
    """Code-only document for code embeddings."""
    code = cell.get("source", "").strip()
    if not code:
        return None

    metadata = {
        "cell_id": safe_scalar(cell.get("cell_id")),
        "cell_index": safe_scalar(cell.get("cell_index")),
        "section": safe_scalar(cell.get("section")),
        "intent": safe_scalar(cell.get("intent")),
        "has_error": safe_scalar(cell.get("has_error")),
        "dependency_score": safe_scalar(cell.get("dependency_score")),
    }

    return Document(
        page_content=code,
        metadata=metadata
    )


# ----------------------------
# Text / explanation document
# ----------------------------
def build_text_document(cell: dict) -> Document | None:
    """Text-only document for semantic embeddings."""
    if cell.get("explanation_error"):
        return None

    purpose = cell.get("purpose", "").strip()
    explanation = cell.get("explanation", "").strip()

    if not purpose and not explanation:
        return None

    text = f"""WHAT:
{purpose}

WHY:
{explanation}
""".strip()

    metadata = {
        "cell_id": safe_scalar(cell.get("cell_id")),
        "cell_index": safe_scalar(cell.get("cell_index")),
        "section": safe_scalar(cell.get("section")),
        "intent": safe_scalar(cell.get("intent")),
        "has_error": safe_scalar(cell.get("has_error")),
    }

    return Document(
        page_content=text,
        metadata=metadata
    )


# ----------------------------
# Builder
# ----------------------------
def build_rag_documents(final_cells: list):
    code_docs = []
    text_docs = []

    for cell in final_cells:
        if cell.get("cell_type") != "code":
            continue

        code_doc = build_code_document(cell)
        if code_doc:
            code_docs.append(code_doc)

        text_doc = build_text_document(cell)
        if text_doc:
            text_docs.append(text_doc)

    print(f"✅ Code docs: {len(code_docs)}")
    print(f"✅ Text docs: {len(text_docs)}")

    return code_docs, text_docs







