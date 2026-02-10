# 6: creates the RAG document object which can be circulated in the RAG pipeline
from langchain_core.documents import Document

def safe_scalar(value):
    """Convert ANY value to str/int/float/bool/None."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def build_code_document(cell: dict) -> Document:
    """Code-only document for CodeT5+ embeddings."""
    page_content = cell.get('source', '').strip()

    metadata = {
        "cell_id": safe_scalar(cell.get("cell_id")),
        "cell_index": safe_scalar(cell.get("cell_index")),
        "cell_type": safe_scalar(cell.get("cell_type")),
        "exec_order": safe_scalar(cell.get("exec_order")),
        "section": safe_scalar(cell.get("section")),
        "intent": safe_scalar(cell.get("intent")),
        "tags": safe_scalar(cell.get("tags")),
        "has_error": safe_scalar(cell.get("has_error")),
        "used": safe_scalar(cell.get("used")),
        "defined": safe_scalar(cell.get("defined")),
        "called_symbols": safe_scalar(cell.get("called_symbols")),
        "dependency_score": safe_scalar(cell.get("dependency_score")),
        "purpose": safe_scalar(cell.get("purpose")),
        "explanation": safe_scalar(cell.get("explanation")),
        "source_length": len(str(cell.get("source", "")))
    }
    return Document(page_content=page_content, metadata=metadata)

def build_text_document(cell: dict) -> Document:
    """Text-focused document for nomic/general embeddings."""
    page_content = f"""
Intent: {cell.get('intent', 'unknown')}
WHAT: {cell.get('purpose', '')}
WHY: {cell.get('explanation', '')}

Section: {cell.get('section', 'General')}
    """.strip()
    metadata = {
        "cell_id": safe_scalar(cell.get("cell_id")),
        "cell_index": safe_scalar(cell.get("cell_index")),
        "cell_type": safe_scalar(cell.get("cell_type")),
        "exec_order": safe_scalar(cell.get("exec_order")),
        "section": safe_scalar(cell.get("section")),
        "intent": safe_scalar(cell.get("intent")),
        "tags": safe_scalar(cell.get("tags")),
        "has_error": safe_scalar(cell.get("has_error")),
        "used": safe_scalar(cell.get("used")),
        "defined": safe_scalar(cell.get("defined")),
        "called_symbols": safe_scalar(cell.get("called_symbols")),
        "dependency_score": safe_scalar(cell.get("dependency_score")),
        "purpose": safe_scalar(cell.get("purpose")),
        "explanation": safe_scalar(cell.get("explanation")),
        "source_length": len(str(cell.get("source", "")))
    }
    return Document(page_content=page_content, metadata=metadata)

def final_cell_to_document(cell: dict) -> Document:
    page_content = f"""
Intent: {cell.get('intent', 'unknown')}
WHAT: {cell.get('purpose', '')}
WHY: {cell.get('explanation', '')}

Code:
{cell.get('source', '')}
    """.strip()

    # ✅ ALL 15 FIELDS - SAFE CONVERSION
    metadata = {
        "cell_id": safe_scalar(cell.get("cell_id")),
        "cell_index": safe_scalar(cell.get("cell_index")),
        "cell_type": safe_scalar(cell.get("cell_type")),
        "exec_order": safe_scalar(cell.get("exec_order")),
        "section": safe_scalar(cell.get("section")),
        "intent": safe_scalar(cell.get("intent")),
        "tags": safe_scalar(cell.get("tags")),
        "has_error": safe_scalar(cell.get("has_error")),
        "used": safe_scalar(cell.get("used")),
        "defined": safe_scalar(cell.get("defined")),
        "called_symbols": safe_scalar(cell.get("called_symbols")),
        "dependency_score": safe_scalar(cell.get("dependency_score")),
        "purpose": safe_scalar(cell.get("purpose")),
        "explanation": safe_scalar(cell.get("explanation")),
        "source_length": len(str(cell.get("source", "")))
    }

    return Document(page_content=page_content, metadata=metadata)


def build_rag_documents(final_cells: list) -> list:
    """Returns TWO document lists: code_docs, text_docs."""
    code_docs = []
    text_docs = []

    for cell in final_cells:
        if cell.get("cell_type") == "code":
            code_docs.append(build_code_document(cell))
            text_docs.append(build_text_document(cell))

    print(f"✅ Created {len(code_docs)} code docs + {len(text_docs)} text docs")
    return code_docs, text_docs






