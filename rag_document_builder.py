# 6: creates the RAG document object which can be circulated in the RAG pipeline
from langchain_core.documents import Document

def safe_scalar(value):
    """Convert ANY value to str/int/float/bool/None."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

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
    docs = []
    for cell in final_cells:
        if cell.get("cell_type") == "code":
            docs.append(final_cell_to_document(cell))

    print(f"✅ Created {len(docs)} RAG documents from {len(final_cells)} total cells")
    return docs






