# 6: creates the RAG document object which can be circulated in the RAG pipeline
from langchain_core.documents import Document

def final_cell_to_document(cell: dict) -> Document:
    """
    Convert a structured NotebookCell into a RAG-ready Document.
    Uses Intent + WHAT + WHY + Code structure for optimal embeddings.
    """

    # Page content: What gets embedded (Intent + Purpose + Explanation + Code)
    page_content = f"""
Intent: {cell.get('intent', 'unknown')}
WHAT: {cell.get('purpose', '')}
WHY: {cell.get('explanation', '')}

Code:
{cell.get('source', '')}
    """.strip()

    # Metadata: Structured fields for filtering and ranking (NOT embedded)
    metadata = {
        "cell_id": cell.get("cell_id"),
        "cell_index": cell.get("cell_index"),
        "cell_type": cell.get("cell_type"),
        "exec_order": cell.get("exec_order"),
        "section": cell.get("section"),
        "intent": cell.get("intent"),
        "tags": cell.get("tags", []),
        "has_error": cell.get("has_error", False),
        "used": cell.get("used", []),
        "defined": cell.get("defined", []),
        "called_symbols": cell.get("called_symbols", []),
        "dependency_score": cell.get("dependency_score", 0)
    }

    return Document(page_content=page_content, metadata=metadata)


def build_rag_documents(final_cells: list) -> list:
    """
    Convert list of final structured cells into RAG Documents.
    Only processes code cells for embedding.
    """
    docs = []
    for cell in final_cells:
        if cell.get("cell_type") == "code":  # only embed code cells
            docs.append(final_cell_to_document(cell))

    print(f"✅ Created {len(docs)} RAG documents from {len(final_cells)} total cells")
    return docs

