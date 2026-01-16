# 6: creates the RAG document object which can be circulated in the RAG pipeline
from langchain_core.documents import Document

def final_cell_to_document(cell: dict) -> Document:
    """
    Convert a structured NotebookCell into a RAG-ready Document.
    """

    text = f"""
Cell ID: {cell['cell_id']}
Cell Type: {cell['cell_type']}
Execution Order: {cell['exec_order']}
Section: {cell.get('section')}

Purpose:
{cell.get('purpose')}

Explanation:
{cell.get('explanation')}

Uses: {", ".join(cell.get("used", []))}
Defines: {", ".join(cell.get("defined", []))}

Code:
{cell.get("source")}
""".strip()

    metadata = {
        "cell_id": cell["cell_id"],
        "cell_type": cell["cell_type"],
        "exec_order": cell["exec_order"],
        "section": cell.get("section")
    }

    return Document(page_content=text, metadata=metadata)


def build_rag_documents(final_cells: list) -> list:
    """
    Convert list of final structured cells into RAG Documents.
    """
    docs = []
    for cell in final_cells:
        if cell["cell_type"] == "code":  # only embed code cells
            docs.append(final_cell_to_document(cell))
    return docs
