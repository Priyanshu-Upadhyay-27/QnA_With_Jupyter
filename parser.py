# 1:
# 1: parser.py
import nbformat
from typing import List, Dict


def load_notebook(path: str):
    return nbformat.read(path, as_version=4)


def parse_notebook(path: str) -> List[Dict]:
    nb = load_notebook(path)
    parsed_cells = []

    for idx, cell in enumerate(nb.cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        output = cell.get("outputs", [])
        metadata = dict(cell.get("metadata", {}))

        exec_count = cell.get("execution_count")
        exec_order = exec_count

        # FIXED: Consistent field names + new useful fields
        parsed_cells.append({
            "id": cell.get("id", f"cell-{idx}"),
            "cell_index": idx,  # Renamed for consistency
            "type": cell_type,
            "exec_order": exec_order,
            "source": source.strip(),
            "outputs": output,
            "metadata": metadata,
            "has_error": any(o.get("output_type") == "error" for o in output),  # NEW
            "tags": metadata.get("tags", []),  # NEW: extract tags
        })

    return parsed_cells


if __name__ == "__main__":
    cells = parse_notebook("SVM Training and EDA.ipynb")

    for c in cells:
        print("=" * 50)
        print(f"""Cell {c['id'][:8]} | Type: {c['type']} | Index: {c['cell_index']} | Exec: {c['exec_order']}""")
        print(f"  Section hint: {c['metadata'].get('section', 'None')}")
        print(f"  Tags: {c['tags']}")
        print(f"  Has error: {c['has_error']}")
        print(f"  Source preview: {c['source'][:100]}...")

