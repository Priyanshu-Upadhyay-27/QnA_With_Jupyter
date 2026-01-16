# 1
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

        exec_count = cell.get("execution_count")
        exec_order = exec_count if exec_count is not None else idx

        parsed_cells.append({
            "id": idx,
            "type": cell_type,
            "exec_order": exec_order,
            "source": source.strip(),
            "outputs": output,
            "metadata": dict(cell.get("metadata", {}))
        })

    return parsed_cells

cells = parse_notebook("SVM Training and EDA.ipynb")

for c in cells:
    print("=" * 50)
    print(f"""Cell {c['id']} | Type: {c['type']} | Exec: {c['exec_order']}
     | Source: {c['source']} | Outputs: {c['outputs']} | Metadata: {c['metadata']}""")
    print("Outputs:", c["outputs"][:300])
    print("Source Code:", c["source"][:500])
