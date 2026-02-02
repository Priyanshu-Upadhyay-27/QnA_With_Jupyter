import json

PATH = "artifacts/final_build_cell.json"

with open(PATH, "r", encoding="utf-8") as f:
    cells = json.load(f)

for idx, cell in enumerate(cells):
    print("=" * 80)
    print(f"CELL {idx + 1} | ID: {cell.get('cell_id')}")
    print("-" * 80)

    for k, v in cell.items():
        print(f"{k}:")
        print(v)
        print()

    input("Press Enter to see next cell...")
