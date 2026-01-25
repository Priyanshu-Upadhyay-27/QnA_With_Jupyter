# 5: final object is build here
from parser import parse_notebook
from analyzer import analyze_code_cell
from explainer import explain_cell
from final_object_builder import assign_sections, build_final_object
import json
from pathlib import Path

NOTEBOOK_PATH = "SVM Training and EDA.ipynb"
OUTPUT_PATH = Path("artifacts/final_build_cell.json")

# Ensure artifacts dir exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: Parse
print("🔍 Parsing notebook...")
parsed = parse_notebook(NOTEBOOK_PATH)

# Step 2: Assign sections
print("📁 Assigning sections...")
parsed = assign_sections(parsed)

final_cells = []

# Step 3: Analyze, explain, and build final object
print("⚙️  Processing cells...")
for i, p in enumerate(parsed):
    print(f"  Processing cell {i + 1}/{len(parsed)}: {p.get('id', 'no-id')[:8]}...")
    analyzed = analyze_code_cell(p.copy())
    explained = explain_cell(analyzed.copy())
    final_cell = build_final_object(p, analyzed, explained)
    final_cells.append(final_cell)

# Step 4: Save
print(f"💾 Saving {len(final_cells)} cells to {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_cells, f, indent=2, ensure_ascii=False, default=str)

print("✅ Done! Your RAG-ready cells are in artifacts/final_build_cell.json")

# Step 5: Print EVERY field of EVERY final object (as requested)
print("\n" + "=" * 80)
print("📋 COMPLETE CUSTOM OBJECTS (every field):")
print("=" * 80)

for i, cell in enumerate(final_cells):
    print(f"\n🔸 CELL {i + 1}: {cell['cell_id'][:8]} ({cell['cell_type']})")
    print("-" * 60)
    for key, value in sorted(cell.items()):
        if isinstance(value, (list, dict)):
            print(f"  {key}: {json.dumps(value, ensure_ascii=False)[:100]}...")
        else:
            print(f"  {key}: {value}")
    print()

print("📊 SUMMARY: Generated", len(final_cells), "complete custom objects")

