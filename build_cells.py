# 5: final object is build here
from parser import parse_notebook
from analyzer import analyze_code_cell
from explainer import explain_cell
from final_object_builder import assign_sections, build_final_object

NOTEBOOK_PATH = "SVM Training and EDA.ipynb"

# Step 1: Parse
parsed = parse_notebook(NOTEBOOK_PATH)

# Step 2: Assign sections
parsed = assign_sections(parsed)

final_cells = []

# Step 3: Analyze, explain, and build final object
for p in parsed:
    analyzed = analyze_code_cell(p.copy())
    explained = explain_cell(analyzed.copy())
    final_cell = build_final_object(p, analyzed, explained)
    final_cells.append(final_cell)

# Step 4: Inspect result
for c in final_cells:
    print("=" * 40)
    for k, v in c.items():
        print(f"{k}: {v}")
