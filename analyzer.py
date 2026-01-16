# 2: Used to extract used and defined variables, imports
import ast
from typing import Set, Dict
from parser import parse_notebook
import builtins
BUILTINS = set(dir(builtins))


def get_defined_vars(code: str) -> Set[str]:
    tree = ast.parse(code)
    defined = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)

    return defined


def get_used_vars(code: str) -> Set[str]:
    tree = ast.parse(code)
    used = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    return used


def analyze_code_cell(cell: Dict) -> Dict:
    if cell["type"] != "code":
        return cell

    code = cell["source"]

    try:
        defined = get_defined_vars(code)
        used = get_used_vars(code)
    except SyntaxError:
        defined, used = set(), set()

    external_inputs = {
        var for var in used - defined
        if var not in BUILTINS
    }

    cell["defined"] = list(defined)
    cell["used"] = list(external_inputs)

    return cell


cells = parse_notebook("SVM Training and EDA.ipynb")
analyzed = [analyze_code_cell(c) for c in cells]

for c in analyzed:
    if c["type"] == "code":
        print(f"\nCell {c['id']}")
        print("Defined:", c["defined"])
        print("Used:", c["used"])