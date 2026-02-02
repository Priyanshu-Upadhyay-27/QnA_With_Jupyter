# 2: analyzer.py
# Purpose:
# Extract structural + semantic signals from code cells.
# These are HEURISTICS, not a full dependency graph.

import ast
from typing import Set, Dict, List
import builtins

BUILTINS = set(dir(builtins))


# --------------------------------------------------
# Extract variables defined in the cell
# --------------------------------------------------
def get_defined_vars(tree: ast.AST) -> Set[str]:
    defined = set()

    for node in ast.walk(tree):

        # x = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)

        # def foo():
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)

        # function arguments
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            for arg in node.args.args:
                defined.add(arg.arg)


        # class Foo:
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)

        # import numpy as np
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])

        # from sklearn.model_selection import train_test_split
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)

        # for i in range(...)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)

        # with open(...) as f
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    defined.add(item.optional_vars.id)

    return defined


# --------------------------------------------------
# Extract variables used (read) in the cell
# --------------------------------------------------
def get_used_vars(tree: ast.AST) -> Set[str]:
    used = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    return used


# --------------------------------------------------
# Extract called functions / methods
# --------------------------------------------------
def extract_called_symbols(tree: ast.AST) -> List[str]:
    """
    Examples:
    - pd.read_csv
    - model.fit
    - plt.show
    """
    called = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):

            # obj.method(...)
            if isinstance(node.func, ast.Attribute):
                parts = []
                obj = node.func

                while isinstance(obj, ast.Attribute):
                    parts.append(obj.attr)
                    obj = obj.value

                if isinstance(obj, ast.Name):
                    parts.append(obj.id)

                called.add(".".join(reversed(parts)))

            # function(...)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)

    return sorted(called)


# --------------------------------------------------
# Main analyzer entry point
# --------------------------------------------------
def analyze_code_cell(cell: Dict) -> Dict:
    if cell.get("type") != "code":
        return cell

    code = cell.get("source", "")

    try:
        tree = ast.parse(code)

        defined = get_defined_vars(tree)
        used = get_used_vars(tree)
        called_symbols = extract_called_symbols(tree)

    except SyntaxError:
        defined, used, called_symbols = set(), set(), []

    # Heuristic external dependencies
    external_inputs = {
        var for var in used
        if var not in defined and var not in BUILTINS
    }

    # Store ALL signals explicitly
    cell["defined"] = sorted(defined)
    cell["used"] = sorted(used)
    cell["external_inputs"] = sorted(external_inputs)
    cell["called_symbols"] = sorted(called_symbols)

    return cell