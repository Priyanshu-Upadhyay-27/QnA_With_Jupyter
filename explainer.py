# 3: Used to write explanation for each code block
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import re

load_dotenv()

llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

SYSTEM_PROMPT = """
You are a precise static code analyzer.

Task: Analyze the provided Python code cell and respond with EXACTLY THREE lines.

Your analysis must explicitly reference variable names and their roles.

------------------------------------------------------------
1. WHAT:
Describe the EXACT operation, algorithm, or technique implemented.
- Name the specific library, function, or class used.
- Mention key variables defined in this cell.
- If a class is instantiated, distinguish between:
  - Class name (e.g., SVC)
  - Instance variable (e.g., model = SVC())
- Identify whether variables act as:
  - input data
  - configuration parameters
  - model/class instance
  - intermediate variables
  - output variables
Maximum 50 words.

------------------------------------------------------------
2. WHY:
Explain the technical purpose of this cell in context.
- Use actual variable names from the code.
- Explain why the defined variables are necessary.
- Clarify how the operation affects later computation (e.g., model training, data transformation, evaluation).
- Do NOT give generic reasons.
Maximum 50 words.

------------------------------------------------------------
3. TAG:
Choose exactly one:
[data_loading, preprocessing, feature_engineering, model_training, evaluation, visualization, utility, other]

------------------------------------------------------------
FORMAT EXACTLY:
WHAT: <text>
WHY: <text>
TAG: <single_word>

------------------------------------------------------------
STRICT RULES:
- Use only information visible in the provided code and variable lists.
- Do NOT assume missing steps.
- Do NOT invent variable names.
- If no executable code exists:
  WHAT: No Code present in this cell
  WHY: No Code present in this cell
  TAG: other
- If the cell only imports libraries, explicitly classify imported names as modules.
- If the cell defines a function or class, explicitly state that it defines (not executes) logic.
- If the cell instantiates a class, distinguish class vs instance.
- If the cell modifies an existing variable, state that it updates or transforms that variable.
"""


def explain_cell(cell: dict) -> dict:
    if cell["type"] != "code":
        cell["purpose"] = "Narrative or section heading"
        cell["explanation"] = cell["source"][:200] + "..." if len(cell["source"]) > 200 else cell["source"]
        cell["intent"] = "narrative"
        return cell

    code = cell["source"]
    used = ", ".join(cell.get("used", []))
    defined = ", ".join(cell.get("defined", []))
    called = ", ".join(cell.get("called_symbols", []))

    prompt = f"""
    Code:
    ```python
    {code}
    ```

    Variables used: {used}
    Variables defined: {defined}
    Called symbols: {called}
    """

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()

    # Parse structured response
    what_match = re.search(r'WHAT:\s*(.+)$', content, re.IGNORECASE | re.MULTILINE)
    why_match = re.search(r'WHY:\s*(.+)$', content, re.IGNORECASE | re.MULTILINE)
    tag_match = re.search(r'TAG:\s*(\w+)$', content, re.IGNORECASE | re.MULTILINE)

    if not (what_match and why_match and tag_match):
        cell["purpose"] = "Unclear code block"
        cell["explanation"] = "The model could not reliably extract the intent of this cell."
        cell["intent"] = "other"
        cell["explanation_error"] = True  # ← VERY IMPORTANT
        return cell

    cell["purpose"] = what_match.group(1).strip()
    cell["explanation"] = why_match.group(1).strip()
    cell["intent"] = tag_match.group(1).lower()
    cell["explanation_error"] = False

    return cell


##############################
# from parser import parse_notebook
# from analyzer import analyze_code_cell
#
# cells = parse_notebook("SVM Training and EDA.ipynb")
# cells = [analyze_code_cell(c) for c in cells]
# cells = [explain_cell(c) for c in cells]
#
# for c in cells:
#     if c["type"] == "code":
#         print("=" * 40)
#         print(f"Cell {c['id']}")
#         print("Purpose:", c["purpose"])
#         print("Explanation:", c["explanation"])
#         print("Intent:", c["intent"])
#         print("Called:", c.get("called_symbols", []))
