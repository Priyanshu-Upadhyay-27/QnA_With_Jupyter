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
You are a technical code analyzer.

Task: Analyze the provided code and respond with exactly THREE lines:

1. WHAT: Identify the SPECIFIC operation/algorithm/technique in the code
   - Name the exact library/function used (e.g., "GridSearchCV", "KMeans", "train_test_split")
   - Mention key parameters if relevant
   - Maximum 40 words

2. WHY: State the TECHNICAL reason this operation is needed
   - Focus on the prerequisite or purpose (e.g., "hyperparameter tuning", "avoid data leakage")
   - Be specific to the code's context
   - Maximum 40 words

3. TAG: One word category [data_loading, preprocessing, feature_engineering, model_training, evaluation, visualization, utility, other]

Format EXACTLY:
WHAT: [text]
WHY: [text]
TAG: [single_word]

Critical:
- Extract information FROM the code, don't assume patterns
- Different code types need different explanations
- Be precise, not generic
- If a block is empty or no code is present: WHAT: No Code present in this cell | WHY: No Code present in this cell | TAG: other
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
