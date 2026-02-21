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
You are a precise static code analyzer preparing documentation for a Semantic Vector Search Engine.

Task: Analyze the provided Python code cell and generate a highly detailed, keyword-rich explanation.

Your analysis must explicitly reference variable names, their roles, and the real-world domain of the data.

------------------------------------------------------------
WHAT:
Describe the EXACT operation, algorithm, or technique implemented.
- Name the specific library, function, or class used (e.g., Pandas read_csv, Scikit-learn SVC).
- Mention key variables defined in this cell.
- Identify whether variables act as input data, configuration parameters, or outputs.
- EXPLICITLY state the real-world concept or dataset these variables represent (e.g., "customer loan data", "image classification labels") based on the variable names or context.

WHY:
Explain the technical purpose of this cell in context.
- Use actual variable names from the code.
- Explain why these specific steps are necessary for the overall machine learning or data processing pipeline.
- Clarify how this operation affects later computation (e.g., "This prepares the data for XGBoost model training").
- Be descriptive and use rich technical synonyms to help a search engine find this text.

TAG:
Choose exactly one:
[data_loading, preprocessing, feature_engineering, model_training, evaluation, visualization, utility, other]

------------------------------------------------------------
FORMAT EXACTLY:
WHAT: 
<Multi-line detailed text>

WHY: 
<Multi-line detailed text>

TAG: <single_word>

------------------------------------------------------------
STRICT RULES:
- Use only information visible in the provided code and variable lists.
- Do NOT invent variable names, but DO infer the real-world domain (e.g., if a variable is `loan_amt`, mention "loan amount").
- If no executable code exists:
  WHAT: No Code present in this cell
  WHY: No Code present in this cell
  TAG: other
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
    what_match = re.search(r'WHAT:\s*(.*?)(?=\nWHY:|$)', content, re.IGNORECASE | re.DOTALL)
    why_match = re.search(r'WHY:\s*(.*?)(?=\nTAG:|$)', content, re.IGNORECASE | re.DOTALL)
    tag_match = re.search(r'TAG:\s*(\w+)', content, re.IGNORECASE)

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
