# 3: Used to write explanation for each code block
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

SYSTEM_PROMPT = """
You are analyzing a Jupyter notebook cell.

Explain:
1. What this cell does.
2. Why it exists in the notebook pipeline.

Use simple technical language.
Do not speculate beyond the code.
"""

def explain_cell(cell: dict) -> dict:
    if cell["type"] != "code":
        cell["purpose"] = "Narrative or section heading"
        cell["explanation"] = cell["source"][:200]
        return cell

    code = cell["source"]
    used = ", ".join(cell.get("used", []))
    defined = ", ".join(cell.get("defined", []))

    prompt = f"""
    Code:
    {code}
    
    Variables used: {used}
    Variables defined: {defined}
    """

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    cell["explanation"] = response.content.strip()
    cell["purpose"] = response.content.split(".")[0]

    return cell


##############################
from parser import parse_notebook
from analyzer import analyze_code_cell

cells = parse_notebook("SVM Training and EDA.ipynb")
cells = [analyze_code_cell(c) for c in cells]
cells = [explain_cell(c) for c in cells]

for c in cells:
    if c["type"] == "code":
        print("="*40)
        print(f"Cell {c['id']}")
        print("Purpose:", c["purpose"])
        print("Explanation:", c["explanation"])

