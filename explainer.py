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


# 1. NEW: The Output Cleaner is now a helper function inside the explainer
def clean_cell_output(outputs: list, max_len: int = 500) -> str:
    """Extracts plain text and errors from Jupyter outputs, ignoring images to protect the LLM."""
    if not outputs:
        return "No output."

    cleaned_text = ""
    for out in outputs:
        out_type = out.get("output_type", "")
        if out_type == "stream":
            cleaned_text += out.get("text", "") + "\n"
        elif out_type in ["execute_result", "display_data"]:
            data = out.get("data", {})
            if "text/plain" in data:
                cleaned_text += data["text/plain"] + "\n"
        elif out_type == "error":
            cleaned_text += f"ERROR: {out.get('ename', '')}: {out.get('evalue', '')}\n"

    return cleaned_text[:max_len].strip() if cleaned_text else "[Output contained non-text data/image]"


# 2. UPDATED: The System Prompt now forces the ACR (Action-Context-Result) format
SYSTEM_PROMPT = """
You are a precise static code analyzer preparing documentation for a Semantic Vector Search Engine.

Task: Analyze the provided Python code cell and its output, then generate a highly detailed, keyword-rich explanation using the Action-Context-Result framework.

Your analysis must explicitly reference variable names, their roles, and the real-world domain of the data.

------------------------------------------------------------
WHAT:
Describe the EXACT operation, algorithm, or technique implemented.
- Name the specific library, function, or class used.
- Mention key variables defined in this cell.
- EXPLICITLY state the real-world concept or dataset these variables represent based on the context.

WHY:
Explain the technical purpose of this cell in context.
- Explain why these specific steps are necessary for the overall machine learning or data processing pipeline.
- Clarify how this operation affects later computation.

RESULT:
Analyze the provided Cell Output.
- If the output is an accuracy score, metric, or statistical result, state it explicitly.
- If it's a dataset summary (e.g., shape, head), mention what the data looks like.
- If it's an error, explain what failed.
- If there is no output, state "No significant output."

TAG:
Choose exactly one:
[data_loading, preprocessing, feature_engineering, model_training, evaluation, visualization, utility, other]

------------------------------------------------------------
FORMAT EXACTLY:
WHAT: 
<Multi-line detailed text>

WHY: 
<Multi-line detailed text>

RESULT:
<Multi-line detailed text>

TAG: <single_word>

------------------------------------------------------------
STRICT RULES:
- Use only information visible in the provided code, variable lists, and output.
- If no executable code exists:
  WHAT: No Code present in this cell
  WHY: No Code present in this cell
  RESULT: No output
  TAG: other
"""


def explain_cell(cell: dict) -> dict:
    if cell["type"] != "code":
        cell["purpose"] = "Narrative or section heading"
        cell["explanation"] = cell["source"][:200] + "..." if len(cell["source"]) > 200 else cell["source"]
        cell["result_summary"] = "None"
        cell["intent"] = "narrative"
        return cell

    code = cell.get("source", "")
    used = ", ".join(cell.get("used", []))
    defined = ", ".join(cell.get("defined", []))
    called = ", ".join(cell.get("called_symbols", []))

    # 3. Clean the output right before passing it to the LLM
    safe_output = clean_cell_output(cell.get("outputs", []))

    # 4. UPDATED: Inject the safe_output into the Human Message
    prompt = f"""
    Code:
    ```python
    {code}
    ```

    Variables used: {used}
    Variables defined: {defined}
    Called symbols: {called}

    Cell Output (Cleaned):
    {safe_output}
    """

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()

    # 5. UPDATED: Regex parser now hunts for the RESULT block too
    what_match = re.search(r'WHAT:\s*(.*?)(?=\nWHY:|$)', content, re.IGNORECASE | re.DOTALL)
    why_match = re.search(r'WHY:\s*(.*?)(?=\nRESULT:|$)', content, re.IGNORECASE | re.DOTALL)
    result_match = re.search(r'RESULT:\s*(.*?)(?=\nTAG:|$)', content, re.IGNORECASE | re.DOTALL)
    tag_match = re.search(r'TAG:\s*(\w+)', content, re.IGNORECASE)

    if not (what_match and why_match and result_match and tag_match):
        cell["purpose"] = "Unclear code block"
        cell["explanation"] = "The model could not reliably extract the intent of this cell."
        cell["result_summary"] = "Parsing failed."
        cell["intent"] = "other"
        cell["explanation_error"] = True
        return cell

    cell["purpose"] = what_match.group(1).strip()
    cell["explanation"] = why_match.group(1).strip()
    cell["result_summary"] = result_match.group(1).strip()  # NEW: Save the summarized result
    cell["intent"] = tag_match.group(1).lower()
    cell["explanation_error"] = False

    return cell