# final_object_builder.py
from explainer import clean_cell_output
def assign_sections(parsed_cells):
    """
    Attach section headers (markdown starting with #) to following cells.
    """
    current = "Untitled Section"
    for c in parsed_cells:
        if c["type"] == "markdown":
            first_line = c["source"].split('\n')[0].strip()
            if first_line.startswith('#'):
                current = first_line.lstrip('# ').strip()
        c["section"] = current
    return parsed_cells



def build_final_object(parsed, analyzed, explained):
    """
    Merge parsed, analyzed, and explained versions into one canonical object.
    """
    # Clean the raw outputs to prevent Base64 Image crashes
    safe_output = clean_cell_output(parsed.get("outputs", []))

    obj = {
        "cell_id": parsed["id"],
        "cell_type": parsed["type"],
        "cell_index": parsed.get("cell_index", -1),
        "exec_order": parsed.get("exec_order", None),
        "section": parsed.get("section"),
        "intent": explained.get("intent", "other"),
        "tags": parsed.get("metadata", {}).get("tags", []),
        "has_error": parsed.get("has_error", False),
        "used": analyzed.get("used", []),
        "defined": analyzed.get("defined", []),
        "called_symbols": analyzed.get("called_symbols", []),
        "purpose": explained.get("purpose", ""),
        "explanation": explained.get("explanation", ""),
        "source": parsed.get("source", ""),
        "result_summary": explained.get("result_summary", ""),

        # FIXED: Use the cleaned, truncated output string!
        "outputs": safe_output,

        "metadata": parsed.get("metadata", {}),
    }

    obj["dependency_score"] = len(obj["used"]) + len(obj["called_symbols"])

    return obj