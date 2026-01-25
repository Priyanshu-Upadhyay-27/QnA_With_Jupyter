# 4 : functions used to build the final custom deifned object
# 4: functions used to build the final custom defined object
def assign_sections(parsed_cells):
    """
    Attach section headers (markdown starting with #) to following cells.
    """
    current = None
    for c in parsed_cells:
        if c["type"] == "markdown":
            first_line = c["source"].split('\n')[0].strip()
            if first_line.startswith('#'):
                current = first_line.lstrip('# ').strip()  # IMPROVED: H1-H6, cleaner name
        c["section"] = current
    return parsed_cells


def build_final_object(parsed, analyzed, explained):
    """
    Merge parsed, analyzed, and explained versions into one canonical object.
    """
    obj = {
        "cell_id": parsed["id"],
        "cell_type": parsed["type"],
        "cell_index": parsed.get("cell_index", -1),  # NEW: position
        "exec_order": parsed.get("exec_order", None),  # FIXED: use .get()
        "section": parsed.get("section"),
        "intent": explained.get("intent", "other"),  # NEW: from explainer
        "tags": parsed.get("metadata", {}).get("tags", []),  # NEW: user tags
        "has_error": parsed.get("has_error", False),  # NEW: error flag
        "used": analyzed.get("used", []),
        "defined": analyzed.get("defined", []),
        "called_symbols": analyzed.get("called_symbols", []),  # NEW: from analyzer
        "purpose": explained.get("purpose", ""),
        "explanation": explained.get("explanation", ""),
        "source": parsed.get("source", ""),
        "outputs": parsed.get("outputs", []),
        "metadata": parsed.get("metadata", {}),
    }

    # NEW: Simple dependency hints
    obj["dependency_score"] = len(obj["used"]) + len(obj["called_symbols"])  # How much it needs upstream context

    return obj

