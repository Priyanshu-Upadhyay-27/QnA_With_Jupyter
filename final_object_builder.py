# 4 : functions used to build the final custom deifned object
def assign_sections(parsed_cells):
    """
    Attach section headers (markdown starting with #) to following cells.
    """
    current = None
    for c in parsed_cells:
        if c["type"] == "markdown" and c["source"].startswith("#"):
            current = c["source"].lstrip("#").strip()
        c["section"] = current
    return parsed_cells


def build_final_object(parsed, analyzed, explained):
    """
    Merge parsed, analyzed, and explained versions into one canonical object.
    """
    return {
        "cell_id": parsed["id"],
        "cell_type": parsed["type"],
        "exec_order": parsed["exec_order"],
        "section": parsed.get("section"),
        "used": analyzed.get("used", []),
        "defined": analyzed.get("defined", []),
        "purpose": explained.get("purpose", ""),
        "explanation": explained.get("explanation", ""),
        "source": parsed.get("source", ""),
        "outputs": parsed.get("outputs"),
        "metadata": parsed.get("metadata", {})
    }
