"""Build the text blob that gets sent to the LLM.

Takes a parsed paper dict (the output of XMLParser.parse) and stitches
together the sections relevant to the reportability rubric. Sections that
are missing or empty are skipped so we do not waste LLM context on blank
headers.

Pure data transformation — no I/O, no LLM. The extractor is responsible
for reading the parsed JSON off disk and passing the dict in.
"""


# Sections we feed the LLM, in order. Introduction gives cohort context;
# methods, results, and data_availability cover the six rubric fields.
# Discussion and conclusion are argumentative and low-signal for our rubric,
# so they are deliberately excluded to save tokens.
DEFAULT_SECTIONS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "data_availability",
)


def build_paper_text(
    parsed: dict,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
) -> str:
    """Return a single text blob for the LLM from a parsed paper dict.

    Args:
        parsed: Output of XMLParser.parse() (metadata + section keys).
        sections: Lowercase section keys to include, in order.

    Returns:
        Sections concatenated with clear headers. Empty sections are
        skipped. Returns an empty string if nothing usable is present.
    """
    parts: list[str] = []

    title = parsed.get("metadata", {}).get("title", "")
    if isinstance(title, str) and title.strip():
        parts.append(f"TITLE:\n{title.strip()}")

    for key in sections:
        value = parsed.get(key, "")
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.upper()}:\n{value.strip()}")

    return "\n\n".join(parts)
