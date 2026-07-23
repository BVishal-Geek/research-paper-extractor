"""Load a list of PMCIDs from a plain-text file.

Accepts any whitespace layout — one PMCID per line, several per line
separated by spaces, or a mix. Empty tokens and non-numeric junk (comment
lines, headers) are silently dropped. Duplicates are removed while preserving
order.

Normalizes to canonical PMCID form: 'PMC<digits>'. Both 'PMC12345' and bare
'12345' (any case) are accepted on input; both come out as 'PMC12345'. This
keeps downstream filenames aligned with what appears inside PubMed XML
(<article-id pub-id-type="pmcid">PMC12345</article-id>) and with what a
ground-truth CSV built from PubMed identifiers will contain.
"""

from pathlib import Path


def load_pmcids(path: str | Path) -> list[str]:
    """Read PMCIDs from a whitespace-separated text file.

    Args:
        path: Path to a text file. Any whitespace (space, tab, newline)
            separates tokens.

    Returns:
        List of canonical 'PMC<digits>' strings, de-duplicated, in the
        order first seen. Empty list if the file has no valid PMCIDs.
    """
    text = Path(path).read_text(encoding="utf-8")

    seen: set[str] = set()
    result: list[str] = []
    for token in text.split():
        pmcid = _normalize(token)
        if pmcid and pmcid not in seen:
            seen.add(pmcid)
            result.append(pmcid)
    return result


def _normalize(token: str) -> str:
    """Return canonical 'PMC<digits>' or empty string if the token is junk."""
    token = token.strip()
    if not token:
        return ""

    # Strip optional 'PMC' prefix (any case). Keep only what's after it.
    numeric = token[3:] if token.upper().startswith("PMC") else token

    if not numeric.isdigit():
        # Not a PMCID at all — probably a header, comment, or stray word.
        return ""
    return f"PMC{numeric}"
