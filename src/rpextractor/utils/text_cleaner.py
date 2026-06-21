import re

def clean_text(text: str) -> str:
    """Clean extracted text by removing citations and extra whitespace."""
    # Remove [1], [2,3], [4-7] style citations
    text = re.sub(r'\[\d+(?:[,\-–\s]*\d+)*\]', '', text)
    
    # Remove (1), (2,3) style citations
    text = re.sub(r'\(\d+(?:[,\-–\s]*\d+)*\)', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()