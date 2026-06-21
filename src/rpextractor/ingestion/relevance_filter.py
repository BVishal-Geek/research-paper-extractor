import difflib
from dataclasses import dataclass, field
from rpextractor.utils.logger import get_logger

logger = get_logger(__name__)

# Keywords organized by category — each hit adds to the relevance score
KEYWORD_CATEGORIES = {
    "dataset": [
        "dataset", "benchmark", "corpus", "training data", "test set",
        "validation set", "annotated data", "ground truth", "data collection",
        "gold standard", "labeled data",
    ],
    "model": [
        "machine learning", "deep learning", "neural network", "transformer",
        "language model", "classification", "regression", "clustering",
        "fine-tuning", "pre-trained", "bert", "gpt", "llm",
    ],
    "metric": [
        "accuracy", "f1", "precision", "recall", "auc", "roc",
        "bleu", "rouge", "perplexity", "mean absolute error", "rmse",
        "performance", "evaluation",
    ],
    "result": [
        "outperform", "state-of-the-art", "baseline", "sota",
        "improvement", "competitive", "superior", "significant",
    ],
}

# Minimum score thresholds
MIN_TOTAL_SCORE = 2        # must match at least 2 keywords overall
MIN_CATEGORY_HITS = 2     # must hit at least 2 distinct categories
FUZZY_THRESHOLD = 0.7    # SequenceMatcher ratio for near-match


@dataclass
class FilterResult:
    is_relevant: bool
    score: int
    categories_hit: list[str]
    matched_terms: list[str]
    pmcid: str = ""


def _fuzzy_match(word: str, keyword: str) -> bool:
    """Return True if word is a close enough match to keyword."""
    ratio = difflib.SequenceMatcher(None, word, keyword).ratio()
    return ratio >= FUZZY_THRESHOLD


def _score_text(text: str) -> tuple[int, list[str], list[str]]:
    """
    Score a block of text against all keyword categories.

    Returns (score, categories_hit, matched_terms).
    """
    if not text:
        return 0, [], []

    text_lower = text.lower()
    words = text_lower.split()

    score = 0
    categories_hit = []
    matched_terms = []

    for category, keywords in KEYWORD_CATEGORIES.items():
        category_matched = []

        for kw in keywords:
            # Exact substring match first (fast path)
            if kw in text_lower:
                category_matched.append(kw)
                continue

            # Token-level fuzzy match for single-word keywords only
            if " " not in kw:
                for word in words:
                    if _fuzzy_match(word, kw):
                        category_matched.append(f"~{kw}")
                        break

        if category_matched:
            score += len(category_matched)
            categories_hit.append(category)
            matched_terms.extend(category_matched)

    return score, categories_hit, matched_terms


def check_relevance(paper: dict) -> FilterResult:
    """
    Check whether a parsed paper JSON is relevant for ML entity extraction.

    Scores abstract + title + existing keywords. A paper is considered
    relevant if it crosses MIN_TOTAL_SCORE and hits MIN_CATEGORY_HITS
    distinct categories.

    Args:
        paper: Dict produced by XMLParser.parse()

    Returns:
        FilterResult with relevance decision and diagnostic info.
    """
    pmcid = paper.get("metadata", {}).get("pmcid", "unknown")

    # Build the text to score: abstract is primary, title + keywords supplement
    abstract = paper.get("abstract", "")
    title = paper.get("metadata", {}).get("title", "")
    existing_keywords = " ".join(paper.get("metadata", {}).get("keywords", []))

    combined_text = f"{title} {abstract} {existing_keywords}"

    score, categories_hit, matched_terms = _score_text(combined_text)

    is_relevant = score >= MIN_TOTAL_SCORE and len(categories_hit) >= MIN_CATEGORY_HITS

    result = FilterResult(
        is_relevant=is_relevant,
        score=score,
        categories_hit=categories_hit,
        matched_terms=matched_terms,
        pmcid=pmcid,
    )

    if is_relevant:
        logger.info(
            f"[{pmcid}] RELEVANT — score={score}, categories={categories_hit}, terms={matched_terms}"
        )
    else:
        logger.info(
            f"[{pmcid}] FILTERED OUT — score={score}, categories={categories_hit}"
        )

    return result


def filter_papers(papers: list[dict]) -> tuple[list[dict], list[FilterResult]]:
    """
    Filter a list of parsed papers, returning only relevant ones.

    Args:
        papers: List of dicts from XMLParser.parse()

    Returns:
        (relevant_papers, all_results) — filtered list and full diagnostic results.
    """
    results = [check_relevance(p) for p in papers if p is not None]
    relevant_papers = [
        paper for paper, result in zip(papers, results) if result.is_relevant
    ]

    total = len(results)
    kept = len(relevant_papers)
    logger.info(f"Relevance filter: {kept}/{total} papers passed ({total - kept} dropped)")

    return relevant_papers, results
