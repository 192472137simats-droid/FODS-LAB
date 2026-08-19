"""
preprocess.py
-------------
Light text cleaning + keyword extraction for the idea text.
Pure Python + collections, in the spirit of the FODS text experiments.
"""

import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "that",
    "helps", "lets", "more", "through", "app", "platform", "tool", "service",
    "website", "marketplace", "using", "based", "powered", "driven", "all", "one",
    "is", "are", "it", "this", "their", "them", "they", "who", "which", "by",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str):
    return [t for t in clean_text(text).split() if len(t) > 2 and t not in STOPWORDS]


def extract_keywords(text: str, top_n: int = 6):
    """Return the most informative words from the idea, most frequent first."""
    counts = Counter(tokens(text))
    if not counts:
        return []
    # frequency first, then keep first-seen order for ties
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ordered[:top_n]]
