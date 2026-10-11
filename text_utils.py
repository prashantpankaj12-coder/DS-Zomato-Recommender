"""Shared text helpers: the SAME cleaning code is used by the notebook (to build the model)
and by the Streamlit app (to clean what a user types), so both always agree."""
import re
from collections import Counter
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DOMAIN_STOP = {"rated", "zomato", "bangalore", "bengaluru", "restaurant", "place", "food", "ordered",
               "order", "went", "visited", "visit", "got", "also", "one", "really", "just", "like"}
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP
_stemmer = PorterStemmer()
stem = lru_cache(maxsize=None)(_stemmer.stem)
URL_RE, NON_ALPHA = re.compile(r"http\S+|www\.\S+"), re.compile(r"[^a-z\s]")


def tokens(text):
    """Lower-case, strip URLs / digits / punctuation, drop short words and stop-words (no stemming)."""
    text = NON_ALPHA.sub(" ", URL_RE.sub(" ", text.lower()))
    return [w for w in text.split() if len(w) > 2 and w not in STOP]


def clean_text(text):
    """Full pipeline used for the model: tokens + Porter stemming."""
    return " ".join(stem(w) for w in tokens(text))


def build_unstem(texts):
    """Map each stem to its most frequent readable word (e.g. 'beauti' -> 'beautiful')."""
    counts = {}
    for t in texts:
        for w in tokens(t):
            counts.setdefault(stem(w), Counter())[w] += 1
    return {s: c.most_common(1)[0][0] for s, c in counts.items()}


def readable(term, unstem):
    """Turn a stemmed unigram / bigram back into readable words for display."""
    return " ".join(unstem.get(p, p) for p in term.split())
