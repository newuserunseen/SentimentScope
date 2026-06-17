# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared text-cleaning utilities used by both train_model.py and app.py.
# ─────────────────────────────────────────────────────────────────────────────

import re
import string
import nltk


# ── Download NLTK resources (robust — catches both LookupError AND OSError) ──
def download_nltk_resources():
    """
    Safely download all required NLTK packages.
    Uses a broad except clause so that OSError (corrupt / partial files on
    Windows) is handled the same way as LookupError (missing files).
    """
    packages = [
        "stopwords",
        "punkt",
        "punkt_tab",
    ]
    for pkg in packages:
        try:
            # Force a fresh download — safe even if already present
            nltk.download(pkg, quiet=True, raise_on_error=False)
        except Exception:
            pass  # never crash the app over an NLTK data issue


download_nltk_resources()

# ── Load stopwords (with a safe fallback if NLTK data is still missing) ──────
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Minimal hardcoded fallback so the app never crashes
    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your",
        "yours","yourself","yourselves","he","him","his","himself","she","her",
        "hers","herself","it","its","itself","they","them","their","theirs",
        "themselves","what","which","who","whom","this","that","these","those",
        "am","is","are","was","were","be","been","being","have","has","had",
        "having","do","does","did","doing","a","an","the","and","but","if","or",
        "because","as","until","while","of","at","by","for","with","about",
        "against","between","into","through","during","before","after","above",
        "below","to","from","up","down","in","out","on","off","over","under",
        "again","further","then","once","here","there","when","where","why",
        "how","all","both","each","few","more","most","other","some","such",
        "only","own","same","so","than","too","very","s","t","can","will",
        "just","should","now","d","ll","m","o","re","ve","y","ain","ma",
    }

# Negation words — intentionally kept so "not good" ≠ "good"
KEEP_WORDS = {"not", "no", "nor", "never", "neither", "n't"}
STOP_WORDS -= KEEP_WORDS


# ── Core cleaning function ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Strip @ / # symbols (keep the word)
      5. Remove punctuation
      6. Remove digits
      7. Collapse whitespace
      8. Remove stopwords (negations kept)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[@#]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


# ── Label helpers ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: ("Negative 😞", "#F43F5E"),
    1: ("Neutral 😐",  "#FBBF24"),
    2: ("Positive 😊", "#22C55E"),
}

def label_to_display(label: int) -> tuple:
    """Return (display_name, hex_color) for a numeric label."""
    return LABEL_MAP.get(label, ("Unknown", "#6B7280"))
