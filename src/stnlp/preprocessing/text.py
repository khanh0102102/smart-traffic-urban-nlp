"""Reusable text cleaning and Vietnamese word-segmentation pipeline."""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

from stnlp.preprocessing.dictionary import normalize_abbreviations
from stnlp.preprocessing.stopwords import remove_stopwords

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean markup/noise and normalize Vietnamese text to Unicode NFC."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFC", str(text))
    value = html.unescape(value)
    value = _HTML_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = value.casefold()
    value = _NON_WORD_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    value = normalize_abbreviations(value)
    return value


@lru_cache(maxsize=1)
def _load_segmenter():
    """Load Underthesea first; use PyVi only if Underthesea cannot be imported."""
    try:
        from underthesea import word_tokenize

        return ("underthesea", word_tokenize)
    except Exception as underthesea_error:
        try:
            from pyvi import ViTokenizer

            return ("pyvi", ViTokenizer.tokenize)
        except Exception as pyvi_error:
            raise RuntimeError(
                "Không tìm thấy bộ tách từ tiếng Việt. Hãy cài underthesea hoặc pyvi."
            ) from pyvi_error


def tokenize_vietnamese(text: str) -> list[str]:
    """Segment a cleaned Vietnamese sentence into tokens; phrases may use underscores."""
    if not text:
        return []
    name, tokenizer = _load_segmenter()
    if name == "underthesea":
        segmented = tokenizer(text, format="text")
    else:
        segmented = tokenizer(text)
    return [token for token in str(segmented).split() if token.strip()]


def preprocess_text(text: str, *, remove_stops: bool = True) -> str:
    """Return a single whitespace-separated, segmented string for vectorizers."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    tokens = tokenize_vietnamese(cleaned)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)
