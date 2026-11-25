"""Composable Vietnamese text preprocessing for citizen incident reports.

Pipeline order: clean text -> Unicode NFC -> abbreviation expansion ->
word segmentation -> conservative stopword removal.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache

from src.data_processing.cleaning import clean_text, normalize_text

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABBREVIATIONS_PATH = ML_ROOT / "data" / "abbreviations_vi.json"
DEFAULT_STOPWORDS_PATH = ML_ROOT / "data" / "stopwords_vi.txt"

@dataclass(frozen=True)
class NLPPreprocessingResult:
    raw_cleaned_text: str
    standardized_text: str
    normalized_text: str
    segmented_text: str
    tokens: tuple[str, ...]
    processed_text: str


def normalize_unicode(value: object) -> str:
    """Return clean text in Unicode NFC, preserving Vietnamese diacritics."""
    return unicodedata.normalize("NFC", clean_text(value))


@lru_cache(maxsize=16)
def load_abbreviations(path: str | Path | None = None) -> dict[str, str]:
    source = Path(path) if path else DEFAULT_ABBREVIATIONS_PATH
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Abbreviation dictionary must be a JSON object of source: replacement pairs.")
    return {
        unicodedata.normalize("NFC", str(shortcut)).casefold().strip():
        unicodedata.normalize("NFC", str(expansion)).strip()
        for shortcut, expansion in payload.items()
        if str(shortcut).strip() and str(expansion).strip()
    }


def normalize_abbreviations(
    value: object,
    mapping: Mapping[str, str] | None = None,
    abbreviations_path: str | Path | None = None,
) -> str:
    """Expand whole-token Vietnamese chat and address abbreviations.

    Short forms are matched case-insensitively and never as substrings of
    longer words. Period-terminated address forms (Q., P., TP.) can be
    immediately followed by a street/ward number or place name.
    """
    text = normalize_unicode(value)
    replacements = dict(mapping) if mapping is not None else load_abbreviations(abbreviations_path)
    replacements = {
        unicodedata.normalize("NFC", str(key)).casefold().strip():
        unicodedata.normalize("NFC", str(val)).strip()
        for key, val in replacements.items()
        if str(key).strip() and str(val).strip()
    }
    for shortcut in sorted(replacements, key=len, reverse=True):
        escaped = re.escape(shortcut)
        if shortcut.endswith("."):
            pattern = re.compile(r"(?<!\w)" + escaped + r"(?=\s|\d|[A-Za-zÀ-ỹ]|$)", re.IGNORECASE)
        else:
            pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
        text = pattern.sub(replacements[shortcut], text)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


@lru_cache(maxsize=16)
def load_stopwords(path: str | Path | None = None) -> set[str]:
    """Load one stopword per line; blank lines and # comments are ignored."""
    source = Path(path) if path else DEFAULT_STOPWORDS_PATH
    words = set()
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        token = unicodedata.normalize("NFC", line).strip().casefold()
        if token and not token.startswith("#"):
            words.add(token)
    return words


def segment_words(
    value: object,
    tokenizer: Callable[..., object] | None = None,
) -> str:
    """Segment Vietnamese words with Underthesea and return whitespace tokens."""
    text = normalize_unicode(value)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if tokenizer is None:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency underthesea. Install with: "
                "python -m pip install -r ml/requirements.txt"
            ) from exc
        tokenizer = word_tokenize
    segmented = tokenizer(text, format="text")
    if isinstance(segmented, (list, tuple)):
        segmented = " ".join(str(token) for token in segmented)
    return re.sub(r"\s+", " ", str(segmented)).strip()


def remove_stopwords(
    segmented_text: str,
    stopwords: Iterable[str] | None = None,
    stopwords_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Remove exact segmented tokens; negation and incident terms stay intact."""
    blocked = set(stopwords) if stopwords is not None else load_stopwords(stopwords_path)
    blocked = {unicodedata.normalize("NFC", word).casefold() for word in blocked}
    protected = {"không", "chưa", "chẳng", "đừng", "bị", "được", "mất", "hỏng"}
    blocked.difference_update(protected)
    tokens = (
        unicodedata.normalize("NFC", token).casefold()
        for token in segmented_text.split()
    )
    return tuple(token for token in tokens if token and token not in blocked)


def preprocess_vietnamese_text(
    value: object,
    *,
    remove_stopwords_enabled: bool = True,
    abbreviations_path: str | Path | None = None,
    stopwords_path: str | Path | None = None,
    tokenizer: Callable[..., object] | None = None,
) -> NLPPreprocessingResult:
    """Run all NLP stages and preserve each intermediate representation."""
    cleaned = normalize_unicode(value)
    standardized = normalize_abbreviations(cleaned, abbreviations_path=abbreviations_path)
    segmented = segment_words(standardized, tokenizer=tokenizer)
    tokens = (
        remove_stopwords(segmented, stopwords_path=stopwords_path)
        if remove_stopwords_enabled
        else tuple(unicodedata.normalize("NFC", token).casefold() for token in segmented.split() if token)
    )
    return NLPPreprocessingResult(
        raw_cleaned_text=cleaned,
        standardized_text=standardized,
        normalized_text=normalize_text(standardized),
        segmented_text=segmented,
        tokens=tokens,
        processed_text=" ".join(tokens),
    )
