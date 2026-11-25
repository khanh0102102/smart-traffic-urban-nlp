"""Vietnamese-safe text cleaning and normalization."""
import html
import re
import unicodedata

URL_RE = re.compile(r"(?:https?://|www\\.)\\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b")
CONTROL_RE = re.compile(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]")
SPACE_RE = re.compile(r"\\s+")

def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\\u200b", " ").replace("\\ufeff", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text

def normalize_text(value: object) -> str:
    """Normalize for matching/deduplication without stripping Vietnamese accents."""
    text = clean_text(value).casefold()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\\w\\s]", " ", text, flags=re.UNICODE)
    return SPACE_RE.sub(" ", text).strip()
