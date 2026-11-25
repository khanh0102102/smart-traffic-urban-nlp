"""Unit tests for the reusable Vietnamese NLP preprocessing stages."""
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp_preprocessing import (
    normalize_unicode,
    normalize_abbreviations,
    segment_words,
    remove_stopwords,
    preprocess_vietnamese_text,
)


def test_unicode_is_normalized_to_nfc():
    decomposed = unicodedata.normalize("NFD", "Đường giao thông")
    normalized = normalize_unicode(decomposed)
    assert normalized == unicodedata.normalize("NFC", normalized)
    assert normalized == "Đường giao thông"


def test_abbreviations_expand_as_whole_words_and_address_forms():
    text = "TP.HCM, Q.1, P.5: Đg. này ko thông thoáng, mn ơi!"
    normalized = normalize_abbreviations(text)
    assert normalized == "thành phố HCM, quận 1, phường 5: đường này không thông thoáng, mọi người ơi!"
    assert normalize_abbreviations("khoảng cách") == "khoảng cách"


def test_word_segmentation_can_use_injected_tokenizer():
    def fake_tokenizer(text, format="text"):
        assert format == "text"
        return "đèn_đường bị hỏng"

    assert segment_words("Đèn đường bị hỏng!", tokenizer=fake_tokenizer) == "đèn_đường bị hỏng"


def test_stopwords_preserve_negation_and_incident_terms():
    tokens = remove_stopwords(
        "đường và đèn_đường không bị hỏng tại khu_vực",
        stopwords={"và", "không", "bị", "hỏng", "tại"},
    )
    assert tokens == ("đường", "đèn_đường", "không", "bị", "hỏng", "khu_vực")


def test_preprocessing_returns_all_pipeline_representations():
    def fake_tokenizer(text, format="text"):
        return "đường này không thông thoáng"

    result = preprocess_vietnamese_text(
        "Đg. này ko thông thoáng",
        tokenizer=fake_tokenizer,
    )
    assert result.raw_cleaned_text == "Đg. này ko thông thoáng"
    assert result.standardized_text == "đường này không thông thoáng"
    assert result.segmented_text == "đường này không thông thoáng"
    assert result.processed_text == "đường không thông thoáng"
    assert result.normalized_text == "đường này không thông thoáng"
