"""Unit tests for deterministic Vietnamese NLP pipeline contracts."""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.nlp_preprocessing import (  # noqa: E402
    normalize_abbreviations,
    normalize_unicode,
    preprocess_vietnamese_text,
    remove_stopwords,
)


def test_unicode_normalization_is_stable_for_decomposed_vietnamese():
    source = unicodedata.normalize("NFD", "Đường giao thông")
    once = normalize_unicode(source)
    twice = normalize_unicode(once)
    assert once == "Đường giao thông"
    assert twice == once
    assert unicodedata.is_normalized("NFC", once)


def test_abbreviation_replacement_does_not_change_longer_words():
    result = normalize_abbreviations("Đg. Nguyễn Huệ bị hỏng, khoảng cách vẫn vậy")
    assert result.startswith("đường Nguyễn Huệ bị hỏng")
    assert "khoảng cách" in result


def test_stopword_removal_preserves_vietnamese_negation_and_damage_terms():
    result = remove_stopwords(
        "đường và đèn_đường không bị hỏng",
        stopwords={"và", "không", "bị", "hỏng"},
    )
    assert result == ("đường", "đèn_đường", "không", "bị", "hỏng")


def test_pipeline_exposes_each_stage_for_auditable_storage():
    def deterministic_tokenizer(text, format="text"):
        assert format == "text"
        return "đường này không thông thoáng"
    result = preprocess_vietnamese_text(
        "Đg. này ko thông thoáng",
        tokenizer=deterministic_tokenizer,
    )
    assert result.raw_cleaned_text == "Đg. này ko thông thoáng"
    assert result.standardized_text == "đường này không thông thoáng"
    assert result.normalized_text == "đường này không thông thoáng"
    assert result.segmented_text == "đường này không thông thoáng"
    assert result.processed_text == "đường không thông thoáng"
