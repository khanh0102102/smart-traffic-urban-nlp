from stnlp.preprocessing import text as text_module


def test_clean_text_removes_html_url_and_normalizes():
    result = text_module.clean_text("<p>ĐƯỜNG bị hỏng!!!</p> https://example.com")
    assert "<p>" not in result
    assert "https" not in result
    assert "đường" in result
    assert "bị" in result


def test_preprocessing_keeps_negation(monkeypatch):
    # Replace external segmenter to make this unit test deterministic and fast.
    monkeypatch.setattr(text_module, "_load_segmenter", lambda: ("test", lambda value: value))
    result = text_module.preprocess_text("Đèn không sáng")
    assert "không" in result.split()
