from stnlp.data.label_suggestions import suggest_label


def test_clear_keyword_suggestion_still_requires_human_review():
    result = suggest_label("Hố ga trước nhà bị mất nắp")
    assert result["suggested_label"] == "R03"
    assert result["review_required"] is True
    assert result["suggestion_status"] == "suggested"


def test_no_keyword_does_not_automatically_become_other():
    result = suggest_label("Tôi cần phản ánh một vấn đề")
    assert result["suggested_label"] is None
    assert result["suggestion_status"] == "no_keyword_match"
    assert result["review_required"] is True


def test_ambiguous_keywords_are_sent_to_manual_review():
    result = suggest_label("Đèn đường và đèn tín hiệu tại ngã tư đều không hoạt động")
    assert result["suggested_label"] is None
    assert result["suggestion_status"] == "ambiguous"
