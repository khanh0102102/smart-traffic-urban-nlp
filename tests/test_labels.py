import pytest

from stnlp.labels import LABEL_CODES, LABEL_TO_NAME, normalize_label


def test_has_exactly_eleven_canonical_labels():
    assert len(LABEL_CODES) == 11
    assert LABEL_CODES[0] == "R01"
    assert LABEL_CODES[-1] == "R11"


def test_normalize_code_and_vietnamese_name():
    assert normalize_label("r03") == "R03"
    assert normalize_label(LABEL_TO_NAME["R03"]) == "R03"


def test_reject_unknown_label():
    with pytest.raises(ValueError):
        normalize_label("R99")
