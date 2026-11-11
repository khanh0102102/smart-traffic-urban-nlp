"""Conservative Vietnamese traffic-domain abbreviation normalization."""

import re

# Chỉ mở rộng các dạng viết tắt tương đối rõ nghĩa. Có thể bổ sung sau khi rà soát dữ liệu.
ABBREVIATIONS: dict[str, str] = {
    "ko": "không",
    "k0": "không",
    "kh": "không",
    "dc": "được",
    "đc": "được",
    "ntn": "như_thế_nào",
    "vs": "với",
    "ql": "quốc_lộ",
    "đt": "điện_tín",
}


def normalize_abbreviations(text: str) -> str:
    tokens = text.split()
    normalized = [ABBREVIATIONS.get(token, token) for token in tokens]
    return re.sub(r"\s+", " ", " ".join(normalized)).strip()
