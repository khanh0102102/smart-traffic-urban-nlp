"""Rule-based label suggestions for human annotators.

The rules intentionally do not assign ground-truth labels. A missing or ambiguous
suggestion must be reviewed by a human.
"""

from __future__ import annotations

import re
import unicodedata

from stnlp.labels import LABEL_TO_NAME


KEYWORDS: dict[str, tuple[str, ...]] = {
    "R01": (
        "ổ gà", "nứt đường", "sụt lún", "mặt đường hư", "đường hư hỏng",
        "hỏng cầu", "mặt cầu", "lún đường", "vỡ mặt đường",
    ),
    "R02": (
        "đèn đường", "chiếu sáng", "cột điện", "mất điện", "dây điện",
        "đèn chiếu sáng", "điện lực",
    ),
    "R03": (
        "hố ga", "cống thoát nước", "thoát nước", "nắp cống", "cống nghẹt",
        "cống bị tắc", "ngập nước", "miệng cống",
    ),
    "R04": (
        "vỉa hè", "lòng đường", "lấn chiếm", "bong tróc vỉa hè",
        "lối đi bộ", "mặt hè",
    ),
    "R05": (
        "cây xanh", "cây đổ", "cành cây", "cây gãy", "vật cản",
        "cây ngã",
    ),
    "R06": (
        "biển báo", "biển chỉ dẫn", "biển báo hiệu", "biển báo giao thông",
    ),
    "R07": (
        "phân luồng", "tổ chức giao thông", "ùn tắc", "kẹt xe",
        "cấm đường", "điều tiết giao thông", "phân làn",
    ),
    "R08": (
        "đèn tín hiệu", "đèn giao thông", "đèn đỏ", "đèn xanh",
        "đèn vàng", "tín hiệu giao thông",
    ),
    "R09": (
        "rác thải", "rác tồn đọng", "vệ sinh môi trường", "xả rác",
        "mùi hôi", "rác sinh hoạt", "đổ rác",
    ),
    "R10": (
        "đỗ xe", "đậu xe", "dừng đỗ", "đậu ô tô", "đỗ ô tô",
        "đỗ xe trái phép", "đỗ xe sai quy định",
    ),
}


def _normalize_for_matching(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or "")).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def suggest_label(text: str) -> dict:
    """Return keyword matches and a review-only suggestion, not a model probability."""
    normalized = _normalize_for_matching(text)
    counts: dict[str, int] = {}
    matches: dict[str, list[str]] = {}

    for code, keywords in KEYWORDS.items():
        found = [phrase for phrase in keywords if _normalize_for_matching(phrase) in normalized]
        if found:
            counts[code] = len(found)
            matches[code] = found

    if not counts:
        return {
            "suggested_label": None,
            "suggested_label_name": None,
            "matched_keywords": [],
            "matched_categories": [],
            "suggestion_status": "no_keyword_match",
            "review_required": True,
            "note": "Không tìm thấy từ khóa quy tắc; cần người gán nhãn xem xét, không tự gán R11.",
        }

    top_score = max(counts.values())
    winners = [code for code, count in counts.items() if count == top_score]
    if len(winners) != 1:
        return {
            "suggested_label": None,
            "suggested_label_name": None,
            "matched_keywords": [phrase for group in matches.values() for phrase in group],
            "matched_categories": winners,
            "suggestion_status": "ambiguous",
            "review_required": True,
            "note": "Có nhiều nhóm cùng điểm từ khóa; cần người gán nhãn quyết định.",
        }

    winner = winners[0]
    return {
        "suggested_label": winner,
        "suggested_label_name": LABEL_TO_NAME[winner],
        "matched_keywords": matches[winner],
        "matched_categories": [winner],
        "suggestion_status": "suggested",
        "review_required": True,
        "note": "Đây chỉ là gợi ý theo từ khóa; phải kiểm duyệt thủ công trước khi dùng huấn luyện.",
    }
