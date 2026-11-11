"""Canonical 11-class label catalogue and conversion helpers."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Label:
    code: str
    name: str


LABELS: tuple[Label, ...] = (
    Label("R01", "Đường/cầu/hạ tầng hư hỏng"),
    Label("R02", "Điện lực/chiếu sáng"),
    Label("R03", "Cống/hố ga/thoát nước"),
    Label("R04", "Vỉa hè/lòng đường"),
    Label("R05", "Cây xanh/vật cản"),
    Label("R06", "Biển báo"),
    Label("R07", "Tổ chức giao thông"),
    Label("R08", "Đèn tín hiệu"),
    Label("R09", "Rác thải/vệ sinh môi trường"),
    Label("R10", "Đỗ xe cản trở"),
    Label("R11", "Khác"),
)

LABEL_CODES: tuple[str, ...] = tuple(item.code for item in LABELS)
LABEL_TO_NAME: dict[str, str] = {item.code: item.name for item in LABELS}


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFC", value).strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


_NAME_TO_CODE = {_fold(item.name): item.code for item in LABELS}


def normalize_label(value: str) -> str:
    """Return canonical code for either R01..R11 or the exact Vietnamese label name."""
    if value is None:
        raise ValueError("Label không được để trống.")
    candidate = value.strip().upper()
    if candidate in LABEL_TO_NAME:
        return candidate
    code = _NAME_TO_CODE.get(_fold(value))
    if code:
        return code
    raise ValueError(f"Nhãn không hợp lệ: {value!r}. Chỉ chấp nhận R01..R11 hoặc tên nhãn chuẩn.")


def label_name(code: str) -> str:
    try:
        return LABEL_TO_NAME[normalize_label(code)]
    except ValueError as exc:
        raise KeyError(str(exc)) from exc
