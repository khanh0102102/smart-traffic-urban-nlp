"""Canonical incident taxonomy shared with the Phase 3 database seed."""
from dataclasses import dataclass
import re
import unicodedata

@dataclass(frozen=True)
class Category:
    code: str
    name: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]

CATEGORIES = (
 Category("R01","Đường/Cầu/Hạ tầng hư hỏng",("r01","duong cau ha tang hu hong","duong hu hong","mat duong","o ga","cau hu hong"),("mặt đường","ổ gà","sụt lún","nứt đường","hư hỏng cầu","lún đường","nứt cầu")),
 Category("R02","Điện lực/Chiếu sáng",("r02","dien luc chieu sang","mat dien","den duong"),("mất điện","cúp điện","đèn đường","chiếu sáng","dây điện","trụ điện","điện lực")),
 Category("R03","Cống/Hố ga/Thoát nước",("r03","cong ho ga thoat nuoc","ngap nuoc","thoat nuoc"),("ngập nước","ngập úng","thoát nước","cống nghẹt","hố ga","nắp cống","nước tràn")),
 Category("R04","Vỉa hè/Lòng đường",("r04","via he long duong","lan chiem via he"),("vỉa hè","lòng đường","lấn chiếm","chiếm dụng vỉa hè","bày bán trên vỉa hè")),
 Category("R05","Cây xanh/Vật cản",("r05","cay xanh vat can","cay nga"),("cây xanh","cây gãy","cây đổ","cành cây","vật cản","chướng ngại vật")),
 Category("R06","Biển báo",("r06","bien bao","bien bao giao thong"),("biển báo","biển chỉ dẫn","biển báo giao thông","biển báo hỏng","biển báo bị che")),
 Category("R07","Tổ chức giao thông",("r07","to chuc giao thong","phan luong giao thong"),("phân luồng","tổ chức giao thông","ùn tắc","kẹt xe","điều tiết giao thông","đường một chiều")),
 Category("R08","Đèn tín hiệu",("r08","den tin hieu","den giao thong"),("đèn tín hiệu","đèn giao thông","đèn đỏ","đèn xanh","đèn vàng","đèn tín hiệu hỏng")),
 Category("R09","Rác thải/Vệ sinh môi trường",("r09","rac thai ve sinh moi truong","rac thai"),("rác thải","đổ rác","rác sinh hoạt","ô nhiễm","mùi hôi","vệ sinh môi trường")),
 Category("R10","Đỗ xe cản trở",("r10","do xe can tro","do xe trai phep"),("đỗ xe","dừng đỗ","đậu xe","đỗ xe trái phép","xe chắn lối","đỗ xe cản trở")),
 Category("R11","Khác",("r11","khac"),()),
)
def _key(value: object) -> str:
    text=unicodedata.normalize("NFKD",str(value)).casefold()
    text="".join(ch for ch in text if not unicodedata.combining(ch))
    text=re.sub(r"[^a-z0-9]+"," ",text)
    return " ".join(text.split())

def canonical_label(value: object) -> tuple[str | None, str | None]:
    """Resolve an explicit source label to a canonical code/name."""
    if value is None:
        return None,None
    key=_key(value)
    for category in CATEGORIES:
        if key in {_key(x) for x in (category.code,category.name,*category.aliases)}:
            return category.code,category.name
    return None,None

def keyword_label(text: str) -> tuple[str | None, str | None, str | None]:
    """Return a weak label only when exactly one category has keyword hits."""
    folded=text.casefold()
    hits=[c for c in CATEGORIES if c.keywords and any(k in folded for k in c.keywords)]
    if len(hits)==1:
        return hits[0].code,hits[0].name,"keyword_rule"
    return None,None,None
