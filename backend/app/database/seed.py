"""Idempotently seed the fixed 11 incident categories."""
from sqlalchemy import select
from app.database.models import IncidentCategory
from app.database.session import SessionLocal

CATEGORIES = [
("R01","Đường/Cầu/Hạ tầng hư hỏng","duong-cau-ha-tang-hu-hong","Hư hỏng mặt đường, cầu và kết cấu hạ tầng.",1),
("R02","Điện lực/Chiếu sáng","dien-luc-chieu-sang","Sự cố điện, mất điện và hệ thống chiếu sáng.",2),
("R03","Cống/Hố ga/Thoát nước","cong-ho-ga-thoat-nuoc","Cống, hố ga, ngập úng và hệ thống thoát nước.",3),
("R04","Vỉa hè/Lòng đường","via-he-long-duong","Hư hỏng hoặc lấn chiếm vỉa hè và lòng đường.",4),
("R05","Cây xanh/Vật cản","cay-xanh-vat-can","Cây xanh gãy đổ, vật cản gây nguy hiểm.",5),
("R06","Biển báo","bien-bao","Biển báo giao thông hư hỏng, sai lệch hoặc bị che khuất.",6),
("R07","Tổ chức giao thông","to-chuc-giao-thong","Bất cập về tổ chức, phân luồng và điều tiết giao thông.",7),
("R08","Đèn tín hiệu","den-tin-hieu","Đèn tín hiệu giao thông không hoạt động hoặc hoạt động sai.",8),
("R09","Rác thải/Vệ sinh môi trường","rac-thai-ve-sinh-moi-truong","Rác thải và vấn đề vệ sinh môi trường đô thị.",9),
("R10","Đỗ xe cản trở","do-xe-can-tro","Phương tiện dừng, đỗ gây cản trở giao thông.",10),
("R11","Khác","khac","Các phản ánh chưa thuộc 10 nhóm còn lại.",11),
]
def seed_categories() -> int:
    """Insert missing categories and refresh descriptions by stable code."""
    inserted = 0
    with SessionLocal.begin() as session:
        existing = {row.code: row for row in session.scalars(select(IncidentCategory)).all()}
        for code, name, slug, description, order in CATEGORIES:
            row = existing.get(code)
            if row is None:
                session.add(IncidentCategory(code=code,name=name,slug=slug,description=description,sort_order=order))
                inserted += 1
            else:
                row.name, row.slug, row.description, row.sort_order, row.is_active = name, slug, description, order, True
    return inserted

if __name__ == "__main__":
    print(f"Seeded {seed_categories()} missing incident categories.")
