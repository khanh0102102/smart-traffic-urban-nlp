"""Admin authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from stnlp.api.dependencies import require_admin
from stnlp.api.schemas import AdminLoginRequest, AdminTokenOut
from stnlp.auth.security import create_access_token, verify_password
from stnlp.database.models import AdminUser
from stnlp.database.session import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=AdminTokenOut)
def login(payload: AdminLoginRequest, db: Session = Depends(get_db)):
    user = (
        db.query(AdminUser)
        .filter(AdminUser.username == payload.username, AdminUser.is_active.is_(True))
        .first()
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không chính xác.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AdminTokenOut(
        access_token=create_access_token(user.username),
        token_type="bearer",
        expires_in_minutes=120,
    )


@router.get("/me")
def current_admin(username: str = Depends(require_admin)):
    return {"username": username, "role": "admin"}
