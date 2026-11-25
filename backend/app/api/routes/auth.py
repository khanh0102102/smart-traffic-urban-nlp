"""Citizen Google sign-in and staff password login API routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, require_roles
from app.core.auth_service import authenticate_google_citizen, authenticate_staff
from app.core.security import create_access_token
from app.database.models import User
from app.database.session import get_db
from app.schemas.auth import AuthTokenResponse, GoogleLoginRequest, StaffLoginRequest, UserPublic

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _token_response(user: User) -> AuthTokenResponse:
    token, expires_in = create_access_token(user.id, user.role)
    return AuthTokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserPublic.model_validate(user),
    )


@router.post("/google", response_model=AuthTokenResponse)
def google_login(payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    """Verify Google ID token and sign in/create a CITIZEN account."""
    return _token_response(authenticate_google_citizen(db, payload.id_token))


@router.post("/staff/login", response_model=AuthTokenResponse)
def staff_login(payload: StaffLoginRequest, db: Session = Depends(get_db)):
    """Sign in a pre-provisioned ADMIN or RECEIVER account."""
    return _token_response(authenticate_staff(db, payload.email, payload.password))


@router.get("/me", response_model=UserPublic)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/staff/me", response_model=UserPublic)
def get_staff_me(current_user: User = Depends(require_roles("ADMIN", "RECEIVER"))):
    return current_user
