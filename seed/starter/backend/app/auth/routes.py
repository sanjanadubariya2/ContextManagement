"""Auth endpoints. Tokens travel only in HttpOnly cookies."""

from fastapi import APIRouter, Cookie, HTTPException, Response

from app.auth.schemas import LoginRequest, UserOut
from app.auth.tokens import (
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    decode_token,
    set_auth_cookies,
)
from app.db.models import find_user_by_email, verify_password

router = APIRouter()


@router.post("/login", status_code=204)
def login(body: LoginRequest, response: Response) -> None:
    # TODO(t20): rate limiting, problem+json errors.
    user = find_user_by_email(body.email)
    if user is None or not verify_password(user, body.password):
        raise HTTPException(status_code=401, detail="Wrong email or password")
    set_auth_cookies(response, create_access_token(user.id), create_refresh_token(user.id))


@router.post("/logout", status_code=204)
def logout(response: Response) -> None:
    clear_auth_cookies(response)


@router.post("/refresh", status_code=204)
def refresh(response: Response, refresh_token: str | None = Cookie(default=None)) -> None:
    # TODO(t24): rotate refresh tokens.
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    claims = decode_token(refresh_token)
    set_auth_cookies(response, create_access_token(claims["sub"]), refresh_token)


@router.get("/me", response_model=UserOut)
def me(access_token: str | None = Cookie(default=None)) -> UserOut:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not logged in")
    claims = decode_token(access_token)
    return UserOut(id=claims["sub"], email=claims.get("email", "unknown@example.com"))
