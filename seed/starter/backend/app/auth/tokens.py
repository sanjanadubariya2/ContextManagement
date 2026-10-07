"""JWT creation and validation, and the cookie helpers that carry tokens."""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Response

ACCESS_TTL = timedelta(minutes=30)
REFRESH_TTL = timedelta(days=7)
ALGORITHM = "HS256"


def _secret() -> str:
    return os.environ["JWT_SECRET"]


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + ACCESS_TTL}, _secret(), ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "typ": "refresh", "exp": now + REFRESH_TTL}
    return jwt.encode(payload, _secret(), ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=[ALGORITHM])


def set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    opts = {"httponly": True, "secure": True, "samesite": "lax", "path": "/"}
    response.set_cookie("access_token", access, max_age=int(ACCESS_TTL.total_seconds()), **opts)
    response.set_cookie("refresh_token", refresh, max_age=int(REFRESH_TTL.total_seconds()), **opts)


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
