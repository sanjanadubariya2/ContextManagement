"""Dashboard endpoints."""

from fastapi import APIRouter, Cookie, HTTPException

from app.auth.tokens import decode_token
from app.db.session import get_connection

router = APIRouter()


@router.get("/summary")
def summary(access_token: str | None = Cookie(default=None)) -> dict:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not logged in")
    user_id = decode_token(access_token)["sub"]
    with get_connection() as conn:
        projects = conn.execute(
            "SELECT count(*) FROM projects WHERE owner_id = %s", (user_id,)
        ).fetchone()[0]
    # TODO(t31): open task counter and recent activity.
    return {"projects": projects, "open_tasks": 0, "recent": []}
