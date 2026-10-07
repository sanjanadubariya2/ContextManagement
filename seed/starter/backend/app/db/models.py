"""Database schema and user lookups. Passwords are hashed with argon2."""

from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.db.session import get_connection

_hasher = PasswordHasher()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    owner_id TEXT REFERENCES users(id),
    name TEXT NOT NULL
);
"""


@dataclass
class User:
    id: str
    email: str
    password_hash: str


def find_user_by_email(email: str) -> User | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
        ).fetchone()
    return User(*row) if row else None


def verify_password(user: User, password: str) -> bool:
    try:
        return _hasher.verify(user.password_hash, password)
    except VerifyMismatchError:
        return False
