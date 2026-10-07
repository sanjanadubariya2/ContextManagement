"""Backend tests for the auth API, written from openapi/auth.yaml."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_login_rejects_bad_credentials():
    res = client.post("/api/auth/login", json={"email": "a@example.com", "password": "wrongpass"})
    assert res.status_code == 401


def test_me_requires_cookie():
    assert client.get("/api/auth/me").status_code == 401
