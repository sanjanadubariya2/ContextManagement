"""Parsing, import resolution, secret redaction, visibility and embeddings."""

from mpc.context.parsing import parse_file, resolve_imports, split_large, ParsedChunk
from mpc.context.secrets import REDACTED, never_index, redact
from mpc.context.visibility import TEAM_GRANTS, chunk_allowed, draft_ticket
from mpc.llm.embeddings import HashEmbedder, cosine
from mpc.tokens import estimate_tokens

PY = '''"""Auth routes."""
from fastapi import APIRouter
from app.auth.tokens import decode_token
from .schemas import LoginRequest

router = APIRouter()

@router.post("/login")
def login(body: LoginRequest):
    return decode_token("x")

class Thing:
    pass
'''

TSX = """// Login page.
import { useState } from "react";
import { login } from "../api/auth";
import { ApiError } from "@/api/client";

export interface Props { next: string }

export default function LoginPage(props: Props) {
  const [email, setEmail] = useState("");
  return <form onSubmit={() => login(email, "pw")}>{email}</form>;
}

const helper = () => 1;
"""


def test_python_symbols_and_imports():
    p = parse_file("backend/app/auth/routes.py", PY)
    assert p.doc == "Auth routes."
    assert [c.symbol for c in p.chunks] == ["<module>", "login", "Thing"]
    assert "@router.post" in p.chunks[1].content  # decorators stay with the function
    known = {"backend/app/auth/tokens.py", "backend/app/auth/schemas.py"}
    assert resolve_imports("backend/app/auth/routes.py", "python", p.imports, known) == [
        "backend/app/auth/tokens.py",
        "backend/app/auth/schemas.py",
    ]


def test_tests_resolve_backend_package():
    known = {"backend/app/main.py", "tests/backend/test_auth.py"}
    assert resolve_imports("tests/backend/test_auth.py", "python", ["app.main"], known) == [
        "backend/app/main.py"
    ]


def test_tsx_components_via_tree_sitter():
    p = parse_file("frontend/src/pages/Login.tsx", TSX)
    assert p.language == "tsx"
    assert p.symbols == ["Props", "LoginPage", "helper"]
    assert p.imports == ["react", "../api/auth", "@/api/client"]
    known = {"frontend/src/api/auth.ts", "frontend/src/api/client.ts"}
    assert resolve_imports("frontend/src/pages/Login.tsx", "tsx", p.imports, known) == [
        "frontend/src/api/auth.ts",
        "frontend/src/api/client.ts",
    ]


def test_large_chunks_are_split_under_cap():
    big = ParsedChunk("huge", ("line of code\n" * 2000) + "x" * 10000)
    parts = split_large(big, 600)
    assert len(parts) > 1 and all(estimate_tokens(p.content) <= 600 for p in parts)


def test_secret_files_never_indexed():
    assert never_index("backend/.env")
    assert never_index("deploy/server.pem")
    assert not never_index("backend/.env.example")
    assert not never_index("backend/app/main.py")


def test_secrets_redacted_but_env_lookups_kept():
    text = (
        'JWT_SECRET = "super-secret-value-123"\n'
        "key = os.environ['JWT_SECRET']\n"
        "anthropic = 'sk-ant-api03-abcdefghijklmnop'\n"
    )
    out, n = redact(text)
    assert "super-secret-value-123" not in out and "sk-ant-api03" not in out
    assert "os.environ['JWT_SECRET']" in out
    assert n == 2 and out.count(REDACTED) == 2


def test_visibility_policy():
    fe, te = TEAM_GRANTS["frontend"], TEAM_GRANTS["testing"]
    assert chunk_allowed("frontend", fe, "frontend", "thread_summary", None)
    assert not chunk_allowed("frontend", fe, "backend", "thread_summary", None)
    assert not chunk_allowed("frontend", fe, "backend", "code", None)
    assert chunk_allowed("testing", te, "backend", "code", None)
    assert chunk_allowed("testing", te, "shared", "requirement", None)
    assert not chunk_allowed("testing", te, "backend", "thread_summary", None)
    assert not chunk_allowed("testing", te, "frontend", "design_note", None)
    assert not chunk_allowed("testing", te, "shared", "decision_rationale", "proposal")
    assert chunk_allowed("testing", te, "shared", "decision_rationale", "approved")


def test_draft_ticket_modes():
    assert draft_ticket("kabir", "testing").mode == "isolated"
    t = draft_ticket("riya", "frontend", task_id="t21")
    assert t.mode == "curated_fork" and t.instance_id == "worker:fe-riya:t21"
    assert t.write_paths == ["frontend/**"] and not t.admitted and not t.expired()


def test_hash_embedder_is_deterministic_and_lexical():
    e = HashEmbedder()
    a, b, c = e.embed(["refresh token rotation", "rotate the refreshToken", "tailwind colors"])
    assert e.embed(["refresh token rotation"])[0] == a
    assert abs(cosine(a, a) - 1) < 1e-9
    assert cosine(a, b) > cosine(a, c)
