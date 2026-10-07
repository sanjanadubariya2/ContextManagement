"""Context Manager integration tests (Postgres + pgvector + Redis).

Increment 2 gate: approved decisions are always pinned, the budget is never
exceeded, superseded chunks are never returned, and packages are traced.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from mpc import db
from mpc.context import artifacts, contracts, registry
from mpc.context.cache import load_package
from mpc.context.impact import analyze
from mpc.context.packager import (
    TicketExpired,
    build_package,
    check_staleness,
    retrieve_candidates,
)
from mpc.context.packing import TaskTooBig
from mpc.context.repo import current_version
from mpc.context.visibility import draft_ticket
from mpc.models import Chunk, ContextTrace, Decision, Member, Task
from mpc.redis_store import CTX_VERSION
from mpc.tokens import estimate_tokens

pytestmark = pytest.mark.db

REQUESTS = [
    ("sneha", "Implement the login API", "t20"),
    ("riya", "Build the login page with field validation", "t21"),
    ("kabir", "Write pytest cases for POST /api/auth/login", "t22"),
    ("aarav", "Add a remember-me checkbox so people stay signed in for a week", None),
    ("rohan", "Add the open-task counter to the dashboard summary endpoint", "t31"),
    ("tanvi", "Test the weather proxy's cache", "t32"),
    ("meera", "Make the colors nicer", None),
    ("ishaan", "zzz qqq unrelated words", None),
]


def _build(s, r, emb, member_id, text, task_id=None, budget=12000, **kw):
    m = s.get(Member, member_id)
    ticket = draft_ticket(m.id, m.team_id, task_id=task_id, token_budget=budget)
    return build_package(s, r, ticket, text, embedder=emb, **kw)


def test_seed_reaches_v14_and_redis_agrees(ws):
    r, _ = ws
    with db.session_scope() as s:
        assert current_version(s) == 14
        assert registry.get_approved(s, "auth.token_transport").value == "http_only_cookie"
    assert r.get(CTX_VERSION) == "14"


@pytest.mark.parametrize("member_id,text,task_id", REQUESTS)
def test_every_approved_decision_on_touched_keys_is_pinned(ws, member_id, text, task_id):
    r, emb = ws
    with db.session_scope() as s:
        pkg = _build(s, r, emb, member_id, text, task_id, use_cache=False)
        approved = registry.approved_map(s, pkg.impact.decision_keys)
        pinned = {i.ref for i in pkg.items if i.section == "pinned"}
        for d in approved.values():
            assert f"decision:{d.id}" in pinned, f"{d.key} not pinned"
        # Pinned decisions come by key and never compete in scoring.
        assert all(i.score is None for i in pkg.items if i.section == "pinned")


def test_trap_pins_cookie_decision_without_similarity(ws):
    """'remember-me' is not textually similar to the cookie decision, yet it must be pinned."""
    r, emb = ws
    with db.session_scope() as s:
        d = registry.get_approved(s, "auth.token_transport")
        pkg = _build(s, r, emb, "aarav", "Add a remember-me checkbox", use_cache=False)
        assert f"decision:{d.id}" in {i.ref for i in pkg.items if i.section == "pinned"}


@pytest.mark.parametrize("budget", [1600, 2500, 4000, 12000])
@pytest.mark.parametrize("member_id,text,task_id", REQUESTS[:4])
def test_budget_never_exceeded(ws, member_id, text, task_id, budget):
    r, emb = ws
    with db.session_scope() as s:
        try:
            pkg = _build(s, r, emb, member_id, text, task_id, budget=budget, use_cache=False)
        except TaskTooBig as e:
            assert e.pinned_tokens > budget
            return
        assert pkg.total_tokens <= budget
        assert estimate_tokens(pkg.render()) <= budget


def test_tiny_budget_is_task_too_big(ws):
    r, emb = ws
    with db.session_scope() as s, pytest.raises(TaskTooBig):
        _build(s, r, emb, "sneha", "Implement the login API", "t20", budget=300, use_cache=False)


def test_superseded_code_chunks_never_returned(ws):
    r, emb = ws
    path = "backend/app/auth/tokens.py"
    with db.session_scope() as s:
        old_ids = set(s.scalars(select(Chunk.id).where(Chunk.path == path, Chunk.status == "active")))
        old = artifacts.current(s, path)
        draft = artifacts.write_draft(
            s, path, old.content + "\n\ndef rotate_refresh_token(token: str) -> str:\n    return token\n",
            created_by="ishaan",
        )
        artifacts.promote(s, draft.id, emb, r=r)
    with db.session_scope() as s:
        assert artifacts.current(s, path).version == old.version + 1
        ticket = draft_ticket("ishaan", "backend")
        cands = retrieve_candidates(
            s, ticket, "refresh token rotation tokens", emb, exclude_decisions=set(), limit=200
        )
        ids = {int(c.item.ref.split(":")[1]) for c in cands}
        assert ids and not (ids & old_ids)
        statuses = set(s.scalars(select(Chunk.status).where(Chunk.id.in_(ids))))
        assert statuses == {"active"}
        assert any("rotate_refresh_token" in c.item.content for c in cands)


def test_superseded_decision_not_returned_and_new_value_pinned(ws):
    r, emb = ws
    with db.session_scope() as s:
        first = registry.propose(s, "dashboard.refresh_seconds", "30", proposed_by="aarav",
                                 team_id="frontend", rationale="Poll every half minute.", embedder=emb)
        registry.approve(s, first.id, approvers=["aarav", "meera"], embedder=emb, r=r)
    with db.session_scope() as s:
        second = registry.propose(s, "dashboard.refresh_seconds", 60, proposed_by="riya",
                                  team_id="frontend", rationale="Halve the load.", embedder=emb)
        v = registry.approve(s, second.id, approvers=["riya", "meera"], embedder=emb, r=r)
    with db.session_scope() as s:
        assert s.get(Decision, first.id).status == "superseded"
        assert s.get(Decision, first.id).superseded_in_version == v
        assert registry.snapshot(s, v - 1)["dashboard.refresh_seconds"] == 30
        assert registry.snapshot(s, v)["dashboard.refresh_seconds"] == 60
        active = s.scalars(select(Chunk).where(Chunk.decision_id == first.id, Chunk.status == "active"))
        assert list(active) == []
        pkg = _build(s, r, emb, "aarav", "Change the dashboard polling interval", use_cache=False)
        refs = {i.ref for i in pkg.items}
        assert f"decision:{second.id}" in refs and f"decision:{first.id}" not in refs


def test_approval_bumps_version_and_invalidates_cache(ws):
    r, emb = ws
    with db.session_scope() as s:
        pkg = _build(s, r, emb, "riya", "Restyle the login button")
        assert r.get(f"pkg:{pkg.package_id}") is not None
        hit = _build(s, r, emb, "riya", "Restyle the login button")
        assert hit.cache_hit and hit.package_id == pkg.package_id
        assert "ui.styling" in pkg.dependencies["decisions"]
        before = current_version(s)
    with db.session_scope() as s:
        d = registry.propose(s, "ui.styling", "css_modules", proposed_by="meera",
                             team_id="frontend", embedder=emb)
        registry.approve(s, d.id, approvers=["meera", "aarav"], embedder=emb, r=r)
    assert r.get(CTX_VERSION) == str(before + 1)
    assert load_package(r, pkg.package_id) is None
    with db.session_scope() as s:
        again = _build(s, r, emb, "riya", "Restyle the login button")
        assert not again.cache_hit and again.ctx_version == before + 1


def test_rolled_back_approval_does_not_touch_redis(ws):
    r, emb = ws
    before = r.get(CTX_VERSION)
    with pytest.raises(RuntimeError):
        with db.session_scope() as s:
            d = registry.propose(s, "integrations.cache_ttl_seconds", 600, proposed_by="ishaan",
                                 team_id="backend", embedder=emb)
            registry.approve(s, d.id, approvers=["ishaan", "rohan"], embedder=emb, r=r)
            raise RuntimeError("abort")
    assert r.get(CTX_VERSION) == before
    with db.session_scope() as s:
        assert registry.get_approved(s, "integrations.cache_ttl_seconds") is None


def test_staleness_detected_after_contract_change(ws):
    r, emb = ws
    with db.session_scope() as s:
        pkg = _build(s, r, emb, "sneha", "Implement the login API", "t20", use_cache=False)
        assert not check_staleness(s, pkg.dependencies).stale
        c = contracts.latest(s, "openapi/auth.yaml")
        contracts.publish(s, c.path, c.content + "\n# v2: name in login response\n",
                          created_by="rohan", r=r)
    with db.session_scope() as s:
        st = check_staleness(s, pkg.dependencies)
        assert st.stale and any("openapi/auth.yaml" in x for x in st.changes)
        assert st.current_version > st.packed_version


def test_testing_never_receives_discussion_or_proposals(ws):
    r, emb = ws
    with db.session_scope() as s:
        p = registry.propose(s, "auth.session_ttl_minutes", 60, proposed_by="sneha",
                             team_id="backend", rationale="Backend thinks 60 is friendlier.",
                             embedder=emb)
        for member, task in [("kabir", "t22"), ("tanvi", "t32"), ("dev", None)]:
            pkg = _build(s, r, emb, member, "login session refresh cookie tests", task,
                         use_cache=False)
            assert pkg.ticket.mode == "isolated"
            assert all(i.section != "live" for i in pkg.items)
            ids = [int(i.ref.split(":")[1]) for i in pkg.items if i.ref.startswith("chunk:")]
            for c in s.scalars(select(Chunk).where(Chunk.id.in_(ids))):
                assert c.kind not in ("thread_summary", "design_note", "discussion") or c.visibility == "testing"
                assert c.decision_id != p.id
        # The backend proposal is also hidden from the frontend, but visible to backend.
        fe = _build(s, r, emb, "riya", "Backend thinks 60 is friendlier session", use_cache=False)
        be = _build(s, r, emb, "rohan", "Backend thinks 60 is friendlier session", use_cache=False)
        rationale = s.scalar(select(Chunk.id).where(Chunk.decision_id == p.id, Chunk.status == "active"))
        assert f"chunk:{rationale}" not in {i.ref for i in fe.items}
        assert f"chunk:{rationale}" in {i.ref for i in be.items}
        registry.reject(s, p.id)


def test_contradiction_flagged(ws):
    with db.session_scope() as s:
        rep = analyze(s, "Return the access token in the response body", "backend")
        assert [c.key for c in rep.contradictions] == ["auth.token_transport"]
        clean = analyze(s, "Set the access token cookie on login", "backend")
        assert clean.contradictions == []


def test_trace_row_written(ws):
    r, emb = ws
    with db.session_scope() as s:
        pkg = _build(s, r, emb, "riya", "Build the login page", "t21", use_cache=False)
        row = s.get(ContextTrace, pkg.trace_id)
        assert row.total_tokens == pkg.total_tokens and row.ctx_version == pkg.ctx_version
        assert row.instance_id == "worker:fe-riya:t21"
        assert {i["ref"] for i in row.items} == {i.ref for i in pkg.items}
        assert row.dependencies["contracts"]["openapi/auth.yaml"] >= 1


def test_expired_ticket_refused(ws):
    r, emb = ws
    with db.session_scope() as s:
        t = draft_ticket("riya", "frontend")
        t.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        with pytest.raises(TicketExpired):
            build_package(s, r, t, "anything", embedder=emb)


def test_registry_types_and_lifecycle(ws):
    _, emb = ws
    with db.session_scope() as s:
        with pytest.raises(registry.RegistryError):
            registry.propose(s, "auth.token_transport", "carrier_pigeon", proposed_by="rohan", embedder=emb)
        with pytest.raises(registry.RegistryError):
            registry.propose(s, "auth.session_ttl_minutes", "soon", proposed_by="rohan", embedder=emb)
        with pytest.raises(registry.RegistryError):
            registry.propose(s, "no.such.key", 1, proposed_by="rohan", embedder=emb)
        approved = registry.get_approved(s, "auth.mechanism")
        with pytest.raises(registry.RegistryError):
            registry.approve(s, approved.id, approvers=["rohan"], embedder=emb)


def test_seed_task_and_secret_handling(ws):
    with db.session_scope() as s:
        assert s.get(Task, "t21").team_id == "frontend"
        env_chunks = s.scalars(select(Chunk).where(Chunk.path == "backend/.env"))
        assert list(env_chunks) == []
        leaked = s.scalars(select(Chunk).where(Chunk.content.contains("dev-only-change-me")))
        assert list(leaked) == []
