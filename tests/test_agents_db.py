"""Increment 3 integration tests: personal agents, Worker and Reviewer
instances, write-back and the sandboxed test runner.

Gate: Sneha's scripted login-API request runs end to end and promotes its
files to a new version.
"""

import pytest
from sqlalchemy import select

from mpc import db
from mpc.agents.personal import run_turn
from mpc.agents.runtime import Adapters, load_script, run_review, run_worker
from mpc.config import SEED_DIR
from mpc.context import artifacts, registry
from mpc.llm.fake import ScriptedAdapter
from mpc.models import Artifact, Chunk, ContextTrace, InstanceRun, Member, Message, Review, ReviewComment, Task, TestRun
from mpc.testing.runner import run_tests

pytestmark = pytest.mark.db


def _member(s, mid):
    return s.get(Member, mid)


def _adapters(**steps):
    return Adapters(by_role={role: ScriptedAdapter(st, role) for role, st in steps.items()})


def test_gate_sneha_login_api_end_to_end(ws):
    r, emb = ws
    script = load_script(str(SEED_DIR / "scripts" / "sneha_login_api.yaml"))
    with db.session_scope() as s:
        before = artifacts.current_versions(s, ["backend/app/main.py", "backend/app/auth/routes.py"])
        msgs_before = s.scalar(select(Message.id).order_by(Message.id.desc()).limit(1))
        turn = run_turn(s, r, emb, Adapters.from_script(script), _member(s, "sneha"),
                        script["request"], task_id="t20")
    res = turn.result
    assert turn.action == "delegate" and res.status == "completed"
    assert res.instance_id == "worker:be-sneha:t20"
    with db.session_scope() as s:
        now = artifacts.current_versions(s, list(res.files))
        assert now == res.files
        for p, v in before.items():
            assert now[p] == v + 1, f"{p} not promoted to a new version"
        assert now["backend/app/errors.py"] == 1
        # Re-embedded: new chunks are live, the old version's are superseded.
        live = list(s.scalars(select(Chunk).where(Chunk.path == "backend/app/main.py", Chunk.status == "active")))
        assert any("install_error_handlers" in c.content for c in live)
        old_ids = [a.id for a in s.scalars(select(Artifact).where(
            Artifact.path == "backend/app/main.py", Artifact.version == before["backend/app/main.py"]))]
        assert all(c.status == "superseded" for c in s.scalars(select(Chunk).where(Chunk.artifact_id.in_(old_ids))))
        run = s.get(InstanceRun, res.run_id)
        assert run.trace_id and s.get(ContextTrace, run.trace_id).instance_id == "worker:be-sneha:t20"
        assert run.steps == 7 and s.get(Task, "t20").status == "done"
        new_msgs = list(s.scalars(select(Message).where(Message.id > msgs_before).order_by(Message.id)))
        assert [m.role for m in new_msgs] == ["user", "assistant"] and "Promoted" in new_msgs[1].content


def test_worker_cannot_write_outside_its_paths_or_secrets(ws):
    r, emb = ws
    steps = [
        {"tool": "write_file", "input": {"path": "backend/app/main.py", "content": "x = 1\n"}},
        {"tool": "read_file", "input": {"path": "backend/.env"}},
        {"tool": "write_file", "input": {"path": "../escape.py", "content": "x = 1\n"}},
        {"tool": "finish", "input": {"summary": "nothing I was allowed to do"}},
    ]
    with db.session_scope() as s:
        res = run_worker(s, r, emb, _adapters(worker=steps), _member(s, "meera"), instructions="try it")
        assert res.status == "completed" and res.files == {}
        drafts = s.scalars(select(Artifact).where(Artifact.instance_id == res.instance_id))
        assert list(drafts) == []


def test_invalid_output_loops_back_until_fixed(ws):
    r, emb = ws
    path = "frontend/src/components/Spinner.tsx"
    steps = [
        {"tool": "write_file", "input": {"path": path, "content": "export default function Spinner( { return <div/>; }"}},
        {"tool": "finish", "input": {"summary": "spinner"}},
        {"tool": "write_file", "input": {"path": path, "content": "export default function Spinner() {\n  return <div className=\"animate-spin\" />;\n}\n"}},
        {"tool": "finish", "input": {"summary": "spinner, fixed"}},
    ]
    with db.session_scope() as s:
        res = run_worker(s, r, emb, _adapters(worker=steps), _member(s, "aarav"), instructions="Add a spinner component")
        assert res.status == "completed" and res.files == {path: 1} and res.steps == 4
        assert "animate-spin" in artifacts.current(s, path).content


class _Interfere:
    """Wraps an adapter; before its Nth call, approves a decision in another
    transaction, as a teammate's vote landing mid-run would."""

    def __init__(self, inner, at_call, key, value, emb, r):
        self.inner, self.at, self.key, self.value, self.emb, self.r = inner, at_call, key, value, emb, r
        self.provider, self.model, self.n = inner.provider, inner.model, 0

    def tool_call(self, messages, tools, **kw):
        self.n += 1
        if self.n == self.at:
            with db.session_scope() as other:
                d = registry.propose(other, self.key, self.value, proposed_by="riya", team_id="frontend", embedder=self.emb)
                registry.approve(other, d.id, approvers=["riya", "meera"], embedder=self.emb, r=self.r)
        return self.inner.tool_call(messages, tools, **kw)


def test_stale_result_is_discarded_and_rerun_on_new_version(ws):
    r, emb = ws
    path = "frontend/src/pages/Dashboard.tsx"
    one_attempt = [
        {"tool": "read_file", "input": {"path": path}},
        {"tool": "write_file", "input": {"path": path, "content": "export default function DashboardPage() {\n  return <p>Dashboard</p>;\n}\n"}},
        {"tool": "finish", "input": {"summary": "poll the dashboard"}},
    ]
    inner = ScriptedAdapter(one_attempt + one_attempt, "worker")
    adapters = Adapters(by_role={"worker": _Interfere(inner, 3, "dashboard.refresh_seconds", 45, emb, r)})
    with db.session_scope() as s:
        res = run_worker(s, r, emb, adapters, _member(s, "riya"), instructions="Change the dashboard polling interval")
        assert res.attempt == 2 and res.status == "completed"
        first = res.previous[0]
        assert first.status == "stale" and "dashboard.refresh_seconds" in first.error
        assert res.ctx_version > first.ctx_version
        discarded = s.scalars(select(Artifact.status).where(
            Artifact.path == path, Artifact.instance_id == first.instance_id, Artifact.status == "discarded"))
        assert list(discarded) == ["discarded"]


def test_review_precheck_flags_contradiction_without_llm(ws):
    r, emb = ws
    path = "frontend/src/api/session.ts"
    edit = [{"tool": "edit_file", "input": {"path": path, "summary": "token cache",
             "content": "export function save(t: string) {\n  localStorage.setItem(\"token\", t);\n}\n"}}]
    with db.session_scope() as s:
        turn = run_turn(s, r, emb, _adapters(personal=edit), _member(s, "riya"), "/do-it-yourself cache the token")
        assert turn.action == "edit_file" and turn.result.status == "completed"
        # An empty reviewer script never submits: the review is incomplete but keeps the precheck.
        res = run_review(s, r, emb, _adapters(reviewer=[]), _member(s, "riya"), paths=[path])
        assert res.status == "failed" and res.verdict == "incomplete"
        comments = list(s.scalars(select(ReviewComment).where(ReviewComment.review_id == res.review_id)))
        assert [(c.source, c.line) for c in comments] == [("precheck", 2)]
        assert "auth.token_transport" in comments[0].body
        assert s.get(Review, res.review_id).paths == [path]


def test_personal_agent_edit_scoped_to_team(ws):
    r, emb = ws
    edit = [{"tool": "edit_file", "input": {"path": "backend/app/main.py", "content": "x = 1\n", "summary": "nope"}}]
    with db.session_scope() as s:
        before = artifacts.current(s, "backend/app/main.py").version
        turn = run_turn(s, r, emb, _adapters(personal=edit), _member(s, "riya"), "edit the backend")
        assert turn.result.status == "failed" and "outside" in turn.result.error
        assert artifacts.current(s, "backend/app/main.py").version == before


def test_slash_commands_override_the_llm(ws):
    r, emb = ws
    with db.session_scope() as s:
        t = run_turn(s, r, emb, _adapters(), _member(s, "aarav"), "/research how do SameSite cookies work?")
        assert t.action == "answer" and "Increment 4" in t.reply and t.llm_calls == 0
        t = run_turn(s, r, emb, _adapters(reviewer=[]), _member(s, "aarav"), "/review")
        assert t.action == "request_review" and "nothing to review" in t.reply


def test_runner_strips_secrets_and_secret_files(ws, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-should-not-leak")
    probe = (
        "import os, pathlib\n\n"
        "def test_no_secrets():\n"
        "    assert 'ANTHROPIC_API_KEY' not in os.environ\n"
        "    assert not any(k.startswith('MPC_') for k in os.environ)\n"
        "    assert not pathlib.Path('backend/.env').exists()\n"
        "    assert pathlib.Path('backend/app/main.py').exists()\n"
    )
    with db.session_scope() as s:
        res = run_tests(s, ["tests/test_probe.py"], overlay={"tests/test_probe.py": probe})
    assert res.ok and res.passed == 1, res.output


def test_runner_times_out(ws):
    slow = "import time\n\ndef test_slow():\n    time.sleep(30)\n"
    with db.session_scope() as s:
        res = run_tests(s, ["tests/test_slow.py"], overlay={"tests/test_slow.py": slow}, timeout_s=3)
    assert res.timed_out and not res.ok


def test_testing_worker_records_test_runs_and_bugs(ws):
    r, emb = ws
    failing = "def test_contract():\n    assert 1 == 2\n"
    steps = [
        {"tool": "write_file", "input": {"path": "tests/backend/test_tiny.py", "content": failing}},
        {"tool": "run_tests", "input": {"paths": ["tests/backend/test_tiny.py"]}},
        {"tool": "report_bug", "input": {"team": "backend", "title": "Tiny contract bug", "description": "1 is not 2"}},
        {"tool": "finish", "input": {"summary": "found one"}},
    ]
    with db.session_scope() as s:
        res = run_worker(s, r, emb, _adapters(worker=steps), _member(s, "tanvi"), instructions="test it")
        assert res.status == "completed" and len(res.bugs) == 1 and res.tests[0].startswith("0 passed, 1 failed")
        tr = s.scalar(select(TestRun).where(TestRun.run_id == res.run_id))
        assert tr.failed == 1 and tr.exit_code == 1
        bug = s.get(Task, res.bugs[0])
        assert bug.team_id == "backend" and bug.created_by == "worker:te-tanvi:adhoc"
