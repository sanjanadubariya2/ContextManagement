"""CLI smoke tests: Increment 1 gate (completions through the adapter) and
the Increment 2 package command."""

import json

import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from mpc import db
from mpc.cli import app
from mpc.models import Message

pytestmark = pytest.mark.db
runner = CliRunner()


def test_ask_goes_through_adapter_and_is_stored(ws):
    with db.session_scope() as s:
        before = s.scalar(select(func.count()).select_from(Message))
    res = runner.invoke(app, ["ask", "sneha", "What's left for the login endpoint?"])
    assert res.exit_code == 0, res.output
    assert "fake reply" in res.output and "fake/fake-1" in res.output
    with db.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(Message)) == before + 2
    r, _ = ws
    assert "login endpoint" in r.lrange("thread:sneha", -2, -2)[0]


def test_context_command_prints_package(ws):
    res = runner.invoke(app, ["context", "sneha", "Implement the login API", "--task", "t20"])
    assert res.exit_code == 0, res.output
    assert "Tokens:" in res.output and "trace #" in res.output
    res = runner.invoke(
        app, ["context", "sneha", "Implement the login API", "--task", "t20", "--json"]
    )
    pkg = json.loads(res.output)
    refs = [i["ref"] for i in pkg["items"]]
    assert any(r.startswith("decision:") for r in refs)
    assert "contract:openapi/auth.yaml@1" in refs
    assert pkg["total_tokens"] <= pkg["budget"]


def test_unknown_member_exits_nonzero(ws):
    res = runner.invoke(app, ["context", "nobody", "x"])
    assert res.exit_code == 1
