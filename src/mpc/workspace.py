"""Workspace setup: migrations, reset and seeding."""

from dataclasses import dataclass, field
from pathlib import Path

import redis
import yaml
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from mpc.config import PACKAGE_DIR, SEED_DIR, get_settings
from mpc.context import artifacts, contracts, registry
from mpc.context.cache import publish_version
from mpc.context.indexer import index_knowledge
from mpc.context.repo import bump_version, current_version, on_commit
from mpc.context.visibility import TEAM_ABBR
from mpc.db import get_engine
from mpc.llm.base import Embedder
from mpc.models import Agent, Member, Message, Task, Team
from mpc.redis_store import push_turn
from mpc.tokens import estimate_tokens


def alembic_config(url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(PACKAGE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url or get_settings().database_url)
    return cfg


def migrate(url: str | None = None) -> None:
    command.upgrade(alembic_config(url), "head")


def reset(url: str | None = None, r: redis.Redis | None = None) -> None:
    """Drop everything in the database and the Redis db. Destructive."""
    with get_engine(url).begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    if r is not None:
        r.flushdb()


def is_seeded(session: Session) -> bool:
    return session.scalar(select(Team.id).limit(1)) is not None


@dataclass
class SeedReport:
    files: int = 0
    chunks: int = 0
    redactions: int = 0
    skipped: list[str] = field(default_factory=list)
    decisions: int = 0
    messages: int = 0
    version: int = 0


def load_fixtures(path: Path | None = None) -> dict:
    return yaml.safe_load((path or SEED_DIR / "fixtures.yaml").read_text(encoding="utf-8"))


def starter_files(root: Path | None = None) -> dict[str, str]:
    root = root or SEED_DIR / "starter"
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def seed(session: Session, r: redis.Redis | None, embedder: Embedder) -> SeedReport:
    fx = load_fixtures()
    rep = SeedReport()

    for t in fx["teams"]:
        session.add(Team(id=t["id"], name=t["name"], write_root=t["write_root"]))
    session.flush()
    for m in fx["members"]:
        session.add(Member(id=m["id"], display_name=m["name"], team_id=m["team"], seat=m["seat"]))
    session.flush()
    for m in fx["members"]:
        session.add(Agent(id=f"agent_{TEAM_ABBR[m['team']]}_{m['id']}", member_id=m["id"]))
    session.flush()

    bump_version(session, "seed baseline: starter project, teams and decision keys", [])

    for k in fx["decision_keys"]:
        registry.register_key(
            session,
            k["key"],
            k["type"],
            allowed_values=k.get("values"),
            owner_team=k.get("owner_team"),
            keywords=k.get("keywords", []),
            is_global_spec=k.get("global_spec", False),
            description=k.get("description", ""),
        )

    files = starter_files()
    known = set(files)
    for path, content in files.items():
        art = artifacts.write_draft(session, path, content, created_by="seed")
        stats = artifacts.promote(session, art.id, embedder, known)
        rep.files += 1
        rep.chunks += stats.chunks
        rep.redactions += stats.redactions
        if stats.skipped:
            rep.skipped.append(f"{path}: {stats.skipped}")

    for c in fx["contracts"]:
        contracts.publish(
            session,
            c["path"],
            files[c["path"]],
            created_by="seed",
            linked_keys=c["linked_keys"],
            keywords=c["keywords"],
            bump=False,
        )

    for d in fx["decisions"]:
        prop = registry.propose(
            session,
            d["key"],
            d["value"],
            proposed_by=d["by"][0],
            team_id=d["team"],
            rationale=d["rationale"],
            embedder=embedder,
        )
        registry.approve(session, prop.id, approvers=d["by"], embedder=embedder)
        rep.decisions += 1

    for t in fx["tasks"]:
        session.add(
            Task(
                id=t["id"],
                title=t["title"],
                description=t.get("description", ""),
                team_id=t["team"],
                files=t.get("files", []),
                contracts=t.get("contracts", []),
                decision_keys=t.get("decision_keys", []),
                created_by=t["created_by"],
            )
        )
    session.flush()

    for i, k in enumerate(fx.get("knowledge", [])):
        index_knowledge(
            session,
            kind=k["kind"],
            visibility=k["visibility"],
            content=k["content"],
            author=k["author"],
            embedder=embedder,
            source_ref=f"seed:{i}",
        )

    team_of = {m["id"]: m["team"] for m in fx["members"]}
    for msg in fx.get("conversation", []):
        m = Message(
            member_id=msg["member"],
            agent_id=f"agent_{TEAM_ABBR[team_of[msg['member']]]}_{msg['member']}",
            channel=msg["channel"],
            role=msg.get("role", "user"),
            content=msg["content"],
            tokens=estimate_tokens(msg["content"]),
        )
        session.add(m)
        session.flush()
        rep.messages += 1
        if r is not None and msg["channel"] == f"personal:{msg['member']}":
            member, role, content, turn = msg["member"], m.role, m.content, m.turn
            on_commit(session, lambda a=member, b=role, c=content, d=turn: push_turn(r, a, b, c, d))

    rep.version = current_version(session)
    if r is not None:
        v = rep.version
        on_commit(session, lambda: publish_version(r, v, []))
    return rep
