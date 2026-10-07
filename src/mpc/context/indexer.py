"""Artifacts indexer and knowledge chunks.

On every promoted file version: parse (Python ast, tree-sitter for
TypeScript), store imports and a summary on the artifact, chunk at
function/component level, redact secrets, embed into pgvector, and mark the
previous version's chunks superseded. Decision rationales, thread summaries,
test failures and research answers are indexed through the same path.
"""

from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from mpc.context.parsing import parse_file, resolve_imports, summarize
from mpc.context.repo import current_turn
from mpc.context.secrets import never_index, redact
from mpc.llm.base import Embedder
from mpc.models import Artifact, Chunk, Decision, DecisionKey
from mpc.tokens import estimate_tokens


@dataclass
class IndexStats:
    path: str
    chunks: int = 0
    redactions: int = 0
    superseded: int = 0
    skipped: str | None = None


def team_for_path(path: str) -> str | None:
    for prefix, team in (("frontend/", "frontend"), ("backend/", "backend"), ("tests/", "testing")):
        if path.startswith(prefix):
            return team
    return None


def classify(path: str) -> tuple[str, str, bool] | None:
    """(kind, visibility, is_global_spec) for a file path, or None to skip.

    Contracts (openapi/) are pinned from the contracts table rather than
    retrieved, so they are not chunked.
    """
    team = team_for_path(path)
    if team:
        return "code", team, False
    if path.startswith("openapi/"):
        return None
    name = path.rsplit("/", 1)[-1].lower()
    if name.startswith("requirements"):
        return "requirement", "shared", True
    if name.startswith("design-notes"):
        return "design_note", "frontend", False
    if name.startswith("schema"):
        return "schema", "backend", False
    return "doc", "shared", False


def _supersede(session: Session, *conds) -> int:
    res = session.execute(
        update(Chunk).where(Chunk.status == "active", *conds).values(status="superseded")
    )
    return res.rowcount or 0


def index_artifact(
    session: Session, artifact: Artifact, embedder: Embedder, known_paths: set[str] | None = None
) -> IndexStats:
    stats = IndexStats(path=artifact.path)
    stats.superseded = _supersede(session, Chunk.path == artifact.path)

    if never_index(artifact.path):
        artifact.summary = f"{artifact.path} — not indexed (secret file)."
        stats.skipped = "secret file"
        return stats

    if known_paths is None:
        known_paths = set(
            session.scalars(select(Artifact.path).where(Artifact.status == "current"))
        )
    parsed = parse_file(artifact.path, artifact.content)
    resolved = resolve_imports(artifact.path, parsed.language, parsed.imports, known_paths)
    artifact.language = parsed.language
    artifact.imports = parsed.imports
    artifact.resolved_imports = resolved
    artifact.summary = summarize(artifact.path, parsed, resolved)

    cls = classify(artifact.path)
    if cls is None:
        stats.skipped = "contract (pinned, not retrieved)"
        return stats
    kind, visibility, global_spec = cls

    turn = current_turn(session)
    texts, rows = [], []
    for pc in parsed.chunks:
        content, n = redact(pc.content)
        stats.redactions += n
        texts.append(f"{artifact.path} {pc.symbol or ''}\n{content}")
        rows.append(
            Chunk(
                kind=kind,
                source_ref=f"artifact:{artifact.id}",
                path=artifact.path,
                artifact_id=artifact.id,
                symbol=pc.symbol,
                content=content,
                tokens=estimate_tokens(content),
                visibility=visibility,
                is_global_spec=global_spec,
                author=artifact.created_by,
                created_turn=turn,
            )
        )
    for row, vec in zip(rows, embedder.embed(texts)):
        row.embedding = vec
        session.add(row)
    stats.chunks = len(rows)
    session.flush()
    return stats


def index_knowledge(
    session: Session,
    *,
    kind: str,
    visibility: str,
    content: str,
    author: str,
    embedder: Embedder,
    source_ref: str,
    is_global_spec: bool = False,
) -> Chunk:
    content, _ = redact(content)
    chunk = Chunk(
        kind=kind,
        source_ref=source_ref,
        content=content,
        tokens=estimate_tokens(content),
        visibility=visibility,
        is_global_spec=is_global_spec,
        author=author,
        created_turn=current_turn(session),
        embedding=embedder.embed([content])[0],
    )
    session.add(chunk)
    session.flush()
    return chunk


def decision_text(decision: Decision, dk: DecisionKey | None) -> str:
    desc = f" ({dk.description})" if dk and dk.description else ""
    return (
        f"Decision {decision.key} = {decision.value!r} [{decision.status}]{desc}. "
        f"Rationale: {decision.rationale or 'none recorded'}"
    )


def index_decision(session: Session, decision: Decision, embedder: Embedder) -> Chunk:
    """(Re)index a decision's rationale. Approved rationales are shared and
    count as important; proposal rationales stay with the proposing team."""
    _supersede(session, Chunk.decision_id == decision.id)
    dk = session.get(DecisionKey, decision.key)
    approved = decision.status == "approved"
    visibility = "shared" if approved or decision.team_id is None else decision.team_id
    content, _ = redact(decision_text(decision, dk))
    chunk = Chunk(
        kind="decision_rationale",
        source_ref=f"decision:{decision.id}",
        decision_id=decision.id,
        symbol=decision.key,
        content=content,
        tokens=estimate_tokens(content),
        visibility=visibility,
        decision_status=decision.status,
        is_global_spec=bool(dk and dk.is_global_spec),
        author=decision.proposed_by,
        created_turn=current_turn(session),
        embedding=embedder.embed([content])[0],
    )
    session.add(chunk)
    session.flush()
    return chunk


def retire_decision(session: Session, decision_id: int) -> int:
    return _supersede(session, Chunk.decision_id == decision_id)


def reindex_all(session: Session, embedder: Embedder) -> list[IndexStats]:
    """Re-index every current artifact (resolves imports across the full set)."""
    current = list(session.scalars(select(Artifact).where(Artifact.status == "current")))
    known = {a.path for a in current}
    return [index_artifact(session, a, embedder, known) for a in current]
