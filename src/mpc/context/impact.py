"""Impact analysis: which files, decision keys and contracts a request
touches, and whether it may contradict an approved decision.

Deterministic. Decision keys are found by typed keywords and value aliases
(never by similarity), so an approved decision is found even when it is not
textually similar to the request.
"""

import re
from dataclasses import asdict, dataclass, field

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from mpc.context.contracts import latest_all
from mpc.context.registry import approved_map
from mpc.context.visibility import WRITE_ROOTS
from mpc.llm.base import Embedder
from mpc.models import Artifact, DecisionKey, Task


@dataclass
class FileHit:
    path: str
    reason: str


@dataclass
class Contradiction:
    key: str
    approved_value: object
    mentioned_value: str
    evidence: str


@dataclass
class ImpactReport:
    team: str
    task_id: str | None
    files: list[FileHit] = field(default_factory=list)
    decision_keys: list[str] = field(default_factory=list)
    key_reasons: dict[str, str] = field(default_factory=dict)
    contracts: list[str] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    open_keys: list[str] = field(default_factory=list)
    cross_team_files: list[str] = field(default_factory=list)

    @property
    def file_paths(self) -> list[str]:
        return [f.path for f in self.files]

    def to_dict(self) -> dict:
        return asdict(self)


def phrase_in(phrase: str, text_lower: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", text_lower) is not None


def _add_key(report: ImpactReport, key: str, reason: str) -> None:
    if key not in report.key_reasons:
        report.decision_keys.append(key)
        report.key_reasons[key] = reason


def analyze(
    session: Session,
    request: str,
    team: str,
    *,
    task: Task | None = None,
    embedder: Embedder | None = None,
    similar_files: int = 3,
    min_similarity: float = 0.12,
) -> ImpactReport:
    report = ImpactReport(team=team, task_id=task.id if task else None)
    haystack = f"{task.title} {task.description} {request}" if task else request
    low = haystack.lower()

    # ---- files
    seen: set[str] = set()

    def add_file(path: str, reason: str) -> None:
        if path not in seen:
            seen.add(path)
            report.files.append(FileHit(path, reason))

    if task:
        for p in task.files:
            add_file(p, "task record")
    current_paths = list(session.scalars(select(Artifact.path).where(Artifact.status == "current")))
    for p in current_paths:
        base = p.rsplit("/", 1)[-1]
        if phrase_in(p, low) or (len(base) > 4 and phrase_in(base, low)):
            add_file(p, "named in request")
    if embedder is not None and similar_files > 0:
        vis = "testing" if team == "testing" else team
        qv = embedder.embed([haystack])[0]
        rows = session.execute(
            text(
                """
                SELECT path, max(1 - (embedding <=> CAST(:q AS vector))) AS sim
                FROM chunks
                WHERE status = 'active' AND kind = 'code' AND visibility = :vis
                GROUP BY path ORDER BY sim DESC LIMIT :k
                """
            ),
            {"q": str(qv), "vis": vis, "k": similar_files},
        )
        for path, sim in rows:
            if sim >= min_similarity:
                add_file(path, f"similar ({sim:.2f})")

    root = WRITE_ROOTS[team].removesuffix("**")
    report.cross_team_files = [
        f.path
        for f in report.files
        if not f.path.startswith(root) and f.path.split("/", 1)[0] in ("frontend", "backend", "tests")
    ]

    # ---- contracts
    contracts = latest_all(session)
    for path, c in contracts.items():
        if task and path in task.contracts:
            report.contracts.append(path)
        elif phrase_in(path, low) or any(phrase_in(k, low) for k in c.keywords):
            report.contracts.append(path)

    # ---- decision keys
    keys = list(session.scalars(select(DecisionKey).order_by(DecisionKey.key)))
    if task:
        for k in task.decision_keys:
            _add_key(report, k, "task record")
    for dk in keys:
        if dk.is_global_spec:
            _add_key(report, dk.key, "global spec")
            continue
        kw = next((k for k in dk.keywords if phrase_in(k, low)), None)
        if kw:
            _add_key(report, dk.key, f"keyword '{kw}'")
            continue
        for value, aliases in (dk.allowed_values or {}).items():
            alias = next((a for a in [value.replace("_", " "), *aliases] if phrase_in(a, low)), None)
            if alias:
                _add_key(report, dk.key, f"mentions '{alias}'")
                break
    for path in report.contracts:
        for k in contracts[path].linked_keys:
            _add_key(report, k, f"linked to {path}")

    # ---- contradictions and open keys
    approved = approved_map(session, report.decision_keys)
    by_key = {dk.key: dk for dk in keys}
    for key in report.decision_keys:
        d = approved.get(key)
        if d is None:
            report.open_keys.append(key)
            continue
        dk = by_key.get(key)
        if dk is None or dk.value_type != "enum":
            continue
        for value, aliases in (dk.allowed_values or {}).items():
            if value == d.value:
                continue
            alias = next((a for a in [value.replace("_", " "), *aliases] if phrase_in(a, low)), None)
            if alias:
                report.contradictions.append(
                    Contradiction(key, d.value, value, f"request mentions '{alias}'")
                )
    return report
