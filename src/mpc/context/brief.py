"""Orientation brief (about 1-2k tokens) for a member's personal agent:
relevant approved decisions, likely files, open tasks and current leases."""

from dataclasses import dataclass

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from mpc.context.impact import ImpactReport
from mpc.context.registry import approved_map, proposals
from mpc.context.repo import current_version
from mpc.models import Artifact, Member, Task
from mpc.redis_store import current_leases
from mpc.tokens import estimate_tokens


@dataclass
class Brief:
    text: str
    tokens: int
    ctx_version: int
    truncated: bool


def build_brief(
    session: Session,
    r: redis.Redis | None,
    member: Member,
    impact: ImpactReport,
    *,
    max_tokens: int = 2000,
) -> Brief:
    version = current_version(session)
    approved = approved_map(session)
    relevant = [k for k in impact.decision_keys if k in approved]
    others = [k for k in approved if k not in relevant]

    sections: list[tuple[str, list[str]]] = []
    sections.append(
        (
            "Approved decisions relevant to this request (binding)",
            [
                f"- {k} = {approved[k].value!r} (approved v{approved[k].approved_in_version})"
                for k in relevant
            ]
            or ["- none"],
        )
    )
    if impact.contradictions:
        sections.append(
            (
                "Possible contradictions (approved decision wins; a change needs a Conflict Card)",
                [
                    f"- {c.key}: approved {c.approved_value!r}, {c.evidence} ({c.mentioned_value})"
                    for c in impact.contradictions
                ],
            )
        )
    summaries = dict(
        session.execute(
            select(Artifact.path, Artifact.summary).where(
                Artifact.status == "current", Artifact.path.in_(impact.file_paths)
            )
        ).all()
    )
    sections.append(
        (
            "Likely files",
            [f"- {f.path} [{f.reason}] {summaries.get(f.path, '')}" for f in impact.files]
            or ["- none identified"],
        )
    )
    if impact.cross_team_files:
        sections.append(
            (
                "Outside your team's paths (needs a cross-team task)",
                [f"- {p}" for p in impact.cross_team_files],
            )
        )
    tasks = session.scalars(
        select(Task)
        .where(Task.team_id == member.team_id, Task.status != "done")
        .order_by(Task.id)
    )
    sections.append(
        (
            f"Open tasks ({member.team_id})",
            [
                f"- {t.id} [{t.status}{' by ' + t.claimed_by if t.claimed_by else ''}] {t.title}"
                for t in tasks
            ]
            or ["- none"],
        )
    )
    leases = current_leases(r) if r is not None else {}
    sections.append(
        ("Current leases", [f"- {p} held by {h}" for p, h in sorted(leases.items())] or ["- none"])
    )
    pending = proposals(session)
    if pending or impact.open_keys:
        lines = [f"- proposal #{d.id}: {d.key} = {d.value!r} by {d.proposed_by}" for d in pending]
        lines += [f"- open key (no approved value yet): {k}" for k in impact.open_keys]
        sections.append(("Pending proposals and open keys", lines))
    sections.append(
        ("Other approved decisions", [f"- {k} = {approved[k].value!r}" for k in others])
    )

    head = (
        f"# Orientation brief for {member.display_name} ({member.team_id}) "
        f"at Context Repo v{version}\n"
    )
    out, used, truncated = [head], estimate_tokens(head), False
    for title, lines in sections:
        block = f"\n## {title}\n"
        if used + estimate_tokens(block) > max_tokens:
            truncated = True
            break
        out.append(block)
        used += estimate_tokens(block)
        for line in lines:
            t = estimate_tokens(line + "\n")
            if used + t > max_tokens:
                truncated = True
                break
            out.append(line + "\n")
            used += t
    text = "".join(out)
    return Brief(text=text, tokens=estimate_tokens(text), ctx_version=version, truncated=truncated)
