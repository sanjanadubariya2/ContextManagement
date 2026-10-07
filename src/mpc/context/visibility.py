"""Team visibility and the admission ticket.

Visibility grants are strings: "<scope>" grants every kind in that scope,
"<scope>:<kind>" grants one kind only. Scopes are team ids or "shared".

| Team     | Receives                                         | Never receives                    |
|----------|--------------------------------------------------|-----------------------------------|
| frontend | approved decisions, contract, frontend code,     | backend implementation discussion |
|          | design notes                                     |                                   |
| backend  | approved decisions, contract, schema, backend    | frontend implementation discussion|
|          | code                                             |                                   |
| testing  | requirements, approved decisions, contracts,     | any implementer reasoning or      |
|          | code at a fixed version, past test failures      | discussion                        |

Approved decisions and contracts are pinned by key, so they reach every team
regardless of these chunk grants.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

Mode = Literal["curated_fork", "fork", "isolated"]

TEAM_GRANTS: dict[str, list[str]] = {
    "frontend": ["frontend", "shared"],
    "backend": ["backend", "shared"],
    "testing": [
        "testing",
        "shared:requirement",
        "shared:decision_rationale",
        "frontend:code",
        "backend:code",
    ],
}

# Kinds a team never receives even when a grant would otherwise match.
TEAM_DENY: dict[str, set[str]] = {
    "frontend": set(),
    "backend": set(),
    "testing": {"thread_summary", "discussion", "design_note"},
}


def grants_allow(grants: list[str], visibility: str, kind: str) -> bool:
    return visibility in grants or f"{visibility}:{kind}" in grants


def chunk_allowed(
    team: str, grants: list[str], visibility: str, kind: str, decision_status: str | None
) -> bool:
    """Python mirror of the SQL filter, used as a second check on every candidate."""
    if kind in TEAM_DENY.get(team, set()) and visibility != team:
        return False
    # Rationales of unapproved proposals are discussion, and testing never sees those.
    if team == "testing" and kind == "decision_rationale" and decision_status != "approved":
        return False
    return grants_allow(grants, visibility, kind)


@dataclass
class Ticket:
    """What the policy guard hands the Context Manager for one subagent instance.

    The guard (Increment 4) is the only issuer of admitted tickets. Before it
    exists, `draft_ticket` builds an unadmitted one so the Context Manager can
    be exercised from the CLI; `admitted` stays False on those.
    """

    ticket_id: str
    task_id: str | None
    instance_id: str
    template: str
    mode: Mode
    team: str
    member_id: str
    visibility: list[str]
    write_paths: list[str]
    leases: list[str] = field(default_factory=list)
    token_budget: int = 12000
    expires_at: str = ""
    admitted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def expired(self, now: datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        return (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(self.expires_at)


TEAM_ABBR = {"frontend": "fe", "backend": "be", "testing": "te"}
WRITE_ROOTS = {"frontend": "frontend/**", "backend": "backend/**", "testing": "tests/**"}


def draft_ticket(
    member_id: str,
    team: str,
    *,
    task_id: str | None = None,
    mode: Mode | None = None,
    template: str = "worker",
    token_budget: int = 12000,
    ttl_minutes: int = 30,
) -> Ticket:
    # Testing workers are isolated from implementer reasoning by template definition.
    if mode is None:
        mode = "isolated" if team == "testing" else "curated_fork"
    tid = task_id or "adhoc"
    return Ticket(
        ticket_id=f"draft-{member_id}-{tid}",
        task_id=task_id,
        instance_id=f"{template}:{TEAM_ABBR[team]}-{member_id}:{tid}",
        template=f"{team}_{template}",
        mode=mode,
        team=team,
        member_id=member_id,
        visibility=list(TEAM_GRANTS[team]),
        write_paths=[WRITE_ROOTS[team]],
        token_budget=token_budget,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(),
    )
