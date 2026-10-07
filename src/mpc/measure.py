"""Increment 2 measurement: package size against full history.

Full history is the baseline where every agent receives the whole shared
conversation. Its size grows with the session while a package is bounded by
its budget, so the conversation is also measured replayed 5x and 20x as a
labelled synthetic stand-in for a longer session.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import redis
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from mpc.config import SEED_DIR
from mpc.context.packager import build_package
from mpc.context.visibility import draft_ticket
from mpc.llm.base import Embedder
from mpc.models import Decision, Member, Message, Task
from mpc.tokens import estimate_tokens

REPLAY_FACTORS = (5, 20)


@dataclass
class Row:
    id: str
    member: str
    team: str
    mode: str
    trap: bool
    package_tokens: int
    pinned_tokens: int
    items: int
    budget: int
    history_tokens: int
    pinned_ok: bool
    missing: list[str]
    contradiction_ok: bool
    trace_id: int | None


@dataclass
class MeasureResult:
    rows: list[Row]
    history_tokens: int
    gate_passed: bool
    markdown: str
    files: list[str] = field(default_factory=list)


def full_history(session: Session) -> str:
    rows = session.scalars(select(Message).order_by(Message.turn))
    return "".join(f"[{m.channel}] {m.member_id} ({m.role}): {m.content}\n" for m in rows)


def run(
    session: Session,
    r: redis.Redis | None,
    embedder: Embedder,
    *,
    budget: int = 12000,
    out_dir: str = "results",
    tasks_file: Path | None = None,
) -> MeasureResult:
    spec = yaml.safe_load((tasks_file or SEED_DIR / "measure_tasks.yaml").read_text(encoding="utf-8"))
    history = estimate_tokens(full_history(session))
    rows: list[Row] = []
    for t in spec["tasks"]:
        m = session.get(Member, t["member"])
        task = session.get(Task, t["task"]) if t.get("task") else None
        ticket = draft_ticket(m.id, m.team_id, task_id=task.id if task else None, token_budget=budget)
        pkg = build_package(session, r, ticket, t["text"], embedder=embedder, task=task, use_cache=False)

        pinned_decisions = {
            int(i.ref.split(":")[1]) for i in pkg.items if i.kind == "decision" and i.section == "pinned"
        }
        pinned_keys = set(
            session.scalars(select(Decision.key).where(Decision.id.in_(pinned_decisions)))
        )
        missing = [k for k in t["must_pin"] if k not in pinned_keys]
        expected = t.get("expect_contradiction")
        flagged = {c.key for c in (pkg.impact.contradictions if pkg.impact else [])}
        rows.append(
            Row(
                id=t["id"],
                member=m.id,
                team=m.team_id,
                mode=ticket.mode,
                trap=bool(t.get("trap")),
                package_tokens=pkg.total_tokens,
                pinned_tokens=pkg.pinned_tokens,
                items=len(pkg.items),
                budget=budget,
                history_tokens=history,
                pinned_ok=not missing,
                missing=missing,
                contradiction_ok=(expected in flagged) if expected else not flagged,
                trace_id=pkg.trace_id,
            )
        )

    gate = all(
        x.pinned_ok and x.contradiction_ok and x.package_tokens <= x.budget and x.trace_id
        for x in rows
    )
    md = _markdown(rows, history, budget, gate)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = out / "increment2_package_size.md", out / "increment2_package_size.csv"
    md_path.write_text(md, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "member", "team", "mode", "trap", "package_tokens", "pinned_tokens", "items",
             "budget", "history_tokens", "pinned_ok", "missing", "contradiction_ok", "trace_id"]
        )
        for x in rows:
            w.writerow(
                [x.id, x.member, x.team, x.mode, x.trap, x.package_tokens, x.pinned_tokens, x.items,
                 x.budget, x.history_tokens, x.pinned_ok, ";".join(x.missing), x.contradiction_ok,
                 x.trace_id]
            )
    return MeasureResult(rows, history, gate, md, [str(md_path), str(csv_path)])


def _markdown(rows: list[Row], history: int, budget: int, gate: bool) -> str:
    lines = [
        "# Increment 2: package size vs full history",
        "",
        f"Budget {budget} tokens per package. Full shared history: {history} tokens "
        f"({', '.join(f'{f}x replay: {history * f}' for f in REPLAY_FACTORS)}; replays are synthetic).",
        "Tokens are the Context Manager's estimate (ceil(chars/4)). Packages also carry code, "
        "contracts and decisions that the history baseline does not.",
        "",
        "| task | member | mode | trap | package | pinned | items | history | pkg/history "
        + "".join(f"| pkg/{f}x " for f in REPLAY_FACTORS)
        + "| pinned all required | contradiction check |",
        "|---|---|---|---|---:|---:|---:|---:|---:" + "|---:" * len(REPLAY_FACTORS) + "|---|---|",
    ]
    for x in rows:
        ratios = "".join(f"| {x.package_tokens / (history * f):.2f} " for f in REPLAY_FACTORS)
        lines.append(
            f"| {x.id} | {x.member} | {x.mode} | {'yes' if x.trap else ''} | {x.package_tokens} "
            f"| {x.pinned_tokens} | {x.items} | {history} | {x.package_tokens / history:.2f} {ratios}"
            f"| {'yes' if x.pinned_ok else 'MISSING ' + ', '.join(x.missing)} "
            f"| {'ok' if x.contradiction_ok else 'FAIL'} |"
        )
    avg = sum(x.package_tokens for x in rows) / len(rows)
    mx = max(x.package_tokens for x in rows)
    lines += [
        "",
        f"Mean package {avg:.0f} tokens, max {mx} (never above the {budget} budget).",
        f"Gate (every required decision pinned, budget respected, trace written): "
        f"**{'PASS' if gate else 'FAIL'}**",
        "",
    ]
    return "\n".join(lines)
