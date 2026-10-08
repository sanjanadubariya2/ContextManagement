"""mpc: headless driver for MultiPlayer-Context (Increments 1-2)."""

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from mpc import db, redis_store
from mpc.config import get_settings
from mpc.llm import LLMConfigError, get_adapter, model_for, resolve_provider
from mpc.llm.embeddings import get_embedder

app = typer.Typer(no_args_is_help=True, help="MultiPlayer-Context command line.")
decision_app = typer.Typer(no_args_is_help=True, help="Decision registry.")
artifact_app = typer.Typer(no_args_is_help=True, help="Versioned artifact store.")
app.add_typer(decision_app, name="decision")
app.add_typer(artifact_app, name="artifact")
# A piped Windows console is cp1252; replace what it can't encode rather than crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
console = Console()


def _redis():
    r = redis_store.get_redis()
    return r if redis_store.ping(r) else None


def _member(session, member_id: str):
    from mpc.models import Member

    m = session.get(Member, member_id)
    if m is None:
        console.print(f"[red]No member {member_id!r}.[/] Run `mpc members` to list them.")
        raise typer.Exit(1)
    return m


# ------------------------------------------------------------------ setup


@app.command()
def init(
    reset: bool = typer.Option(False, "--reset", help="Drop all data first (destructive)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the reset confirmation."),
):
    """Migrate the schema and seed the starter workspace."""
    from mpc import workspace

    if not db.ping():
        console.print("[red]Postgres is not reachable.[/] Start it with `docker compose up -d`.")
        raise typer.Exit(1)
    r = _redis()
    if reset:
        if not yes:
            typer.confirm("This drops every table and flushes the Redis db. Continue?", abort=True)
        workspace.reset(r=r)
    workspace.migrate()
    with db.session_scope() as s:
        if workspace.is_seeded(s):
            console.print("Schema migrated; workspace already seeded (use --reset to reseed).")
            return
        rep = workspace.seed(s, r, get_embedder())
    console.print(
        f"Seeded: {rep.files} files, {rep.chunks} chunks ({rep.redactions} redactions), "
        f"{rep.decisions} approved decisions, {rep.messages} messages. "
        f"Context Repo at [bold]v{rep.version}[/]."
    )
    for line in rep.skipped:
        console.print(f"  not indexed: {line}")


@app.command()
def migrate():
    """Apply Alembic migrations only."""
    from mpc import workspace

    workspace.migrate()
    console.print("Migrated to head.")


@app.command()
def status():
    """Store health, Context Repo version and row counts."""
    from mpc.context.repo import current_turn, current_version
    from mpc.models import Artifact, Chunk, ContextTrace, Decision, Message

    pg, r = db.ping(), _redis()
    s = get_settings()
    console.print(f"Postgres: {'up' if pg else 'DOWN'}   Redis: {'up' if r else 'DOWN'}")
    try:
        provider, why = resolve_provider()
        model = model_for(provider)
        console.print(f"LLM: {provider} ({model}) - {why}   Embeddings: {s.embed_provider}")
    except LLMConfigError as e:
        console.print(f"[red]LLM: {e}[/]")
    if not pg:
        raise typer.Exit(1)
    with db.session_scope() as sess:
        v = current_version(sess)
        cached = r.get(redis_store.CTX_VERSION) if r else None
        console.print(f"Context Repo: v{v} (Redis ctx_version={cached})   turn {current_turn(sess)}")
        for label, q in [
            ("approved decisions", select(func.count()).where(Decision.status == "approved")),
            ("proposals", select(func.count()).where(Decision.status == "proposal")),
            ("current files", select(func.count()).where(Artifact.status == "current")),
            ("active chunks", select(func.count()).where(Chunk.status == "active")),
            ("superseded chunks", select(func.count()).where(Chunk.status == "superseded")),
            ("messages", select(func.count()).select_from(Message)),
            ("context traces", select(func.count()).select_from(ContextTrace)),
        ]:
            console.print(f"  {label}: {sess.scalar(q)}")


@app.command()
def members():
    """List teams, members and their personal agents."""
    from mpc.models import Agent, Member

    with db.session_scope() as s:
        t = Table("team", "seat", "member", "agent")
        rows = s.execute(
            select(Member, Agent.id).join(Agent, Agent.member_id == Member.id).order_by(
                Member.team_id, Member.seat
            )
        )
        for m, agent_id in rows:
            t.add_row(m.team_id, str(m.seat), f"{m.display_name} ({m.id})", agent_id)
        console.print(t)


# -------------------------------------------------------------- agents


def _turn(member: str, text: str, task: Optional[str], script: Optional[dict]):
    from mpc.agents.personal import run_turn
    from mpc.agents.runtime import Adapters
    from mpc.models import Task

    adapters = Adapters.from_script(script) if script else Adapters()
    with db.session_scope() as s:
        m = _member(s, member)
        if task and s.get(Task, task) is None:
            console.print(f"[red]No task {task!r}.[/] Run `mpc tasks` to list them.")
            raise typer.Exit(1)
        res = run_turn(s, _redis(), get_embedder(), adapters, m, text, task_id=task)
    console.print(res.reply, markup=False, highlight=False)
    provider = "script" if script else adapters.for_role("personal").provider
    console.print(
        f"[dim]action {res.action} · {provider} · {res.llm_calls} personal-agent calls "
        f"(in {res.input_tokens}, out {res.output_tokens}) · brief {res.brief_tokens} tokens at "
        f"v{res.ctx_version}[/]"
    )
    return res


@app.command()
def ask(
    member: str,
    text: str,
    task: Optional[str] = typer.Option(None, "--task", help="Task id, e.g. t20."),
    script: Optional[str] = typer.Option(None, "--script", help="Replay planned steps offline."),
):
    """One request to a member's personal agent (it may answer, edit, delegate or request review).

    Slash commands override its choice: /delegate, /review, /do-it-yourself.
    """
    from mpc.agents.runtime import load_script

    _turn(member, text, task, load_script(script) if script else None)


@app.command("run-script")
def run_script(path: str):
    """Run a scripted scenario end to end and check its expectations."""
    from mpc.agents.runtime import load_script
    from mpc.context.artifacts import current_versions

    sc = load_script(path)
    console.print(f"[bold]{sc.get('name', path)}[/]: {sc['member']} asks {sc['request']!r}")
    res = _turn(sc["member"], sc["request"], sc.get("task"), sc)
    exp = sc.get("expect") or {}
    run = res.result
    checks: list[tuple[str, bool]] = []
    if "action" in exp:
        checks.append((f"action is {exp['action']}", res.action == exp["action"]))
    if "status" in exp:
        checks.append((f"run status is {exp['status']}", bool(run) and run.status == exp["status"]))
    if "verdict" in exp:
        checks.append((f"review verdict is {exp['verdict']}", bool(run) and run.verdict == exp["verdict"]))
    if "promoted" in exp:
        with db.session_scope() as s:
            now = current_versions(s, list(exp["promoted"]))
        for p in exp["promoted"]:
            ok = bool(run) and p in run.files and now.get(p) == run.files[p]
            checks.append((f"{p} promoted (now v{now.get(p)})", ok))
    if "bugs" in exp:
        checks.append((f"{exp['bugs']} bug task(s) filed", bool(run) and len(run.bugs) == exp["bugs"]))
    for label, ok in checks:
        console.print(f"  {'[green]PASS[/]' if ok else '[red]FAIL[/]'} {label}")
    if not all(ok for _, ok in checks):
        raise typer.Exit(1)


@app.command()
def tasks(team: Optional[str] = typer.Option(None, "--team")):
    """The task boards."""
    from mpc.models import Task

    with db.session_scope() as s:
        q = select(Task).order_by(Task.team_id, Task.id)
        if team:
            q = q.where(Task.team_id == team)
        t = Table("id", "team", "status", "claimed by", "title", "created by")
        for x in s.scalars(q):
            t.add_row(x.id, x.team_id, x.status, x.claimed_by or "", x.title, x.created_by)
        console.print(t)


@app.command()
def instances(limit: int = typer.Option(20, "--limit")):
    """Recent subagent runs."""
    from mpc.models import InstanceRun

    with db.session_scope() as s:
        t = Table("run", "instance", "status", "try", "v", "files", "steps", "tokens in/out")
        for x in s.scalars(select(InstanceRun).order_by(InstanceRun.id.desc()).limit(limit)):
            files = ", ".join(f"{p.rsplit('/', 1)[-1]}@{v}" for p, v in (x.files or {}).items())
            t.add_row(str(x.id), x.instance_id, x.status, str(x.attempt), f"v{x.ctx_version}" if x.ctx_version else "",
                      files, str(x.steps), f"{x.input_tokens}/{x.output_tokens}")
        console.print(t)


@app.command()
def instance(run_id: int):
    """One subagent run in full."""
    from mpc.models import InstanceRun

    with db.session_scope() as s:
        x = s.get(InstanceRun, run_id)
        if x is None:
            console.print(f"[red]No run #{run_id}.[/]")
            raise typer.Exit(1)
        data = {c.name: getattr(x, c.name) for c in InstanceRun.__table__.columns}
    console.print_json(json.dumps(data, default=str))


@app.command()
def reviews(limit: int = typer.Option(10, "--limit")):
    """Recent reviews and their comments."""
    from mpc.models import Review, ReviewComment

    with db.session_scope() as s:
        for rv in s.scalars(select(Review).order_by(Review.id.desc()).limit(limit)):
            console.print(f"[bold]Review #{rv.id}[/] {rv.verdict} · {', '.join(rv.paths)} · requested by {rv.requested_by}")
            if rv.summary:
                console.print(f"  {rv.summary}", markup=False)
            for c in s.scalars(select(ReviewComment).where(ReviewComment.review_id == rv.id).order_by(ReviewComment.id)):
                where = f"{c.path}:{c.line}" if c.line else (c.path or "")
                console.print(f"  [{c.source}] {where} {c.body}", markup=False)


@app.command("run-tests")
def run_tests_cmd(paths: list[str] = typer.Argument(None, help="Paths under tests/.")):
    """Run tests against the current code in the sandboxed runner."""
    from mpc.models import TestRun
    from mpc.testing.runner import run_tests

    with db.session_scope() as s:
        res = run_tests(s, paths or ["tests"])
        s.add(TestRun(paths=res.paths, exit_code=res.exit_code, passed=res.passed, failed=res.failed,
                      errors=res.errors, timed_out=res.timed_out, duration_ms=res.duration_ms, output=res.output))
    console.print(res.output, markup=False, highlight=False)
    console.print(f"[bold]{res.summary()}[/] in {res.duration_ms} ms")
    if not res.ok:
        raise typer.Exit(1)


# ------------------------------------------------------- context manager


@app.command()
def brief(member: str, text: str):
    """Print a member's orientation brief for a request."""
    from mpc.context.brief import build_brief
    from mpc.context.impact import analyze

    with db.session_scope() as s:
        m = _member(s, member)
        impact = analyze(s, text, m.team_id, embedder=get_embedder())
        b = build_brief(s, _redis(), m, impact)
    console.print(b.text, markup=False, highlight=False)
    console.print(f"[dim]{b.tokens} tokens{' (truncated)' if b.truncated else ''}[/]")


@app.command()
def impact(member: str, text: str, task: Optional[str] = typer.Option(None, "--task")):
    """Print the impact analysis for a request (input to the policy guard)."""
    from mpc.context.impact import analyze
    from mpc.models import Task

    with db.session_scope() as s:
        m = _member(s, member)
        t = s.get(Task, task) if task else None
        rep = analyze(s, text, m.team_id, task=t, embedder=get_embedder())
    console.print_json(json.dumps(rep.to_dict(), default=str))


@app.command()
def context(
    member: str,
    text: str,
    task: Optional[str] = typer.Option(None, "--task", help="Task id, e.g. t21."),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="curated_fork | fork | isolated (default: by team)."
    ),
    budget: Optional[int] = typer.Option(None, "--budget", help="Token budget."),
    show: bool = typer.Option(False, "--show", help="Print the rendered package."),
    as_json: bool = typer.Option(False, "--json", help="Print the package as JSON."),
    no_cache: bool = typer.Option(False, "--no-cache"),
):
    """Build a member's context package and print it with its trace."""
    from mpc.context.packager import build_package
    from mpc.context.packing import TaskTooBig
    from mpc.context.visibility import draft_ticket
    from mpc.models import Task

    with db.session_scope() as s:
        m = _member(s, member)
        if task and s.get(Task, task) is None:
            console.print(f"[red]No task {task!r}.[/]")
            raise typer.Exit(1)
        ticket = draft_ticket(
            m.id,
            m.team_id,
            task_id=task,
            mode=mode,  # type: ignore[arg-type]
            token_budget=budget or get_settings().default_token_budget,
        )
        try:
            pkg = build_package(
                s, _redis(), ticket, text, embedder=get_embedder(), use_cache=not no_cache
            )
        except TaskTooBig as e:
            console.print(f"[red]Task too big:[/] {e}")
            raise typer.Exit(2)

    if as_json:
        console.print_json(json.dumps(pkg.to_dict(), default=str))
        return
    if show:
        console.print(pkg.render(), markup=False, highlight=False)

    console.print(
        f"[bold]Package {pkg.package_id[:12]}[/] for {ticket.instance_id} · mode {ticket.mode} · "
        f"Context Repo v{pkg.ctx_version} · {'cache hit · ' if pkg.cache_hit else ''}"
        f"trace #{pkg.trace_id}"
    )
    console.print(
        "[yellow]Unadmitted draft ticket: the policy guard that issues real tickets "
        "arrives in Increment 4.[/]"
    )
    t = Table("section", "ref", "kind", "tokens", "score", "sim", "why")
    for it in pkg.items:
        t.add_row(
            it.section,
            it.ref,
            it.kind,
            str(it.tokens),
            "" if it.score is None else f"{it.score:.3f}",
            "" if it.sim is None else f"{it.sim:.2f}",
            (it.reason or (it.path or ""))[:48],
        )
    console.print(t)
    sections = ", ".join(f"{k} {v}" for k, v in pkg.by_section.items())
    console.print(
        f"Tokens: [bold]{pkg.total_tokens}[/] / {pkg.budget} (pinned {pkg.pinned_tokens}; {sections})"
    )
    reasons: dict[str, int] = {}
    for d in pkg.dropped:
        reasons[d.reason] = reasons.get(d.reason, 0) + 1
    if reasons:
        console.print("Dropped: " + ", ".join(f"{n} {r}" for r, n in reasons.items()))
    deps = pkg.dependencies
    console.print(
        f"Stamp: v{deps['ctx_version']} · {len(deps['decisions'])} decision keys · "
        f"contracts {deps['contracts']} · {len(deps['files'])} files"
    )
    if pkg.impact and pkg.impact.contradictions:
        for c in pkg.impact.contradictions:
            console.print(
                f"[red]Possible contradiction:[/] {c.key} is approved as {c.approved_value!r}; {c.evidence}"
            )


@app.command()
def trace(trace_id: int):
    """Show a stored context trace."""
    from mpc.models import ContextTrace

    with db.session_scope() as s:
        row = s.get(ContextTrace, trace_id)
        if row is None:
            console.print(f"[red]No trace #{trace_id}.[/]")
            raise typer.Exit(1)
        data = {
            "id": row.id,
            "created_at": row.created_at,
            "member": row.member_id,
            "instance": row.instance_id,
            "task": row.task_id,
            "mode": row.mode,
            "ctx_version": row.ctx_version,
            "tokens": f"{row.total_tokens}/{row.token_budget}",
            "pinned_tokens": row.pinned_tokens,
            "cache_hit": row.cache_hit,
            "items": row.items,
            "dropped": len(row.dropped),
            "dependencies": row.dependencies,
        }
    console.print_json(json.dumps(data, default=str))


@app.command()
def stale(trace_id: int):
    """Check whether a traced package is stale against the current repo."""
    from mpc.context.packager import check_staleness
    from mpc.models import ContextTrace

    with db.session_scope() as s:
        row = s.get(ContextTrace, trace_id)
        if row is None:
            console.print(f"[red]No trace #{trace_id}.[/]")
            raise typer.Exit(1)
        st = check_staleness(s, row.dependencies)
    if st.stale:
        console.print(f"[red]Stale[/] (packed v{st.packed_version}, now v{st.current_version}):")
        for c in st.changes:
            console.print(f"  {c}")
    else:
        console.print(f"Fresh (packed v{st.packed_version}, now v{st.current_version}).")


@app.command()
def index():
    """Re-index every current artifact."""
    from mpc.context.indexer import reindex_all

    with db.session_scope() as s:
        stats = reindex_all(s, get_embedder())
    console.print(
        f"Indexed {len(stats)} files: {sum(x.chunks for x in stats)} chunks, "
        f"{sum(x.redactions for x in stats)} redactions."
    )


@app.command()
def measure(
    budget: int = typer.Option(12000, "--budget"),
    out: str = typer.Option("results", "--out", help="Directory for the results files."),
):
    """Increment 2 measurement: package size vs full history on 10 scripted tasks."""
    from mpc.measure import run

    with db.session_scope() as s:
        res = run(s, _redis(), get_embedder(), budget=budget, out_dir=out)
    console.print(res.markdown, markup=False, highlight=False)
    console.print(f"[dim]Wrote {', '.join(res.files)}[/]")
    if not res.gate_passed:
        raise typer.Exit(1)


# ---------------------------------------------------------- decisions


@decision_app.command("keys")
def decision_keys():
    """List typed decision keys."""
    from mpc.models import DecisionKey

    with db.session_scope() as s:
        t = Table("key", "type", "values", "owner", "global")
        for dk in s.scalars(select(DecisionKey).order_by(DecisionKey.key)):
            vals = ", ".join(dk.allowed_values or {}) if dk.value_type == "enum" else ""
            t.add_row(dk.key, dk.value_type, vals, dk.owner_team or "-", "yes" if dk.is_global_spec else "")
        console.print(t)


@decision_app.command("list")
def decision_list(all_: bool = typer.Option(False, "--all", help="Include superseded and rejected.")):
    """List approved decisions and open proposals."""
    from mpc.models import Decision

    with db.session_scope() as s:
        q = select(Decision).order_by(Decision.key, Decision.id)
        if not all_:
            q = q.where(Decision.status.in_(["approved", "proposal"]))
        t = Table("id", "key", "value", "status", "version", "by")
        for d in s.scalars(q):
            v = d.approved_in_version or ""
            t.add_row(str(d.id), d.key, json.dumps(d.value), d.status, f"v{v}" if v else "", d.proposed_by)
        console.print(t)


@decision_app.command("propose")
def decision_propose(
    member: str,
    key: str,
    value: str,
    rationale: str = typer.Option("", "--rationale", "-r"),
):
    """File a proposal (agents and members may only propose)."""
    from mpc.context.registry import RegistryError, propose

    with db.session_scope() as s:
        m = _member(s, member)
        try:
            d = propose(
                s, key, value, proposed_by=m.id, team_id=m.team_id, rationale=rationale,
                embedder=get_embedder(),
            )
        except RegistryError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        console.print(f"Proposal #{d.id}: {d.key} = {d.value!r}")


@decision_app.command("approve")
def decision_approve(
    decision_id: int,
    by: str = typer.Option(..., "--by", help="Comma-separated approving members."),
):
    """Approve a proposal. Quorum rules (2 of 3, votes) arrive in Increment 4."""
    from mpc.context.registry import RegistryError, approve

    approvers = [x.strip() for x in by.split(",") if x.strip()]
    with db.session_scope() as s:
        for a in approvers:
            _member(s, a)
        try:
            v = approve(s, decision_id, approvers=approvers, embedder=get_embedder(), r=_redis())
        except RegistryError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
    console.print(f"Approved #{decision_id}; Context Repo is now [bold]v{v}[/].")


@decision_app.command("reject")
def decision_reject(decision_id: int):
    """Reject an open proposal."""
    from mpc.context.registry import RegistryError, reject

    with db.session_scope() as s:
        try:
            reject(s, decision_id)
        except RegistryError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
    console.print(f"Rejected #{decision_id}.")


@decision_app.command("history")
def decision_history(key: str):
    """Every decision ever filed on a key."""
    from mpc.context.registry import history

    with db.session_scope() as s:
        t = Table("id", "value", "status", "approved", "superseded", "by")
        for d in history(s, key):
            t.add_row(
                str(d.id),
                json.dumps(d.value),
                d.status,
                f"v{d.approved_in_version}" if d.approved_in_version else "",
                f"v{d.superseded_in_version}" if d.superseded_in_version else "",
                d.proposed_by,
            )
        console.print(t)


@decision_app.command("snapshot")
def decision_snapshot(version: int):
    """Approved values as of a Context Repo version."""
    from mpc.context.registry import snapshot

    with db.session_scope() as s:
        console.print_json(json.dumps(snapshot(s, version)))


# ---------------------------------------------------------- artifacts


@artifact_app.command("list")
def artifact_list():
    """Current file versions."""
    from mpc.models import Artifact

    with db.session_scope() as s:
        t = Table("path", "version", "lang", "team", "imports")
        for a in s.scalars(select(Artifact).where(Artifact.status == "current").order_by(Artifact.path)):
            t.add_row(a.path, str(a.version), a.language, a.team_id or "-", str(len(a.resolved_imports)))
        console.print(t)


@artifact_app.command("show")
def artifact_show(path: str):
    """A file's current version, summary and imports."""
    from mpc.context.artifacts import current

    with db.session_scope() as s:
        a = current(s, path)
        if a is None:
            console.print(f"[red]No current version of {path}.[/]")
            raise typer.Exit(1)
        console.print(f"[bold]{a.path}[/] v{a.version} ({a.language})\n{a.summary}")
        console.print(f"Imports: {a.resolved_imports or a.imports}")


@artifact_app.command("put")
def artifact_put(
    member: str,
    path: str,
    file: typer.FileText = typer.Argument(..., help="Local file with the new content."),
):
    """Write a draft and promote it (validation arrives with the guard in Increment 4)."""
    from mpc.context import artifacts

    content = file.read()
    with db.session_scope() as s:
        _member(s, member)
        art = artifacts.write_draft(s, path, content, created_by=member)
        stats = artifacts.promote(s, art.id, get_embedder(), r=_redis())
        console.print(f"{path} is now v{art.version}: {stats.chunks} chunks, {stats.redactions} redactions.")


@app.command("llm-check")
def llm_check():
    """Make one tiny real call to the configured LLM and report the result."""
    try:
        provider, why = resolve_provider()
        adapter = get_adapter()
    except LLMConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"Provider: {provider} ({why}), model {adapter.model}")
    if provider == "fake":
        console.print("[yellow]The fake provider is active (MPC_LLM_PROVIDER=fake); no live call made.[/]")
        raise typer.Exit(1)
    try:
        c = adapter.complete([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=1024)
    except Exception as e:  # provider SDK errors, translated below
        console.print(f"[red]{_explain_llm_error(provider, adapter.model, e)}[/]")
        raise typer.Exit(1)
    if c.stop_reason == "refusal":
        console.print(f"[yellow]The model declined ({c.refusal_category}); the key and model work.[/]")
        return
    console.print(f"[green]OK[/]: {c.model} replied {c.text.strip()!r} "
                  f"(in {c.input_tokens}, out {c.output_tokens}, stop {c.stop_reason})")


def _explain_llm_error(provider: str, model: str, e: Exception) -> str:
    if provider == "gemini":
        from google.genai import errors as gerr

        if isinstance(e, gerr.APIError):
            msg = (e.message or "").strip()
            if e.code in (400, 401, 403) and "key" in msg.lower():
                return f"The Gemini API key was rejected ({e.code}): {msg}"
            if e.code == 404:
                return f"Model {model!r} was not found (404). Set MPC_GEMINI_MODEL. {msg}"
            if e.code == 429:
                return f"Quota or rate limit hit (429); the key works. {msg}"
            return f"Gemini API error {e.code}: {msg}"
        return f"Could not reach the Gemini API: {e}"
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected (401)."
    if isinstance(e, anthropic.NotFoundError):
        return f"Model {model!r} was not found (404). Set MPC_LLM_MODEL."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited (429); the key works."
    if isinstance(e, anthropic.APIStatusError):
        return f"Anthropic API error {e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API (network or proxy)."
    return f"{type(e).__name__}: {e}"


def main() -> None:
    try:
        app()
    except LLMConfigError as e:
        console.print(f"[red]{e}[/]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
