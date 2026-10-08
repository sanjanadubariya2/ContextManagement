# MultiPlayer-Context

Nine developers in three teams, each with a personal agent, building one website while a
central Context Manager decides what every agent knows. This repository implements
**Increment 1 (foundations)**, **Increment 2 (Context Manager core)** and **Increment 3
(agents on the Context Manager)** of the plan. All run headless and are driven from the `mpc` CLI.

## Setup

```sh
docker compose up -d          # Postgres 17 + pgvector on :5433, Redis 7 on :6380
uv sync
uv run mpc init               # migrate, seed 3 teams / 9 members / starter project, Context Repo -> v14
uv run mpc status
```

`mpc init --reset` wipes and reseeds. Settings come from `MPC_*` environment variables or
`.env` (see `.env.example`).

**LLM.** Gemini or Claude, behind one adapter interface (`complete`, `tool_call`, `embed`).
`MPC_LLM_PROVIDER=gemini` uses the Gemini API through Google's `google-genai` SDK with
`GEMINI_API_KEY` (default model `gemini-3.8-flash`; `MPC_GEMINI_MODEL=gemini-3.1-pro-preview` for
the stronger preview model). `MPC_LLM_PROVIDER=anthropic` uses Claude with `ANTHROPIC_API_KEY`
(default `claude-opus-5-5`). `auto`, the default, picks Gemini when its key is set, else Claude.
There is no silent fallback: with no usable key the CLI stops and names what is missing. The
offline `fake` provider runs only when `MPC_LLM_PROVIDER=fake` is set, as the tests do.

Keys go in `.env` (saved) or the shell; the shell wins. `uv run mpc status` shows the active
provider and why, and `uv run mpc llm-check` makes one tiny live call and names the problem if it
fails (key rejected, model not found, quota).

**Embeddings.** Default `hash`: deterministic, offline, lexical feature hashing, so experiments
are repeatable and cost nothing. `MPC_EMBED_PROVIDER=gemini` uses `gemini-embedding-001` (same
key) for semantic retrieval; `voyage` is also available. All produce 1024-d vectors. After
switching, rebuild with `uv run mpc init --reset -y` so every chunk uses the same embedder.

## Commands

| Command | What it does |
|---|---|
| `mpc ask sneha "..." [--task t20]` | One request to Sneha's personal agent: it answers, makes a small edit, delegates to a Worker or requests a review. `/delegate`, `/review`, `/do-it-yourself` override its choice; `--script` replays planned steps offline |
| `mpc run-script seed/scripts/<name>.yaml` | Run a scripted scenario end to end and check its expectations (PASS/FAIL) |
| `mpc instances` / `mpc instance 3` | Subagent runs: status, attempt, packed version, promoted files, tokens |
| `mpc reviews` / `mpc tasks [--team backend]` | Reviews with their comments / the task boards |
| `mpc run-tests tests/backend` | Run tests against the current code in the sandboxed runner |
| `mpc context sneha "implement login API" --task t20` | Build a context package and print it with its trace (`--show` renders it, `--json` dumps it, `--mode`, `--budget`) |
| `mpc brief aarav "..."` / `mpc impact aarav "..."` | Orientation brief / impact analysis |
| `mpc trace 12` / `mpc stale 12` | Inspect a stored trace / staleness-check it against the current repo |
| `mpc decision keys \| list \| propose \| approve \| reject \| history \| snapshot` | Decision registry |
| `mpc artifact list \| show \| put` | Versioned artifact store (`put` writes a draft and promotes it) |
| `mpc measure` | Increment 2 measurement → `results/increment2_package_size.{md,csv}` |
| `mpc members`, `mpc index`, `mpc migrate` | Housekeeping |

## Scenarios (offline, repeatable)

Scripts in `seed/scripts/` replay the LLM's tool calls; everything else is the real runtime. Run
them in this order on a fresh workspace (`uv run mpc init --reset -y`):

```sh
uv run mpc run-script seed/scripts/kabir_login_tests.yaml   # Testing worker finds 422-vs-400, files bug t33
uv run mpc run-script seed/scripts/sneha_login_api.yaml     # Increment 3 gate: login API, files promoted
uv run mpc run-tests tests/backend/test_login_contract.py   # Kabir's tests now pass
uv run mpc run-script seed/scripts/riya_login_page.yaml     # a Worker stores the token in localStorage...
uv run mpc run-script seed/scripts/riya_review.yaml         # ...and the isolated Reviewer catches it
```

With `ANTHROPIC_API_KEY` set, the same requests work unscripted:
`uv run mpc ask sneha "Implement the login API per openapi/auth.yaml" --task t20`.

## What is built

**Increment 1.** Alembic migrations (`src/mpc/migrations`) for teams, members, agents, decisions,
context_repo_versions, contracts, tasks, artifacts, context_traces and messages. The seeded
starter project (`seed/starter`: Next.js frontend, FastAPI backend, tests,
`openapi/auth.yaml`, `openapi/dashboard.yaml`, requirements and design notes) is loaded into the
artifact store. Also: the 9 members as fixtures, the LLM adapter and the Typer CLI.

**Increment 2** (`src/mpc/context`):

- `registry.py`: typed decision keys (enum/int/bool/string, with value aliases), the
  proposal → approved → superseded lifecycle, version history and snapshots. Each approval bumps
  the Context Repo version, then after commit publishes `ctx_version` to Redis and invalidates
  cached packages that depend on the key.
- `indexer.py`, `parsing.py`, `secrets.py`: the artifacts indexer. It parses Python with `ast` and
  TypeScript/TSX with tree-sitter, stores imports (resolved to artifact paths) and summaries,
  chunks at function/component level, and embeds into pgvector. Old chunks are marked superseded.
  `.env`/key files are never indexed, and secrets are redacted before embedding.
- `packing.py` (pure) and `packager.py`: the package builder. It pins items by key (task record,
  approved decisions on touched keys, contracts), retrieves pgvector candidates filtered to the
  ticket's team visibility, and adds live turns for fork modes. Candidates are scored with
  `0.5·sim + 0.3·e^(−λΔt) + 0.2·I` and packed greedily under the budget, followed by dependency
  closure over direct imports. Every item carries provenance and is wrapped as escaped data. The
  package is stamped with the version and its dependencies, and a trace row is written. Also here:
  the Redis package cache, on-demand `expand`, and `check_staleness` for write-back.
- `impact.py`, `brief.py`: deterministic impact analysis (files, decision keys, contracts,
  possible contradictions, cross-team files) and the ≤2k-token orientation brief.

Gate: **passes.** `mpc measure` pins every required approved decision on all 10 scripted tasks,
including three trap tasks, and stays under budget with a trace row each. The tests assert that
approved decisions are always pinned, the budget is never exceeded, and superseded chunks are
never returned.

**Increment 3** (`src/mpc/agents`, `src/mpc/testing`):

- `personal.py`: one personal agent per member as a LangGraph graph
  (intake → orient → decide → answer | edit | delegate | review). The decide node's tool call is
  the only LLM choice; slash commands override it. The conversation is stored in Postgres with
  recent turns in Redis, and the orientation brief rides in the system prompt.
- `templates.py`: Worker and Reviewer, one variant per team (write paths, skills, review
  criteria). Workers are curated forks; Testing workers and all Reviewers are isolated.
- `runtime.py`: the execution runtime. Instances get IDs such as `worker:be-sneha:t20`, a draft
  workspace in the artifact store, and scoped file tools (`list_files`, `read_file`, `write_file`,
  `request_more_context`). Writes outside the template's paths, secret files and path escapes are
  refused. Write-back validates syntax (an invalid result loops back to the instance), runs the
  staleness check (a stale result is discarded and re-run once on the new version), promotes
  drafts (re-embedding them), files proposals and records the run in `instance_runs`. Reviews
  combine the Reviewer's comments with a deterministic pre-check that flags lines contradicting
  approved decisions.
- `testing/runner.py`: Testing workers run pytest on a temporary copy of the code plus their
  drafts, in a subprocess with a timeout, with only an allowlist of OS environment variables and
  no secret files. Results go to `test_runs`; `report_bug` puts a task on the owning team's board.

Gate: **passes.** `mpc run-script seed/scripts/sneha_login_api.yaml` runs Sneha's request end to
end and promotes `backend/app/main.py` and `backend/app/auth/routes.py` to v2 (and the new
`backend/app/errors.py` to v1).

```sh
uv run pytest      # 107 tests; DB tests use the separate mpc_test database and Redis db 1
```

## Deliberate choices and limits

- **Tickets are drafts.** The policy guard that issues admitted tickets is Increment 4. Until then
  the CLI builds an unadmitted `draft_ticket` with the same shape, and the Context Manager
  enforces its visibility, budget and expiry.
- **No admission control yet.** The two-Worker cap, file leases, atomic task claims and diff
  validation against decisions belong to the policy guard (Increment 4). The runtime already
  enforces write paths, and honours `leases` once a ticket carries them.
- **Scripts stand in for the LLM offline.** `riya_login_page.yaml` makes a deliberate mistake so
  the review scenario has something to catch. Live Claude runs need `ANTHROPIC_API_KEY`; the
  adapter's request shape is unit-tested against a stubbed SDK client.
- **Python tests only.** The runner executes pytest; Playwright end-to-end tests need a running
  site and come with the live session.
- **Approval has no quorum yet.** `registry.approve` is the primitive that governance (2-of-3,
  Conflict Cards, votes) will call in Increment 4.
- **Tokens are estimated** as `ceil(chars/4)`, offline and deterministic. The budget invariant
  holds exactly against this estimator, not against a provider tokenizer.
- **Testing isolation.** Testing gets requirements, approved rationales, code from every team,
  test failures and its own thread. It never gets other teams' discussion, design notes or
  proposal rationales, and runs in `isolated` mode (no live turns).
- **First measurement, read with care.** The seeded conversation is short (≈1.4k tokens), so
  packages (≈2.7k mean) are currently larger than raw history; they also carry code and contracts
  the history lacks. The report adds 5× and 20× replays (labelled synthetic) to show the bounded
  package against growing history. Increment 5 measures this properly with the persona runner.
- Lexical hash embeddings rank "the right file" less reliably than semantic ones; decision
  pinning does not depend on them.
