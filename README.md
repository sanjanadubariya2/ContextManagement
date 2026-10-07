# MultiPlayer-Context

Nine developers in three teams, each with a personal agent, building one website while a
central Context Manager decides what every agent knows. This repository implements
**Increment 1 (foundations)** and **Increment 2 (Context Manager core)** of the plan. Both
run headless and are driven from the `mpc` CLI.

## Setup

```sh
docker compose up -d          # Postgres 17 + pgvector on :5433, Redis 7 on :6380
uv sync
uv run mpc init               # migrate, seed 3 teams / 9 members / starter project, Context Repo -> v14
uv run mpc status
```

`mpc init --reset` wipes and reseeds. Settings come from `MPC_*` environment variables or
`.env` (see `.env.example`).

**LLM.** With `MPC_LLM_PROVIDER=auto` (default), the Anthropic adapter is used when credentials
exist (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile). Otherwise a
deterministic offline `fake` provider is used. The default model is `claude-opus-5-5` with effort
`medium` and server-side refusal fallbacks on. The small model for Researcher/Memorizer is
`claude-haiku-4-5`. Which provider and budget to use is still the plan's open question; the
adapter (`complete`, `tool_call`, `embed`) keeps that swappable.

**Embeddings.** Default `hash`: deterministic, offline, lexical feature hashing, so experiments
are repeatable and cost nothing. `MPC_EMBED_PROVIDER=voyage` (`uv sync --extra voyage`) gives
semantic embeddings. Both produce 1024-d vectors.

## Commands

| Command | What it does |
|---|---|
| `mpc ask sneha "..."` | One turn with Sneha's personal agent through the adapter; conversation in Postgres, recent turns in Redis, orientation brief in the system prompt |
| `mpc context sneha "implement login API" --task t20` | Build a context package and print it with its trace (`--show` renders it, `--json` dumps it, `--mode`, `--budget`) |
| `mpc brief aarav "..."` / `mpc impact aarav "..."` | Orientation brief / impact analysis |
| `mpc trace 12` / `mpc stale 12` | Inspect a stored trace / staleness-check it against the current repo |
| `mpc decision keys \| list \| propose \| approve \| reject \| history \| snapshot` | Decision registry |
| `mpc artifact list \| show \| put` | Versioned artifact store (`put` writes a draft and promotes it) |
| `mpc measure` | Increment 2 measurement → `results/increment2_package_size.{md,csv}` |
| `mpc members`, `mpc index`, `mpc migrate` | Housekeeping |

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
never returned:

```sh
uv run pytest      # 82 tests; DB tests use the separate mpc_test database and Redis db 1
```

## Deliberate choices and limits

- **Tickets are drafts.** The policy guard that issues admitted tickets is Increment 4. Until then
  the CLI builds an unadmitted `draft_ticket` with the same shape, and the Context Manager
  enforces its visibility, budget and expiry.
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
