"""Subagent templates. A template is fixed configuration; every spawn creates
a fresh instance from it that runs one task and is discarded.

Worker and Reviewer have one variant per team (write paths, skills and review
criteria differ). Researcher and Memorizer are shared templates and arrive in
Increment 4. The context mode belongs to the template and is never chosen at
runtime.
"""

from dataclasses import dataclass

from mpc.context.visibility import Mode
from mpc.llm.base import ToolSpec


@dataclass(frozen=True)
class Template:
    name: str  # backend_worker
    kind: str  # worker | reviewer
    team: str
    mode: Mode
    write_paths: tuple[str, ...]
    tools: tuple[str, ...]
    skills: str
    max_steps: int = 16
    token_budget: int = 12000


_WORKER_TOOLS = ("list_files", "read_file", "write_file", "request_more_context", "finish")
_REVIEWER_TOOLS = ("list_files", "read_file", "request_more_context", "submit_review")

_SKILLS = {
    "frontend": """You build the Next.js (pages router) TypeScript frontend under frontend/.
- Style with Tailwind utility classes; forms follow the approved forms decision.
- The frontend never reads, stores or sends auth tokens: call the API through apiFetch
  (credentials included) and learn who is logged in from /api/auth/me.
- Match the OpenAPI contract exactly: paths, status codes, problem+json errors.""",
    "backend": """You build the FastAPI backend under backend/.
- The OpenAPI contract is the source of truth: paths, status codes, schemas.
- Errors are application/problem+json with type, title, status and detail.
- Auth follows the approved decisions (mechanism, token transport, hashing, TTL);
  use the helpers in backend/app/auth/tokens.py rather than re-implementing them.""",
    "testing": """You write tests under tests/ from the requirements and the contracts only.
- You do not see implementers' reasoning, by design: test what the contract and the
  requirements promise, not what the code happens to do.
- Backend tests use pytest and FastAPI's TestClient (`from app.main import app`).
- Run your tests with run_tests. A failure that shows the implementation breaks the
  contract is a bug for the owning team: file it with report_bug.""",
}

_REVIEW_CRITERIA = {
    "frontend": "Check: no token handling in the browser, apiFetch for every call, contract "
    "paths and status codes, validation shown as the approved decisions say.",
    "backend": "Check: contract conformance (paths, codes, schemas), problem+json errors, auth "
    "per the approved decisions, no secrets in code, no unhandled exceptions that become 500s.",
    "testing": "Check: tests assert the contract and requirements (not implementation details), "
    "cover error paths, need no network or secrets, and are deterministic.",
}

TEMPLATES: dict[str, Template] = {}
for _team, _root in (("frontend", "frontend/**"), ("backend", "backend/**"), ("testing", "tests/**")):
    TEMPLATES[f"{_team}_worker"] = Template(
        name=f"{_team}_worker",
        kind="worker",
        team=_team,
        # Testing workers are isolated from implementer reasoning.
        mode="isolated" if _team == "testing" else "curated_fork",
        write_paths=(_root,),
        tools=_WORKER_TOOLS + (("run_tests", "report_bug") if _team == "testing" else ()),
        skills=_SKILLS[_team],
    )
    TEMPLATES[f"{_team}_reviewer"] = Template(
        name=f"{_team}_reviewer",
        kind="reviewer",
        team=_team,
        mode="isolated",
        write_paths=(),
        tools=_REVIEWER_TOOLS,
        skills=_SKILLS[_team] + "\n\nReview criteria: " + _REVIEW_CRITERIA[_team],
        max_steps=10,
    )


def template_for(kind: str, team: str) -> Template:
    """The team variant is a lookup of the member's team, never an LLM choice."""
    return TEMPLATES[f"{team}_{kind}"]


TOOL_SPECS: dict[str, ToolSpec] = {
    "list_files": ToolSpec(
        "list_files",
        "List current file paths in the workspace, optionally under a prefix such as 'backend/app/'.",
        {"type": "object", "properties": {"prefix": {"type": "string"}}},
    ),
    "read_file": ToolSpec(
        "read_file",
        "Read a file. Returns your own draft if you have written one, else the version your "
        "context package was built against.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    "write_file": ToolSpec(
        "write_file",
        "Write the complete new content of a file into your draft workspace. Only paths your "
        "template may write are accepted. Drafts are promoted only after validation at finish.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    "request_more_context": ToolSpec(
        "request_more_context",
        "Ask the Context Manager for more context on a topic, within your ticket's scope.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
    "finish": ToolSpec(
        "finish",
        "Finish the task. Your drafts are validated, checked for staleness and promoted. "
        "Optionally propose decisions for the team to approve (agents can only propose).",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {},
                            "rationale": {"type": "string"},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            "required": ["summary"],
        },
    ),
    "submit_review": ToolSpec(
        "submit_review",
        "Submit your review. verdict is approve or changes_requested; each comment names a path "
        "and, when it helps, a line.",
        {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["approve", "changes_requested"]},
                "summary": {"type": "string"},
                "comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "line": {"type": "integer"},
                            "body": {"type": "string"},
                        },
                        "required": ["body"],
                    },
                },
            },
            "required": ["verdict", "summary"],
        },
    ),
    "run_tests": ToolSpec(
        "run_tests",
        "Run pytest on test paths under tests/ against the current code plus your drafts, in a "
        "temporary copy with a timeout and no secrets.",
        {
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
            "required": ["paths"],
        },
    ),
    "report_bug": ToolSpec(
        "report_bug",
        "File a bug task on the owning team's board when a test shows the implementation "
        "breaking the contract or the requirements.",
        {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": ["frontend", "backend"]},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["team", "title", "description"],
        },
    ),
}
