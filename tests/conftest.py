"""Integration tests run against a separate database (mpc_test) and Redis db 1,
which are wiped and reseeded once per test session. Tests that need them skip
when the docker compose stores are not running."""

import os

os.environ["MPC_DATABASE_URL"] = os.environ.get(
    "MPC_TEST_DATABASE_URL", "postgresql+psycopg://mpc:mpc@localhost:5433/mpc_test"
)
os.environ["MPC_REDIS_URL"] = os.environ.get("MPC_TEST_REDIS_URL", "redis://localhost:6380/1")
os.environ["MPC_LLM_PROVIDER"] = "fake"
os.environ["MPC_EMBED_PROVIDER"] = "hash"

import pytest  # noqa: E402

from mpc import db, redis_store, workspace  # noqa: E402
from mpc.llm.embeddings import HashEmbedder  # noqa: E402


@pytest.fixture(scope="session")
def ws():
    """A freshly seeded workspace: (redis client, embedder)."""
    if not db.ping():
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    r = redis_store.get_redis()
    if not redis_store.ping(r):
        pytest.skip("Redis not reachable; run `docker compose up -d`")
    workspace.reset(r=r)
    workspace.migrate()
    emb = HashEmbedder()
    with db.session_scope() as s:
        workspace.seed(s, r, emb)
    return r, emb
