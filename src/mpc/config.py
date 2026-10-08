"""Settings, read from MPC_* environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
SEED_DIR = REPO_ROOT / "seed"

EMBED_DIM = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MPC_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://mpc:mpc@localhost:5433/mpc"
    redis_url: str = "redis://localhost:6380/0"

    # auto (default): Gemini if GEMINI_API_KEY/GOOGLE_API_KEY is set, else Anthropic if
    # ANTHROPIC_API_KEY is set, else an error. Or force: gemini | anthropic | fake (tests only).
    llm_provider: str = "auto"
    gemini_model: str = "gemini-3.8-flash"
    gemini_small_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    llm_model: str = "claude-opus-5-5"  # Anthropic models
    llm_small_model: str = "claude-haiku-4-5"
    llm_effort: str = "medium"
    llm_max_tokens: int = 16000
    llm_fallbacks: bool = True

    embed_provider: str = "hash"  # hash | gemini | voyage (changing it needs `mpc init --reset`)
    voyage_model: str = "voyage-3.5"

    # Context Manager
    default_token_budget: int = 12000
    brief_token_budget: int = 2000
    score_w_sim: float = 0.5
    score_w_recency: float = 0.3
    score_w_importance: float = 0.2
    recency_lambda: float = 0.02  # per turn
    candidate_pool: int = 60
    # Candidates less similar than this are never packed, so packages don't
    # fill the budget with unrelated material just because there is room.
    min_similarity: float = 0.08
    live_turns: int = 8
    live_share: float = 0.25  # max share of free budget given to live turns
    closure_reserve: float = 0.10  # share of free budget held back for import closure
    package_cache_ttl_s: int = 3600

    # Agents (Increment 3)
    expansion_budget: int = 3000  # tokens per request_more_context call
    max_reruns: int = 1  # automatic re-runs of a stale instance
    test_timeout_s: int = 120
    test_python: str = ""  # interpreter for Testing workers' tests; default: this one


def load_env_files() -> list[Path]:
    """Export .env into the process environment so third-party SDKs see it too.

    pydantic-settings reads .env only for MPC_* fields; ANTHROPIC_API_KEY and
    VOYAGE_API_KEY must reach os.environ, where the Anthropic and Voyage SDKs
    look. Variables already set in the shell win over the file.
    """
    loaded = []
    for path in dict.fromkeys((Path.cwd() / ".env", REPO_ROOT / ".env")):
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


@lru_cache
def get_settings() -> Settings:
    load_env_files()
    return Settings()
