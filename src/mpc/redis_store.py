"""Redis access. Key ownership follows the plan:

Context Manager: ctx_version, pkg:*, pkgdeps:*, thread:*
Policy guard (Increment 4): claim:*, lease:*, count:*  -- read-only here.
"""

import json
from functools import lru_cache

import redis

from mpc.config import get_settings

CTX_VERSION = "ctx_version"
THREAD_MAX = 50


@lru_cache
def get_redis(url: str | None = None) -> redis.Redis:
    return redis.Redis.from_url(url or get_settings().redis_url, decode_responses=True)


def ping(r: redis.Redis | None = None) -> bool:
    try:
        return bool((r or get_redis()).ping())
    except Exception:
        return False


def thread_key(member_id: str) -> str:
    return f"thread:{member_id}"


def push_turn(r: redis.Redis, member_id: str, role: str, content: str, turn: int) -> None:
    key = thread_key(member_id)
    r.rpush(key, json.dumps({"role": role, "content": content, "turn": turn}))
    r.ltrim(key, -THREAD_MAX, -1)


def recent_turns(r: redis.Redis, member_id: str, n: int) -> list[dict]:
    return [json.loads(x) for x in r.lrange(thread_key(member_id), -n, -1)]


def current_leases(r: redis.Redis) -> dict[str, str]:
    """File leases held by the policy guard (empty until Increment 4)."""
    out = {}
    for key in r.scan_iter("lease:*"):
        out[key.removeprefix("lease:")] = r.get(key) or ""
    return out
