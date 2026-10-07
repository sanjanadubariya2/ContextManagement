"""Package cache in Redis, with a reverse index from dependencies to packages.

pkg:<id>            cached package JSON
pkgdeps:<dep>       set of package ids depending on <dep>
                    dep = "key:<decision key>" | "contract:<path>" | "file:<path>"
"""

import json

import redis

from mpc.redis_store import CTX_VERSION


def store_package(r: redis.Redis, pkg_id: str, payload: dict, deps: list[str], ttl: int) -> None:
    pipe = r.pipeline()
    pipe.set(f"pkg:{pkg_id}", json.dumps(payload), ex=ttl)
    for dep in deps:
        pipe.sadd(f"pkgdeps:{dep}", pkg_id)
        pipe.expire(f"pkgdeps:{dep}", ttl)
    pipe.execute()


def load_package(r: redis.Redis, pkg_id: str) -> dict | None:
    raw = r.get(f"pkg:{pkg_id}")
    return json.loads(raw) if raw else None


def invalidate(r: redis.Redis, deps: list[str]) -> int:
    """Drop every cached package that depends on any of deps."""
    dropped = 0
    for dep in deps:
        ids = r.smembers(f"pkgdeps:{dep}")
        if ids:
            dropped += r.delete(*[f"pkg:{i}" for i in ids])
        r.delete(f"pkgdeps:{dep}")
    return dropped


def publish_version(r: redis.Redis, version: int, deps: list[str]) -> None:
    r.set(CTX_VERSION, version)
    invalidate(r, deps)
