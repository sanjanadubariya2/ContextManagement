"""Local test runner for Testing workers.

Tests run in a separate process with a timeout, on a temporary copy of the
code (current versions plus the instance's drafts), with no secrets in the
environment: only an allowlist of OS variables is passed through, so API keys
and database URLs never reach generated code. Secret files are not copied.
"""

import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from mpc.config import get_settings
from mpc.context.secrets import never_index
from mpc.models import Artifact

# OS plumbing a Python process needs; nothing else is inherited.
SAFE_ENV = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
            "TMPDIR", "LANG", "LC_ALL", "HOME", "USERPROFILE")
OUTPUT_TAIL = 4000


@dataclass
class TestResult:
    __test__ = False  # not a pytest class

    paths: list[str]
    exit_code: int | None
    passed: int
    failed: int
    errors: int
    timed_out: bool
    duration_ms: int
    output: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        if self.timed_out:
            return f"timed out after {self.duration_ms} ms"
        return f"{self.passed} passed, {self.failed} failed, {self.errors} errors (exit {self.exit_code})"


class TestPathError(ValueError):
    __test__ = False


def check_paths(paths: list[str]) -> list[str]:
    clean = []
    for p in paths:
        norm = posixpath.normpath(p.replace("\\", "/"))
        if norm.startswith("..") or posixpath.isabs(norm) or not norm.startswith("tests"):
            raise TestPathError(f"test paths must be under tests/: {p!r}")
        clean.append(norm)
    return clean or ["tests"]


def materialize(session: Session, root: Path, overlay: dict[str, str] | None = None) -> int:
    """Write current file versions plus overlay drafts under root; secret files are skipped."""
    files = {
        a.path: a.content
        for a in session.scalars(select(Artifact).where(Artifact.status == "current"))
    }
    files.update(overlay or {})
    n = 0
    for path, content in files.items():
        if never_index(path):
            continue
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        n += 1
    return n


def _count(pattern: str, text: str) -> int:
    m = re.findall(pattern, text)
    return int(m[-1]) if m else 0


def run_tests(
    session: Session,
    paths: list[str],
    *,
    overlay: dict[str, str] | None = None,
    timeout_s: int | None = None,
    python: str | None = None,
) -> TestResult:
    s = get_settings()
    paths = check_paths(paths)
    timeout_s = timeout_s or s.test_timeout_s
    tmp = Path(tempfile.mkdtemp(prefix="mpc-tests-"))
    try:
        materialize(session, tmp, overlay)
        env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
        env["PYTHONPATH"] = str(tmp / "backend")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [python or s.test_python or sys.executable, "-m", "pytest", "-q", "-p",
               "no:cacheprovider", "--rootdir", str(tmp), *paths]
        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=timeout_s)
            out, code, timed_out = proc.stdout + proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            code, timed_out = None, True
        ms = int((time.monotonic() - start) * 1000)
        tail = out[-OUTPUT_TAIL:].replace(str(tmp), "<tmp>")
        return TestResult(
            paths=paths,
            exit_code=code,
            passed=_count(r"(\d+) passed", out),
            failed=_count(r"(\d+) failed", out),
            errors=_count(r"(\d+) errors?\b", out),
            timed_out=timed_out,
            duration_ms=ms,
            output=tail,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
