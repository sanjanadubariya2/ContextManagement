"""Secret handling: some files are never indexed, and everything else is
scanned so secrets are redacted before anything is embedded or stored as a
chunk."""

import posixpath
import re

REDACTED = "[REDACTED]"

_NEVER_INDEX_NAMES = re.compile(r"^(\.env(\..*)?|.*\.pem|.*\.key|id_rsa.*|.*\.p12|.*\.pfx)$")
_ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}

_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
]
# key = "literal" assignments whose name looks secret. The value must be a
# quoted literal or a bare token, so code like os.environ["X"] is left alone.
_ASSIGN = re.compile(
    r"""(?ix)
    \b([A-Z0-9_]*(?:secret|password|passwd|api[_-]?key|token|private[_-]?key)[A-Z0-9_]*)
    (\s*[:=]\s*)
    (["'])([^"'\n]{8,})\3
    """
)


def never_index(path: str) -> bool:
    name = posixpath.basename(path)
    if name in _ALLOWED_ENV_TEMPLATES:
        return False
    return bool(_NEVER_INDEX_NAMES.match(name))


def redact(text: str) -> tuple[str, int]:
    """Return (redacted text, number of redactions)."""
    count = 0

    def _sub(_m: re.Match) -> str:
        nonlocal count
        count += 1
        return REDACTED

    for pat in _PATTERNS:
        text = pat.sub(_sub, text)

    def _assign(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}"

    text = _ASSIGN.sub(_assign, text)
    return text, count
