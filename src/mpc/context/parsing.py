"""File parsing for the artifacts indexer.

Python uses the standard `ast` module; TypeScript/TSX uses tree-sitter. Each
parser returns top-level symbol chunks (function, class, component, const),
the raw import specifiers, and a one-line docstring/lead comment. Imports are
resolved to artifact paths separately so dependency closure can follow them.
"""

import ast
import posixpath
import re
from dataclasses import dataclass, field

from mpc.tokens import estimate_tokens

MAX_CHUNK_TOKENS = 600


@dataclass
class ParsedChunk:
    symbol: str | None
    content: str


@dataclass
class ParsedFile:
    language: str
    imports: list[str] = field(default_factory=list)
    chunks: list[ParsedChunk] = field(default_factory=list)
    doc: str = ""
    symbols: list[str] = field(default_factory=list)


def language_for(path: str) -> str:
    ext = posixpath.splitext(path)[1].lower()
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "typescript",
        ".jsx": "tsx",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
    }.get(ext, "text")


def parse_file(path: str, content: str) -> ParsedFile:
    lang = language_for(path)
    if lang == "python":
        parsed = _parse_python(content)
    elif lang in ("typescript", "tsx"):
        parsed = _parse_typescript(content, tsx=(lang == "tsx"))
    elif lang == "markdown":
        parsed = _parse_markdown(content)
    else:
        parsed = ParsedFile(language=lang, chunks=[ParsedChunk(None, content)])
    parsed.language = lang
    parsed.chunks = [c for chunk in parsed.chunks for c in split_large(chunk)]
    parsed.chunks = [c for c in parsed.chunks if c.content.strip()]
    return parsed


def split_large(chunk: ParsedChunk, max_tokens: int = MAX_CHUNK_TOKENS) -> list[ParsedChunk]:
    if estimate_tokens(chunk.content) <= max_tokens:
        return [chunk]
    out, buf = [], []
    part = 1
    for line in chunk.content.splitlines(keepends=True):
        if buf and estimate_tokens("".join(buf) + line) > max_tokens:
            out.append(ParsedChunk(f"{chunk.symbol or 'part'}#{part}", "".join(buf)))
            buf, part = [], part + 1
        # A single over-long line is hard-wrapped so no chunk can exceed the cap.
        while estimate_tokens(line) > max_tokens:
            cut = max_tokens * 4
            out.append(ParsedChunk(f"{chunk.symbol or 'part'}#{part}", line[:cut]))
            line, part = line[cut:], part + 1
        buf.append(line)
    if buf:
        out.append(ParsedChunk(f"{chunk.symbol or 'part'}#{part}", "".join(buf)))
    return out


# ------------------------------------------------------------------ python


def _parse_python(content: str) -> ParsedFile:
    out = ParsedFile(language="python")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        out.chunks = [ParsedChunk(None, content)]
        return out
    doc_lines = (ast.get_docstring(tree) or "").strip().splitlines()
    out.doc = doc_lines[0] if doc_lines else ""
    lines = content.splitlines(keepends=True)
    header: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            out.imports.append(mod)
            # `from pkg import module` may import a submodule; keep both forms.
            out.imports.extend(f"{mod}.{a.name}" if node.module else mod + a.name for a in node.names)
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
        text = "".join(lines[start : node.end_lineno])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.symbols.append(node.name)
            out.chunks.append(ParsedChunk(node.name, text))
        else:
            header.append(text)
    if header:
        out.chunks.insert(0, ParsedChunk("<module>", "".join(header)))
    out.imports = list(dict.fromkeys(out.imports))
    return out


# -------------------------------------------------------------- typescript

_TS_LANGS: dict[bool, object] = {}
_DECL_TYPES = {
    "function_declaration",
    "generator_function_declaration",
    "class_declaration",
    "abstract_class_declaration",
    "lexical_declaration",
    "variable_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
}
_IMPORT_RE = re.compile(r"""(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"]""")


def _ts_parser(tsx: bool):
    from tree_sitter import Language, Parser
    import tree_sitter_typescript as tsts

    if tsx not in _TS_LANGS:
        raw = tsts.language_tsx() if tsx else tsts.language_typescript()
        _TS_LANGS[tsx] = Language(raw)
    return Parser(_TS_LANGS[tsx])


def _decl_name(node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode()
    for child in node.children:
        if child.type == "variable_declarator":
            n = child.child_by_field_name("name")
            if n is not None:
                return n.text.decode()
    return None


def _parse_typescript(content: str, tsx: bool) -> ParsedFile:
    out = ParsedFile(language="tsx" if tsx else "typescript")
    try:
        parser = _ts_parser(tsx)
    except Exception:  # tree-sitter unavailable: regex imports, one chunk
        out.imports = [a or b for a, b in _IMPORT_RE.findall(content)]
        out.chunks = [ParsedChunk(None, content)]
        return out

    src = content.encode()
    tree = parser.parse(src)
    header: list[str] = []
    for node in tree.root_node.children:
        text = src[node.start_byte : node.end_byte].decode()
        if node.type == "import_statement":
            spec = node.child_by_field_name("source")
            if spec is not None:
                out.imports.append(spec.text.decode().strip("'\""))
            header.append(text + "\n")
            continue
        decl = node
        if node.type == "export_statement":
            src_node = node.child_by_field_name("source")
            if src_node is not None:  # re-export: export { x } from "./y"
                out.imports.append(src_node.text.decode().strip("'\""))
            decl = node.child_by_field_name("declaration") or next(
                (c for c in node.children if c.type in _DECL_TYPES), node
            )
        if decl.type in _DECL_TYPES:
            name = _decl_name(decl) or ("default" if "default" in text[:20] else None)
            if name:
                out.symbols.append(name)
            out.chunks.append(ParsedChunk(name, text))
        elif node.type == "comment" and not out.doc and not out.chunks:
            out.doc = text.lstrip("/* ").splitlines()[0].strip()
            header.append(text + "\n")
        else:
            header.append(text + "\n")
    if "".join(header).strip():
        out.chunks.insert(0, ParsedChunk("<module>", "".join(header)))
    out.imports = list(dict.fromkeys(out.imports))
    return out


# ---------------------------------------------------------------- markdown


def _parse_markdown(content: str) -> ParsedFile:
    out = ParsedFile(language="markdown")
    sections = re.split(r"(?m)^(?=#{1,3} )", content)
    for sec in sections:
        if not sec.strip():
            continue
        title = sec.splitlines()[0].lstrip("# ").strip()
        if not out.doc:
            out.doc = title
        out.chunks.append(ParsedChunk(title, sec))
    return out


# ------------------------------------------------------- import resolution


def resolve_imports(path: str, language: str, imports: list[str], known: set[str]) -> list[str]:
    """Map raw import specifiers to paths of artifacts in the store."""
    resolved: list[str] = []
    if language == "python":
        for spec in imports:
            hit = _resolve_python(path, spec, known)
            if hit and hit != path:
                resolved.append(hit)
    elif language in ("typescript", "tsx"):
        for spec in imports:
            hit = _resolve_ts(path, spec, known)
            if hit and hit != path:
                resolved.append(hit)
    return list(dict.fromkeys(resolved))


def _resolve_python(path: str, spec: str, known: set[str]) -> str | None:
    here = posixpath.dirname(path)
    level = len(spec) - len(spec.lstrip("."))
    module = spec.lstrip(".")
    if level:
        base = here
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        roots = [base]
    else:
        roots, d = [], here
        while True:
            roots.append(d)
            if not d:
                break
            d = posixpath.dirname(d)
        # Tests import the backend as a top-level package (tests/ -> backend/app).
        roots += sorted({p.split("/", 1)[0] for p in known if "/" in p} - set(roots))
    rel = module.replace(".", "/")
    for root in roots:
        stem = posixpath.join(root, rel) if rel else root
        for cand in (stem + ".py", posixpath.join(stem, "__init__.py")):
            cand = cand.lstrip("/")
            if cand in known:
                return cand
    return None


def _resolve_ts(path: str, spec: str, known: set[str]) -> str | None:
    if spec.startswith("@/"):
        team_root = path.split("/", 1)[0]
        base = posixpath.join(team_root, "src", spec[2:])
    elif spec.startswith("."):
        base = posixpath.normpath(posixpath.join(posixpath.dirname(path), spec))
    else:
        return None  # package import
    for suffix in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx"):
        if base + suffix in known:
            return base + suffix
    return None


def summarize(path: str, parsed: ParsedFile, resolved: list[str]) -> str:
    parts = [path]
    if parsed.doc:
        parts.append(f"— {parsed.doc.rstrip('.')}.")
    if parsed.symbols:
        parts.append(f"Defines: {', '.join(parsed.symbols[:12])}.")
    if resolved:
        parts.append(f"Imports: {', '.join(resolved)}.")
    return " ".join(parts)
