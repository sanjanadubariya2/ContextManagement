"""Embedding providers.

`HashEmbedder` is the default: deterministic, offline feature hashing over
identifier-aware tokens (camelCase and snake_case are split) plus bigrams.
It captures lexical similarity only, which is enough for repeatable
context-management experiments; `VoyageEmbedder` is the semantic option.
Both produce EMBED_DIM-dimensional, L2-normalised vectors.
"""

import hashlib
import math
import re
from collections import Counter

from mpc.config import EMBED_DIM, get_settings

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "with we you i our will should can do does not".split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text):
        parts = _CAMEL.findall(word) or [word]
        for p in parts:
            p = p.lower()
            if len(p) > 1 and p not in _STOP:
                out.append(p)
    return out


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        n = int.from_bytes(h, "little")
        return n % self.dim, (1.0 if (n >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        vec = [0.0] * self.dim
        for feat, tf in feats.items():
            idx, sign = self._bucket(feat)
            vec[idx] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # empty text: any fixed unit vector keeps cosine defined
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class VoyageEmbedder:
    name = "voyage"

    def __init__(self, model: str | None = None, dim: int = EMBED_DIM):
        import voyageai  # optional extra: uv sync --extra voyage

        self.client = voyageai.Client()
        self.model = model or get_settings().voyage_model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            res = self.client.embed(
                texts[i : i + 128], model=self.model, output_dimension=self.dim
            )
            out.extend(res.embeddings)
        return out


def get_embedder():
    provider = get_settings().embed_provider
    if provider == "voyage":
        return VoyageEmbedder()
    if provider == "gemini":
        from mpc.llm.gemini_provider import GeminiEmbedder

        return GeminiEmbedder()
    return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are unit length
