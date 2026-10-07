"""Pure packing core: scoring, greedy packing under a token budget, and
dependency closure. No database or Redis here, so the invariants are unit
tested directly:

* pinned items are always included, never scored; if they alone exceed the
  budget the task is too big and must be split;
* the package never exceeds the budget;
* candidates are ranked by Score = w1*sim + w2*exp(-lambda*dt) + w3*I.
"""

import math
from dataclasses import dataclass, field
from typing import Callable
from xml.sax.saxutils import escape, quoteattr

from mpc.tokens import estimate_tokens


class TaskTooBig(Exception):
    def __init__(self, pinned_tokens: int, budget: int, refs: list[str]):
        self.pinned_tokens = pinned_tokens
        self.budget = budget
        self.refs = refs
        super().__init__(
            f"pinned context needs {pinned_tokens} tokens but the budget is {budget}; split the task"
        )


@dataclass
class Item:
    ref: str
    kind: str
    section: str  # pinned | live | retrieved | closure
    content: str
    author: str = ""
    status: str = ""
    path: str | None = None
    score: float | None = None
    sim: float | None = None
    reason: str = ""
    tokens: int = 0

    def render(self) -> str:
        # Retrieved content is wrapped as data, separate from instructions,
        # and every item carries provenance (author, status).
        attrs = " ".join(
            f"{k}={quoteattr(str(v))}"
            for k, v in (
                ("ref", self.ref),
                ("kind", self.kind),
                ("section", self.section),
                ("author", self.author),
                ("status", self.status),
            )
            if v
        )
        return f"<item {attrs}>\n{escape(self.content)}\n</item>\n"

    def measure(self) -> "Item":
        self.tokens = estimate_tokens(self.render())
        return self


@dataclass
class Candidate:
    item: Item
    sim: float
    age_turns: int
    important: bool


@dataclass
class Weights:
    w_sim: float = 0.5
    w_recency: float = 0.3
    w_importance: float = 0.2
    lam: float = 0.02


def score(sim: float, age_turns: int, important: bool, w: Weights) -> float:
    importance = 1.5 if important else 1.0
    return w.w_sim * sim + w.w_recency * math.exp(-w.lam * max(0, age_turns)) + w.w_importance * importance


@dataclass
class Dropped:
    ref: str
    tokens: int
    score: float | None
    reason: str


@dataclass
class PackResult:
    items: list[Item]
    dropped: list[Dropped]
    total_tokens: int
    pinned_tokens: int
    header_tokens: int
    by_section: dict[str, int] = field(default_factory=dict)


def pack(
    *,
    budget: int,
    header_tokens: int,
    pinned: list[Item],
    live: list[Item],
    candidates: list[Candidate],
    weights: Weights,
    closure: Callable[[list[Item]], list[Item]] | None = None,
    live_share: float = 0.25,
    closure_reserve: float = 0.10,
    min_similarity: float = 0.0,
) -> PackResult:
    for it in [*pinned, *live, *(c.item for c in candidates)]:
        it.measure()

    pinned_tokens = sum(i.tokens for i in pinned)
    used = header_tokens + pinned_tokens
    if used > budget:
        raise TaskTooBig(used, budget, [i.ref for i in pinned])

    items = list(pinned)
    dropped: list[Dropped] = []
    refs = {i.ref for i in items}

    # Live turns: most recent first, capped at a share of the free budget,
    # then restored to chronological order.
    live_cap = used + int((budget - used) * live_share)
    kept_live: list[Item] = []
    for it in reversed(live):
        if used + it.tokens <= live_cap:
            kept_live.append(it)
            used += it.tokens
        else:
            dropped.append(Dropped(it.ref, it.tokens, None, "live share exhausted"))
    items.extend(reversed(kept_live))

    # Scored candidates, greedy by score; some budget is held back for closure.
    retrieval_cap = used + int((budget - used) * (1 - closure_reserve))
    for c in candidates:
        c.item.sim = c.sim
        c.item.score = round(score(c.sim, c.age_turns, c.important, weights), 4)
    ranked = sorted(candidates, key=lambda c: (-(c.item.score or 0), c.item.ref))
    for c in ranked:
        it = c.item
        if it.ref in refs:
            continue
        if c.sim < min_similarity:
            dropped.append(Dropped(it.ref, it.tokens, it.score, "below similarity floor"))
            continue
        if used + it.tokens <= retrieval_cap:
            items.append(it)
            refs.add(it.ref)
            used += it.tokens
        else:
            dropped.append(Dropped(it.ref, it.tokens, it.score, "over budget"))

    # Dependency closure: direct imports of selected code, if they still fit.
    if closure is not None:
        for it in closure(items):
            it.measure()
            if it.ref in refs:
                continue
            if used + it.tokens <= budget:
                items.append(it)
                refs.add(it.ref)
                used += it.tokens
            else:
                dropped.append(Dropped(it.ref, it.tokens, None, "closure over budget"))

    by_section: dict[str, int] = {}
    for it in items:
        by_section[it.section] = by_section.get(it.section, 0) + it.tokens
    assert used <= budget, "packer exceeded its budget"
    return PackResult(items, dropped, used, pinned_tokens, header_tokens, by_section)
