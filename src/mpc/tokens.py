"""Token estimation for budgeting.

The Context Manager is deterministic and runs offline, so it budgets with a
fixed estimator instead of calling a tokenizer API per item. ~4 characters
per token is conservative for English and code on current Claude tokenizers;
every budget check in the packager uses this same function, so the "never
exceeds the budget" invariant holds exactly with respect to it.
"""

import math


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
