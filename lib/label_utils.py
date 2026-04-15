"""Utilities for parsing graph node labels and extracting symbols.

Node labels in GraphIDYOM are often comma-joined JSON objects (for example:
note tokens and optional interval tokens). A naive string split on commas is
not safe because JSON objects themselves contain commas.
"""

from __future__ import annotations

from typing import List, Optional


TERMINAL_SYMBOLS = frozenset({"END", "NO_EVENT", "START", "<END>", "<START>"})


def extract_json_objects(label: str) -> List[str]:
    """Extract JSON object substrings from a mixed label string.

    Uses brace-depth parsing so commas inside JSON objects are handled safely.
    """

    s = str(label).strip()
    if not s or s in TERMINAL_SYMBOLS:
        return []

    objs: List[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(s[start : i + 1])
                start = None
    return objs


def fallback_symbol_from_label(label: str) -> Optional[str]:
    """Best-effort symbol extraction when no TokenCodec is available."""

    s = str(label).strip()
    if not s:
        return None
    if s in TERMINAL_SYMBOLS:
        return s

    objs = extract_json_objects(s)
    if objs:
        return objs[-1]

    if "," in s:
        return s.rsplit(",", 1)[-1].strip() or None
    return s

