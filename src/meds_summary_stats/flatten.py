"""Flattening of nested stats documents into ``path -> value`` maps.

Comparison operates on flat paths rather than on the nested document, because the tolerance policy is
naturally expressed as globs over paths (``codes.digests.thresholded.*.alphabetical``) and because a
flat map makes "present in one document but not the other" a first-class, checkable condition rather
than a recursive special case.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

#: Separator between path segments.
SEP = "."

#: Replacement for a separator character occurring inside a key.
#:
#: Keys in a stats document are vocabulary names, split names, and calendar years -- none of which
#: normally contain a dot, but ETL-produced split names are not guaranteed to behave.
ESCAPED_SEP = "\\."


def escape(key: str) -> str:
    r"""Escape a path segment so it cannot be mistaken for two segments.

    Examples:
        >>> escape("train"), escape("v1.2")
        ('train', 'v1\\.2')
    """
    return key.replace(SEP, ESCAPED_SEP)


def flatten(document: Any, prefix: str = "") -> dict[str, Any]:
    r"""Flatten a nested document to a map from dotted path to scalar.

    A list of scalars is kept whole as a single leaf, because every such list in a stats document
    is an unordered collection -- column names, shard names, configured quantiles. Indexing them
    positionally would make inserting one member look like a change to every member after it: add
    one column to ``layout.extra_data_columns`` and each subsequent index reports
    ``changed from 'emar_id' to 'drg_severity'``, which is twenty findings describing one event and
    never actually names the added column. Comparison diffs these leaves by membership instead.

    Lists that contain containers are still flattened by index, where position is the only
    identifier an element has. Empty containers are preserved as leaves, because "this mapping is
    empty" is itself a fact worth diffing -- an ETL that stops emitting any split at all should not
    look identical to one that never had the key.

    Args:
        document: The nested structure to flatten.
        prefix: Path prefix for recursion.

    Returns:
        A flat mapping from path to scalar value.

    Examples:
        >>> from pprint import pprint
        >>> pprint(flatten({"counts": {"n_subjects": 10, "nested": {"a": 1}}, "ok": True}))
        {'counts.n_subjects': 10, 'counts.nested.a': 1, 'ok': True}

        A list of scalars stays whole, so it can be diffed by membership:

        >>> pprint(flatten({"cols": ["a", "b"]}))
        {'cols': ['a', 'b']}

        Lists of containers still index positionally, position being their only identifier:

        >>> pprint(flatten({"rows": [{"a": 1}, {"a": 2}]}))
        {'rows.0.a': 1, 'rows.1.a': 2}

        Empty containers survive as leaves rather than vanishing:

        >>> pprint(flatten({"by_split": {}, "shards": []}))
        {'by_split': {}, 'shards': []}

        Nulls are values, not absences -- a field that went from a number to null is a change, not a
        removal:

        >>> flatten({"a": None})
        {'a': None}

        Dots inside keys are escaped so that a split literally named ``v1.2`` cannot collide with a
        nested ``v1`` mapping:

        >>> flatten({"splits": {"v1.2": 5}})
        {'splits.v1\\.2': 5}
    """
    if isinstance(document, dict) and document:
        out: dict[str, Any] = {}
        for key, value in document.items():
            path = f"{prefix}{SEP}{escape(str(key))}" if prefix else escape(str(key))
            out.update(flatten(value, path))
        return out
    if isinstance(document, list) and document:
        if all(not isinstance(value, dict | list) for value in document):
            return {prefix: document}
        out = {}
        for i, value in enumerate(document):
            out.update(flatten(value, f"{prefix}{SEP}{i}" if prefix else str(i)))
        return out
    return {prefix: document}


#: Splits on separators that are not escaped, so a key containing a dot stays one segment.
_UNESCAPED_SEP = re.compile(r"(?<!\\)" + re.escape(SEP))


def split_path(path: str) -> list[str]:
    r"""Split a flattened path into segments, honoring the escaping :func:`escape` applied.

    Splitting on every ``.`` would undo the escaping and silently break any key containing one --
    which is not hypothetical: every entry under ``dataset_digest.by_file`` is a filename ending in
    ``.parquet``, and a rule written as ``dataset_digest.by_file.*`` would match none of them.

    Args:
        path: The flattened path to split.

    Returns:
        The path's segments.

    Examples:
        >>> split_path("counts.n_subjects")
        ['counts', 'n_subjects']

        An escaped dot does not split:

        >>> split_path("dataset_digest.by_file.data/0\\.parquet")
        ['dataset_digest', 'by_file', 'data/0\\.parquet']

        Which is exactly what :func:`escape` produces, so the round trip is consistent:

        >>> split_path("splits.by_split." + escape("v1.2"))
        ['splits', 'by_split', 'v1\\.2']
    """
    return _UNESCAPED_SEP.split(path)


def matches(path: str, pattern: str) -> bool:
    """Whether ``path`` matches a glob ``pattern``.

    ``*`` matches within a single segment only, so ``codes.*`` selects the immediate children of
    ``codes`` rather than its whole subtree. Use ``**`` to cross segment boundaries. Without that
    distinction a rule intended for top-level counts would silently capture every nested quantile
    beneath them.

    Args:
        path: The dotted path to test.
        pattern: The glob to test against.

    Returns:
        True on match.

    Examples:
        >>> matches("counts.n_subjects", "counts.*")
        True
        >>> matches("counts.numeric.mean", "counts.*")
        False
        >>> matches("counts.numeric.mean", "counts.**")
        True

        A trailing ``**`` also matches the node itself, so ``codes.**`` covers ``codes`` and
        everything under it:

        >>> matches("codes", "codes.**"), matches("codes.digests.alphabetical", "codes.**")
        (True, True)

        Exact paths match themselves:

        >>> matches("counts.n_subjects", "counts.n_subjects")
        True

        Wildcards work mid-path, which is how per-key rules are written:

        >>> ontology = "code_metadata.ontology.by_vocabulary"
        >>> matches(f"{ontology}.LOINC.n_subjects", f"{ontology}.*.n_subjects")
        True
        >>> matches(f"{ontology}.LOINC.numeric.mean", f"{ontology}.*.n_subjects")
        False
    """
    if pattern.endswith(f"{SEP}**") and path == pattern[: -len(f"{SEP}**")]:
        return True
    return _match_parts(split_path(path), split_path(pattern))


def _match_parts(path_parts: list[str], pattern_parts: list[str]) -> bool:
    """Recursively match segment lists, treating ``**`` as spanning any number of segments.

    Examples:
        >>> _match_parts(["a", "b", "c"], ["a", "**"])
        True
        >>> _match_parts(["a", "b", "c"], ["a", "*"])
        False
        >>> _match_parts([], [])
        True
        >>> _match_parts(["a"], ["**"])
        True
    """
    if not pattern_parts:
        return not path_parts
    head, *rest = pattern_parts
    if head == "**":
        if not rest:
            return True
        return any(_match_parts(path_parts[i:], rest) for i in range(len(path_parts) + 1))
    if not path_parts:
        return False
    return fnmatch.fnmatchcase(path_parts[0], head) and _match_parts(path_parts[1:], rest)


def specificity(pattern: str) -> tuple[int, int, int]:
    """Rank a pattern's specificity, for resolving which of several matching rules wins.

    Ordering is by number of literal (non-wildcard) segments, then by total segment count, then by
    pattern length. A more literal pattern always beats a more wildcarded one, so a rule written for
    ``counts.n_subjects`` is not overridden by a blanket ``counts.*``.

    Args:
        pattern: The glob to rank.

    Returns:
        A sort key; larger is more specific.

    Examples:
        >>> specificity("counts.n_subjects") > specificity("counts.*")
        True
        >>> specificity("counts.*") > specificity("**")
        True
        >>> specificity("splits.by_split.*.n_subjects") > specificity("splits.by_split.**")
        True
    """
    parts = pattern.split(SEP)
    literal = sum(1 for p in parts if "*" not in p)
    return (literal, len(parts), len(pattern))
