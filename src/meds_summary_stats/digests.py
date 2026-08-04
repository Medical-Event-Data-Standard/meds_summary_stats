r"""Canonical, reproducible digests over string sequences.

Digests are how this package reports on a dataset's *vocabulary* without ever emitting the vocabulary
itself. Two extractions of the same dataset must produce byte-identical digests, so the encoding here is
pinned deliberately and should be treated as part of the on-disk stats format: changing it is a breaking
change to :data:`meds_summary_stats.extract.STATS_SCHEMA_VERSION`.

The canonical encoding is UTF-8, one item per line, each line terminated by ``\n`` (including the last).
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DIGEST_PREFIX = "sha256:"


def digest(items: Iterable[str]) -> str:
    r"""Return the canonical digest of ``items``, in the order given.

    Order is significant: this function does not sort. Use :func:`ordered_by` to build the ordering.

    Args:
        items: The strings to digest. Consumed exactly once.

    Returns:
        A ``"sha256:<hex>"`` string.

    Examples:
        >>> digest(["a", "b"])
        'sha256:911169ddaaf146aff539f58c26c489af3b892dff0fe283c1c264c65ae5aa59a2'
        >>> digest(["b", "a"]) == digest(["a", "b"])
        False

        The empty sequence has a well-defined digest, so an empty vocabulary is still comparable:

        >>> digest([])
        'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

        Digests are stable across call styles because the encoding is pinned:

        >>> digest(iter(["a", "b"])) == digest(["a", "b"])
        True

        Newlines inside an item would be ambiguous against the line separator, so they are rejected:

        >>> digest(["a\nb"])
        Traceback (most recent call last):
            ...
        ValueError: Digest items may not contain newlines; got 'a\nb'
    """
    h = hashlib.sha256()
    for item in items:
        if "\n" in item:
            raise ValueError(f"Digest items may not contain newlines; got {item!r}")
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return f"{DIGEST_PREFIX}{h.hexdigest()}"


def ordered_by(codes: Sequence[str], weights: Sequence[float]) -> list[str]:
    """Order ``codes`` by descending ``weight``, breaking ties alphabetically.

    The alphabetical tie-break is what makes frequency-ordered digests reproducible: without it, two
    equally frequent codes could appear in either order depending on how the group-by happened to shard.

    Args:
        codes: The codes to order.
        weights: A parallel sequence of weights (typically occurrence or subject counts).

    Returns:
        The codes, sorted by descending weight then ascending code.

    Raises:
        ValueError: If ``codes`` and ``weights`` have different lengths.

    Examples:
        >>> ordered_by(["a", "b", "c"], [1, 5, 3])
        ['b', 'c', 'a']

        Ties fall back to alphabetical order, regardless of input order:

        >>> ordered_by(["z", "a", "m"], [2, 2, 2])
        ['a', 'm', 'z']
        >>> ordered_by(["a", "m", "z"], [2, 2, 2])
        ['a', 'm', 'z']

        Nulls are read as zero weight, so codes with no measurements sort last:

        >>> ordered_by(["a", "b"], [None, 1])
        ['b', 'a']

        >>> ordered_by(["a"], [1, 2])
        Traceback (most recent call last):
            ...
        ValueError: codes and weights must be the same length; got 1 and 2
    """
    if len(codes) != len(weights):
        raise ValueError(f"codes and weights must be the same length; got {len(codes)} and {len(weights)}")
    return [c for _, c in sorted(zip((-(w or 0) for w in weights), codes, strict=True))]


def format_moment(value: float | None, sig_digits: int) -> str:
    """Render a float for inclusion in a moments digest, at fixed significant-digit precision.

    Rounding is what makes a moments digest useful rather than a random-number generator: floating point
    summation order varies with shard count and thread count, so raw means are not bit-reproducible even
    for identical data. Rounding to a handful of significant digits absorbs that jitter while still
    tripping on any real distributional shift.

    Args:
        value: The value to format. ``None`` and non-finite values get distinct sentinel spellings.
        sig_digits: Number of significant digits to retain.

    Returns:
        A short, stable string.

    Examples:
        >>> format_moment(123.456789, 6)
        '123.457'
        >>> format_moment(123.456789, 3)
        '123.0'

        Values that differ only below the retained precision collapse to the same string, which is the
        entire point:

        >>> format_moment(1.0000001, 6) == format_moment(1.0000002, 6)
        True
        >>> format_moment(1.0001, 6) == format_moment(1.0002, 6)
        False

        Zero, nulls, and non-finite values each get an unambiguous spelling:

        >>> [format_moment(v, 6) for v in (0.0, None, float("nan"), float("inf"), float("-inf"))]
        ['0.0', 'null', 'nan', 'inf', '-inf']

        Negative and very small magnitudes round on significant digits, not decimal places:

        >>> format_moment(-0.000123456789, 4)
        '-0.0001235'
    """
    if value is None:
        return "null"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0.0"
    exponent = math.floor(math.log10(abs(value)))
    return repr(round(value, sig_digits - 1 - exponent))
