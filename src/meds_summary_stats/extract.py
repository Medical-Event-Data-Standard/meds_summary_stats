"""Extraction of a privacy-safe summary fingerprint from a MEDS dataset.

The output is a single JSON document whose every leaf is a scalar, so it flattens cleanly to
``path -> value`` for comparison. Nothing in it identifies a subject, and -- by construction -- nothing
in it republishes the dataset's code vocabulary: codes are reported only through digests, aggregate
distributions, and namespace/ontology rollups that are suppressed below a subject-count floor.

The data is aggregated in streaming passes over chunks of a bounded number of rows, which never split
a subject and whose results combine exactly, so peak memory does not grow with the size of the
dataset. The only frames materialized whole are per-subject and per-code, both of which are orders
of magnitude smaller than the data itself.
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
from pathlib import Path
from typing import Any

import meds
import polars as pl

from . import __version__
from .config import NO_VOCABULARY, SUPPRESSED, ExtractConfig
from .digests import combine_digests, digest, file_digest, format_moment, ordered_by
from .ontology import is_recognized, key_expr, vocabulary_expr

#: Version of the emitted stats document format.
#:
#: Bumped whenever the meaning of an existing field changes or a digest encoding changes. Comparison
#: refuses to diff across different values.
STATS_SCHEMA_VERSION = 1

#: Fractions of total measurement volume for which vocabulary-concentration counts are reported.
_CONCENTRATION_LEVELS: tuple[float, ...] = (0.5, 0.9, 0.99)

#: Vocabulary head sizes for which cumulative measurement share is reported.
_TOP_K_LEVELS: tuple[int, ...] = (10, 100, 1000)

#: Default row budget for one aggregation pass; see ``_plan_chunks``.
#:
#: Peak memory is bounded by the chunk rather than the dataset. Measured on a synthetic ICU-shaped
#: cohort of 256M rows, this budget peaks under 2 GB resident and is as fast as any larger one;
#: bigger chunks were measured both slower and hungrier.
DEFAULT_MAX_ROWS_PER_CHUNK = 10_000_000

_REQUIRED_DATA_COLUMNS = ("subject_id", "code")
_CORE_DATA_COLUMNS = ("subject_id", "time", "code", "numeric_value")


class NotAMEDSDatasetError(ValueError):
    """Raised when the target directory has no readable MEDS data shards."""


def _round_sig(value: float | None, sig_digits: int = 6) -> float | None:
    """Round a float to a fixed number of significant digits, passing through null and non-finite.

    Emitted floats are rounded so that irrelevant last-bit jitter -- which varies with shard count and
    thread scheduling -- does not show up as a diff.

    Examples:
        >>> _round_sig(123.4567891), _round_sig(0.000123456789), _round_sig(0.0)
        (123.457, 0.000123457, 0.0)
        >>> _round_sig(None) is None
        True
        >>> _round_sig(float("nan"))
        nan
    """
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value == 0:
        return value
    return round(value, sig_digits - 1 - math.floor(math.log10(abs(value))))


def _iso(value: datetime.datetime | None) -> str | None:
    """Render a datetime as an ISO 8601 string, passing through null.

    Examples:
        >>> _iso(datetime(2020, 1, 2, 3, 4, 5))
        '2020-01-02T03:04:05'
        >>> _iso(None) is None
        True
    """
    return None if value is None else value.isoformat()


def _qkey(q: float) -> str:
    """Render a quantile as its stats-document key.

    Percentile spelling rather than the raw fraction, because a key of ``0.5`` would contain the
    path separator and force escaping into every policy rule and every report line that mentions a
    quantile. Fractional percentiles use an underscore for the same reason.

    Examples:
        >>> [_qkey(q) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
        ['p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99']
        >>> [_qkey(q) for q in (0.001, 0.999, 0.125)]
        ['p0_1', 'p99_9', 'p12_5']

        No key contains a path separator, at any precision:

        >>> any("." in _qkey(q) for q in (0.001, 0.0125, 0.5, 0.9999))
        False
    """
    return f"p{q * 100:g}".replace(".", "_")


def _distribution(series: pl.Series, cfg: ExtractConfig) -> dict[str, Any]:
    """Summarize a numeric series as counts, moments, and quantiles.

    Below ``cfg.min_numeric_cell_size`` non-null values, only the count is emitted: quantiles over a
    handful of observations are close to reporting the observations themselves.

    Args:
        series: The values to summarize. Nulls are excluded from every statistic.
        cfg: Supplies the suppression floor and the quantiles to report.

    Returns:
        A dict with ``n`` and, when not suppressed, ``mean``, ``std``, and ``quantiles``.

    Examples:
        >>> cfg = ExtractConfig(min_numeric_cell_size=3, quantiles=(0.5,))
        >>> _distribution(pl.Series([1.0, 2.0, 3.0, None]), cfg)
        {'n': 3, 'mean': 2.0, 'std': 1.0, 'quantiles': {'p50': 2.0}}

        Below the floor, the summary collapses to a bare count plus an explicit marker, so a reader
        can tell "suppressed" from "absent":

        >>> _distribution(pl.Series([1.0, 2.0]), cfg)
        {'n': 2, 'suppressed': True}

        An all-null series is reported as empty rather than as a wall of nulls:

        >>> _distribution(pl.Series([None, None], dtype=pl.Float64), cfg)
        {'n': 0, 'suppressed': True}
    """
    n = series.len() - series.null_count()
    if n < cfg.min_numeric_cell_size:
        return {"n": n, "suppressed": True}
    return {
        "n": n,
        "mean": _round_sig(series.mean()),
        "std": _round_sig(series.std()),
        "quantiles": {_qkey(q): _round_sig(series.quantile(q, "linear")) for q in cfg.quantiles},
    }


def _concentration(counts: list[int]) -> dict[str, Any]:
    """Describe how concentrated measurement volume is across the vocabulary.

    A healthy ETL has a stable head-vs-tail profile. A mapping regression that collapses many distinct
    codes into one, or explodes one code into many, moves these numbers sharply even when the total
    measurement count is unchanged.

    Args:
        counts: Per-code measurement counts, in any order.

    Returns:
        Cumulative share for a few head sizes, and the number of codes needed to cover a few volume
        fractions.

    Examples:
        >>> from pprint import pprint
        >>> pprint(_concentration([100, 1, 1, 1]))
        {'frac_measurements_in_top_10': 1.0,
         'frac_measurements_in_top_100': 1.0,
         'frac_measurements_in_top_1000': 1.0,
         'n_codes_covering_50pct': 1,
         'n_codes_covering_90pct': 1,
         'n_codes_covering_99pct': 3}

        A flat vocabulary needs proportionally more codes to cover the same volume:

        >>> pprint(_concentration([1] * 100))
        {'frac_measurements_in_top_10': 0.1,
         'frac_measurements_in_top_100': 1.0,
         'frac_measurements_in_top_1000': 1.0,
         'n_codes_covering_50pct': 50,
         'n_codes_covering_90pct': 90,
         'n_codes_covering_99pct': 99}

        An empty vocabulary yields nulls rather than a division by zero:

        >>> _concentration([])["frac_measurements_in_top_10"] is None
        True
    """
    total = sum(counts)
    out: dict[str, Any] = {}
    if not total:
        for k in _TOP_K_LEVELS:
            out[f"frac_measurements_in_top_{k}"] = None
        for level in _CONCENTRATION_LEVELS:
            out[f"n_codes_covering_{level:.0%}".replace("%", "pct")] = None
        return dict(sorted(out.items()))

    descending = sorted(counts, reverse=True)
    cumulative: list[int] = []
    running = 0
    for c in descending:
        running += c
        cumulative.append(running)

    for k in _TOP_K_LEVELS:
        head = cumulative[min(k, len(cumulative)) - 1]
        out[f"frac_measurements_in_top_{k}"] = _round_sig(head / total)
    for level in _CONCENTRATION_LEVELS:
        target = level * total
        n_needed = next(i + 1 for i, c in enumerate(cumulative) if c >= target)
        out[f"n_codes_covering_{level:.0%}".replace("%", "pct")] = n_needed
    return dict(sorted(out.items()))


def _finite_numeric() -> pl.Expr:
    """Expression selecting ``numeric_value`` where it is present and finite, else null.

    Moments must exclude NaN and infinity: in polars (as in numpy) a single NaN propagates through
    ``mean``, so one bad row would otherwise erase the statistic for an entire code.

    Examples:
        >>> df = pl.DataFrame({"numeric_value": [1.0, float("nan"), float("inf"), None, 3.0]})
        >>> df.select(mean=_finite_numeric().mean(), n=_finite_numeric().count()).to_dicts()
        [{'mean': 2.0, 'n': 2}]
    """
    nv = pl.col("numeric_value")
    return pl.when(nv.is_not_null() & nv.is_finite()).then(nv).otherwise(None)


def _suppress_keyed(
    rows: list[dict[str, Any]],
    key: str,
    cfg: ExtractConfig,
    sum_fields: tuple[str, ...],
    count_name: str,
) -> dict[str, Any]:
    """Drop keyed entries backed by too few subjects, rolling them into a single aggregate.

    The suppressed entries' keys are never emitted -- only how many there were and their pooled
    volume. Pooled subject counts are deliberately omitted, because summing distinct-subject counts
    across keys double-counts and would read as a larger cohort than exists.

    Args:
        rows: One dict per keyed entry. Each must carry ``key`` and ``n_subjects``.
        key: The field naming the entry.
        cfg: Supplies ``min_cell_size``.
        sum_fields: Numeric fields to total across suppressed entries.
        count_name: Name for the count of suppressed entries in the rollup.

    Returns:
        A mapping from surviving key to its stats, plus a ``<suppressed>`` entry when anything was
        dropped.

    Examples:
        >>> from pprint import pprint
        >>> cfg = ExtractConfig(min_cell_size=10)
        >>> rows = [
        ...     {"split": "train", "n_subjects": 800, "n_measurements": 9000},
        ...     {"split": "tuning", "n_subjects": 100, "n_measurements": 1000},
        ...     {"split": "pilot", "n_subjects": 3, "n_measurements": 40},
        ... ]
        >>> pprint(_suppress_keyed(rows, "split", cfg, ("n_measurements",), "n_splits"))
        {'<suppressed>': {'n_measurements': 40, 'n_splits': 1},
         'train': {'n_measurements': 9000, 'n_subjects': 800},
         'tuning': {'n_measurements': 1000, 'n_subjects': 100}}

        With nothing below the floor, no rollup entry is emitted at all:

        >>> pprint(_suppress_keyed(rows[:2], "split", cfg, ("n_measurements",), "n_splits"))
        {'train': {'n_measurements': 9000, 'n_subjects': 800},
         'tuning': {'n_measurements': 1000, 'n_subjects': 100}}

        A key colliding with the sentinel would make the rollup ambiguous, so it is rejected:

        >>> _suppress_keyed([{"split": SUPPRESSED, "n_subjects": 99}], "split", cfg, (), "n_splits")
        Traceback (most recent call last):
            ...
        ValueError: Key '<suppressed>' collides with the suppression sentinel
    """
    kept: dict[str, Any] = {}
    rollup: dict[str, Any] = {count_name: 0, **dict.fromkeys(sum_fields, 0)}
    n_suppressed = 0

    for row in rows:
        name = row[key]
        if name == SUPPRESSED:
            raise ValueError(f"Key {SUPPRESSED!r} collides with the suppression sentinel")
        if (row.get("n_subjects") or 0) < cfg.min_cell_size:
            n_suppressed += 1
            for field in sum_fields:
                rollup[field] += row.get(field) or 0
            continue
        kept[name] = {k: v for k, v in sorted(row.items()) if k != key}

    if n_suppressed:
        rollup[count_name] = n_suppressed
        kept[SUPPRESSED] = dict(sorted(rollup.items()))
    return dict(sorted(kept.items()))


def _read_dataset_metadata(meds_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Read ``metadata/dataset.json``, tolerating its absence.

    Returns:
        A ``(present, contents)`` pair. Contents are passed through verbatim: every field in the MEDS
        dataset metadata schema is dataset-level provenance, not subject data.

    Examples:
        >>> with yaml_disk({"metadata": {"dataset.json": {"dataset_name": "x", "etl_version": "1.2"}}}) as d:
        ...     _read_dataset_metadata(d)
        (True, {'dataset_name': 'x', 'etl_version': '1.2'})

        >>> with yaml_disk({"data": {}}) as d:
        ...     _read_dataset_metadata(d)
        (False, {})
    """
    fp = meds_dir / meds.dataset_metadata_filepath
    if not fp.is_file():
        return (False, {})
    return (True, json.loads(fp.read_text()))


def _shard_paths(meds_dir: Path) -> list[Path]:
    """Locate the dataset's data shards.

    Args:
        meds_dir: The MEDS dataset root.

    Returns:
        Sorted shard paths.

    Raises:
        NotAMEDSDatasetError: If no shards exist under ``data/``.

    Examples:
        >>> spec = {"data": {"train": {"0.parquet": {"subject_id": [1]}},
        ...                  "held_out": {"0.parquet": {"subject_id": [2]}}}}
        >>> with yaml_disk(spec) as d:
        ...     [p.relative_to(d).as_posix() for p in _shard_paths(d)]
        ['data/held_out/0.parquet', 'data/train/0.parquet']

        A directory with no shards is an error rather than an empty result, because an empty
        fingerprint would silently "pass" every comparison:

        >>> with yaml_disk({"metadata": {"dataset.json": {}}}) as d:
        ...     _shard_paths(d)
        Traceback (most recent call last):
            ...
        meds_summary_stats.extract.NotAMEDSDatasetError: No MEDS data shards found under .../data
    """
    data_dir = meds_dir / meds.data_subdirectory
    shards = sorted(data_dir.rglob("*.parquet"))
    if not shards:
        raise NotAMEDSDatasetError(f"No MEDS data shards found under {data_dir}")
    return shards


def _shard_name(meds_dir: Path, shard: Path) -> str:
    """Render a shard's identity relative to the dataset root, without its suffix.

    Examples:
        >>> _shard_name(Path("/x"), Path("/x/data/train/0.parquet"))
        'train/0'
    """
    return shard.relative_to(meds_dir / meds.data_subdirectory).with_suffix("").as_posix()


def _layout(meds_dir: Path, shards: list[Path]) -> dict[str, Any]:
    """Describe the dataset's physical layout and column schema.

    Schema drift is the cheapest regression to detect and among the most damaging to miss, so column
    names and Arrow dtypes are captured per shard and reported both as a union and as a per-column
    count of the shards carrying them. A column present in only some shards is a real ETL bug that a
    union-only view would hide.

    Examples:
        >>> from pprint import pprint
        >>> spec = {"data": {
        ...     "0.parquet": {"subject_id": [1], "code": ["A"], "numeric_value": [1.0]},
        ...     "1.parquet": {"subject_id": [2], "code": ["B"]},
        ... }}
        >>> with yaml_disk(spec) as d:
        ...     pprint(_layout(d, _shard_paths(d)))
        {'code_metadata_columns': None,
         'column_dtype_conflicts': {},
         'column_presence': {'code': 2, 'numeric_value': 1, 'subject_id': 2},
         'data_columns': {'code': 'String', 'numeric_value': 'Float64', 'subject_id': 'Int64'},
         'extra_data_columns': [],
         'has_code_metadata': False,
         'has_dataset_metadata': False,
         'has_subject_splits': False,
         'missing_core_columns': ['time'],
         'n_data_shards': 2,
         'shard_names': ['0', '1'],
         'shard_names_digest': 'sha256:...'}

        A column typed differently in different shards is reported rather than resolved silently.
        An ETL that half-migrated a dtype produces exactly this, and it is invisible in any
        single-shard view:

        >>> spec = {"data": {
        ...     "0.parquet": {"subject_id": [1], "code": ["A"]},
        ...     "1.parquet": pa.table({"subject_id": pa.array([2], pa.int32()),
        ...                            "code": pa.array(["B"])}),
        ... }}
        >>> with yaml_disk(spec) as d:
        ...     layout = _layout(d, _shard_paths(d))
        >>> layout["column_dtype_conflicts"]
        {'subject_id': ['Int32', 'Int64']}
        >>> layout["data_columns"]["subject_id"]
        'Int64'
    """
    presence: dict[str, int] = {}
    observed: dict[str, set[str]] = {}
    dtypes: dict[str, str] = {}
    for shard in shards:
        for name, dtype in pl.scan_parquet(shard).collect_schema().items():
            presence[name] = presence.get(name, 0) + 1
            observed.setdefault(name, set()).add(str(dtype))
            # `data_columns` reports the first shard (in sorted order) that carries the column, so
            # the value is deterministic; `column_dtype_conflicts` carries the full truth.
            dtypes.setdefault(name, str(dtype))
    conflicts = {name: sorted(seen) for name, seen in sorted(observed.items()) if len(seen) > 1}

    names = [_shard_name(meds_dir, s) for s in shards]
    code_metadata_fp = meds_dir / meds.code_metadata_filepath
    code_metadata_columns = None
    if code_metadata_fp.is_file():
        schema = pl.scan_parquet(code_metadata_fp).collect_schema()
        code_metadata_columns = {k: str(v) for k, v in sorted(schema.items())}

    return {
        "n_data_shards": len(shards),
        "shard_names": sorted(names) if len(names) <= 64 else None,
        "shard_names_digest": digest(sorted(names)),
        "data_columns": dict(sorted(dtypes.items())),
        "column_dtype_conflicts": conflicts,
        "column_presence": dict(sorted(presence.items())),
        "missing_core_columns": [c for c in _CORE_DATA_COLUMNS if c not in dtypes],
        "extra_data_columns": sorted(set(dtypes) - set(_CORE_DATA_COLUMNS)),
        "has_code_metadata": code_metadata_fp.is_file(),
        "has_subject_splits": (meds_dir / meds.subject_splits_filepath).is_file(),
        "has_dataset_metadata": (meds_dir / meds.dataset_metadata_filepath).is_file(),
        "code_metadata_columns": code_metadata_columns,
    }


#: Maximum number of files for which per-file digests are listed individually.
_MAX_FILES_LISTED = 64


def _schema_files(meds_dir: Path) -> dict[str, list[Path]]:
    """Group the files the MEDS schema defines, ignoring everything else in the directory.

    "Everything else" is deliberate. A MEDS root routinely carries task labels, cached
    intermediates, notes, and editor droppings, none of which are part of the dataset and none of
    which should make two datasets compare unequal.

    Args:
        meds_dir: The MEDS dataset root.

    Returns:
        Paths grouped by role: ``data``, ``code_metadata``, ``subject_splits``, ``dataset_metadata``.

    Examples:
        >>> spec = {
        ...     "data": {"train": {"0.parquet": {"subject_id": [1]}},
        ...              "notes.txt": "not part of the schema"},
        ...     "metadata": {"codes.parquet": {"code": ["A"]}, "scratch.json": {"ignored": True}},
        ...     "README.md": "also ignored",
        ... }
        >>> with yaml_disk(spec) as d:
        ...     {role: [p.relative_to(d).as_posix() for p in paths]
        ...      for role, paths in _schema_files(d).items()}
        {'data': ['data/train/0.parquet'], 'code_metadata': ['metadata/codes.parquet'],
         'subject_splits': [], 'dataset_metadata': []}
    """
    optional = {
        "code_metadata": meds_dir / meds.code_metadata_filepath,
        "subject_splits": meds_dir / meds.subject_splits_filepath,
        "dataset_metadata": meds_dir / meds.dataset_metadata_filepath,
    }
    grouped: dict[str, list[Path]] = {"data": sorted((meds_dir / meds.data_subdirectory).rglob("*.parquet"))}
    grouped.update({role: ([path] if path.is_file() else []) for role, path in optional.items()})
    return grouped


def _dataset_digest_block(meds_dir: Path, cfg: ExtractConfig) -> dict[str, Any]:
    """Checksum the dataset, so two extractions can assert byte-level identity.

    Summary statistics answer "did anything I measure change?". This answers the stronger and much
    simpler question "is this the same dataset at all?", which is what you want when pinning a
    dataset version: at ``dataset_version: 3.1`` you should get exactly this digest, and any
    deviation is worth a look before you even read the statistics.

    What it covers and what it does not:

    * Only files the MEDS schema defines. Task labels, caches, and stray files in the root are
      ignored, so they cannot make two copies of the same dataset compare unequal.
    * ``metadata/dataset.json`` is digested separately and left out of ``combined``, because it
      carries ``created_at`` and ``etl_version`` -- fields that change on every run by design.
      Folding them in would make the combined digest change every time and mean nothing.
    * ``combined`` hashes the *set* of file contents, not their paths, so renaming or re-nesting
      shards without changing their contents leaves it unchanged. Shard naming is tracked
      separately by ``layout.shard_names_digest``.

    This is a digest of *bytes*, which makes it the strictest available identity check and also
    means it is sensitive to re-encoding: the same rows rewritten by a different parquet writer, or
    with different compression or row-group sizes, produce a different digest. When that happens
    the digest moves while every statistic stays put, and that combination is itself the diagnosis.

    Args:
        meds_dir: The MEDS dataset root.
        cfg: Extraction configuration.

    Returns:
        The ``dataset_digest`` block, or ``{"method": "none"}`` when disabled.

    Examples:
        >>> from pprint import pprint
        >>> spec = {
        ...     "data": {"0.parquet": {"subject_id": [1], "code": ["A"]}},
        ...     "metadata": {"codes.parquet": {"code": ["A"]},
        ...                  "dataset.json": {"dataset_version": "1.0"}},
        ... }
        >>> with yaml_disk(spec) as d:
        ...     block = _dataset_digest_block(d, ExtractConfig())
        >>> pprint(block)
        {'by_file': {'data/0.parquet': 'sha256:...',
                     'metadata/codes.parquet': 'sha256:...'},
         'code_metadata': 'sha256:...',
         'combined': 'sha256:...',
         'data': {'digest': 'sha256:...', 'n_files': 1},
         'dataset_metadata': 'sha256:...',
         'method': 'files',
         'n_files': 3,
         'subject_splits': None}

        ``yaml_disk`` is a singleton whose context manager tears down the previous root, so each
        variant below is written and digested one at a time rather than in a shared ``with``:

        >>> def digest_of(contents):
        ...     with yaml_disk(contents) as root:
        ...         return _dataset_digest_block(root, ExtractConfig())

        Two byte-identical datasets agree, and a one-row change disagrees:

        >>> changed = {**spec, "data": {"0.parquet": {"subject_id": [2], "code": ["A"]}}}
        >>> digest_of(spec)["combined"] == digest_of(spec)["combined"]
        True
        >>> digest_of(spec)["combined"] == digest_of(changed)["combined"]
        False

        A file outside the MEDS schema cannot affect it:

        >>> noisy = {**spec, "scratch.txt": "irrelevant", "labels": {"task.parquet": {"x": [1]}}}
        >>> digest_of(spec)["combined"] == digest_of(noisy)["combined"]
        True

        Neither can ``dataset.json``, which is why it is excluded -- an ETL rerun bumps its
        ``created_at`` and would otherwise invalidate the digest every time:

        >>> rerun = {**spec, "metadata": {"codes.parquet": {"code": ["A"]},
        ...                               "dataset.json": {"dataset_version": "1.0",
        ...                                                "created_at": "2026-08-04T00:00:00"}}}
        >>> digest_of(spec)["combined"] == digest_of(rerun)["combined"]
        True
        >>> digest_of(spec)["dataset_metadata"] == digest_of(rerun)["dataset_metadata"]
        False

        >>> with yaml_disk(spec) as d:
        ...     _dataset_digest_block(d, ExtractConfig(dataset_digest="none"))
        {'method': 'none'}
    """
    if cfg.dataset_digest == "none":
        return {"method": "none"}

    grouped = _schema_files(meds_dir)
    per_file = {
        path.relative_to(meds_dir).as_posix(): file_digest(path)
        for paths in grouped.values()
        for path in paths
    }

    def role_digest(role: str) -> str | None:
        paths = grouped[role]
        if not paths:
            return None
        return combine_digests(per_file[p.relative_to(meds_dir).as_posix()] for p in paths)

    # `combined` covers the data and the metadata that describes it, but not dataset.json.
    combined_roles = ("data", "code_metadata", "subject_splits")
    combined_source = [
        per_file[p.relative_to(meds_dir).as_posix()] for role in combined_roles for p in grouped[role]
    ]

    return {
        "method": "files",
        "n_files": len(per_file),
        "combined": combine_digests(combined_source),
        "data": {"n_files": len(grouped["data"]), "digest": role_digest("data")},
        "code_metadata": role_digest("code_metadata"),
        "subject_splits": role_digest("subject_splits"),
        "dataset_metadata": role_digest("dataset_metadata"),
        "by_file": dict(sorted(per_file.items())) if len(per_file) <= _MAX_FILES_LISTED else None,
    }


def _base_frame(shards: list[Path], available: set[str]) -> pl.LazyFrame:
    """Build the normalized lazy frame every aggregate reads from.

    Each shard is scanned and normalized independently before being concatenated, rather than opened
    as one multi-file scan. That costs nothing at plan time and buys the ability to read a dataset
    whose shards disagree on schema -- a column present in some and not others, or typed ``Int32``
    here and ``Int64`` there. Such a dataset is broken, and a tool for finding broken ETL output has
    to be able to open it and say so (see ``layout.column_dtype_conflicts``) rather than crash.

    Missing optional columns are materialized as typed nulls so downstream expressions need not
    branch on schema, and ``numeric_value`` is widened to ``Float64`` so moments do not accumulate
    ``Float32`` error over billions of rows.

    Args:
        shards: The data shard paths.
        available: Column names present in at least one shard.

    Returns:
        A lazy frame with exactly the four core MEDS data columns.

    Raises:
        NotAMEDSDatasetError: If a column MEDS requires is absent from every shard.

    Examples:
        >>> spec = {"data": {"0.parquet": {"subject_id": [1, 2], "code": ["A", "B"]}}}
        >>> with yaml_disk(spec) as d:
        ...     shards = _shard_paths(d)
        ...     _base_frame(shards, {"subject_id", "code"}).collect()
        shape: (2, 4)
        ┌────────────┬──────────────┬──────┬───────────────┐
        │ subject_id ┆ time         ┆ code ┆ numeric_value │
        │ ---        ┆ ---          ┆ ---  ┆ ---           │
        │ i64        ┆ datetime[μs] ┆ str  ┆ f64           │
        ╞════════════╪══════════════╪══════╪═══════════════╡
        │ 1          ┆ null         ┆ A    ┆ null          │
        │ 2          ┆ null         ┆ B    ┆ null          │
        └────────────┴──────────────┴──────┴───────────────┘

        >>> with yaml_disk({"data": {"0.parquet": {"code": ["A"]}}}) as d:
        ...     _base_frame(_shard_paths(d), {"code"})
        Traceback (most recent call last):
            ...
        meds_summary_stats.extract.NotAMEDSDatasetError: Data shards are missing required
        column(s): subject_id

        Shards that disagree on dtype are still readable, normalized to the wider type:

        >>> spec = {"data": {
        ...     "0.parquet": {"subject_id": [1], "code": ["A"]},
        ...     "1.parquet": pa.table({"subject_id": pa.array([2], pa.int32()),
        ...                            "code": pa.array(["B"])}),
        ... }}
        >>> with yaml_disk(spec) as d:
        ...     _base_frame(_shard_paths(d), {"subject_id", "code"}).collect()["subject_id"].to_list()
        [1, 2]
    """
    if missing := [c for c in _REQUIRED_DATA_COLUMNS if c not in available]:
        raise NotAMEDSDatasetError(f"Data shards are missing required column(s): {', '.join(missing)}")

    def normalize(shard: Path) -> pl.LazyFrame:
        lf = pl.scan_parquet(shard)
        present = set(lf.collect_schema())

        def column(name: str, dtype: pl.DataType) -> pl.Expr:
            if name in present:
                return pl.col(name).cast(dtype, strict=False)
            return pl.lit(None, dtype).alias(name)

        return lf.select(
            column("subject_id", pl.Int64),
            column("time", pl.Datetime("us")),
            column("code", pl.String),
            column("numeric_value", pl.Float64),
        )

    frames = [normalize(shard) for shard in shards]
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")


def _code_vocabulary_map(meds_dir: Path) -> pl.LazyFrame | None:
    """Build the distinct ``(code, vocabulary)`` mapping implied by ``parent_codes``.

    Args:
        meds_dir: The MEDS dataset root.

    Returns:
        A lazy frame with ``code`` and ``vocabulary`` columns, or ``None`` if there is no code
        metadata file to read.

    Examples:
        >>> spec = {"metadata": {"codes.parquet": {
        ...     "code": ["HR", "GLU", "X"],
        ...     "parent_codes": [["LOINC/8867-4", "SNOMED/364075005"], ["LOINC/2345-7"], None],
        ... }}}
        >>> with yaml_disk(spec) as d:
        ...     _code_vocabulary_map(d).collect().sort("code", "vocabulary")
        shape: (3, 2)
        ┌──────┬────────────┐
        │ code ┆ vocabulary │
        │ ---  ┆ ---        │
        │ str  ┆ str        │
        ╞══════╪════════════╡
        │ GLU  ┆ LOINC      │
        │ HR   ┆ LOINC      │
        │ HR   ┆ SNOMED     │
        └──────┴────────────┘

        >>> with yaml_disk({"data": {}}) as d:
        ...     _code_vocabulary_map(d) is None
        True

        Code metadata without a ``parent_codes`` column carries no ontology linkage, so there is
        nothing to map:

        >>> with yaml_disk({"metadata": {"codes.parquet": {"code": ["HR"]}}}) as d:
        ...     _code_vocabulary_map(d) is None
        True
    """
    fp = meds_dir / meds.code_metadata_filepath
    if not fp.is_file():
        return None
    schema = pl.scan_parquet(fp).collect_schema()
    if "parent_codes" not in schema:
        return None
    return (
        pl.scan_parquet(fp)
        .select("code", "parent_codes")
        .explode("parent_codes", empty_as_null=True)
        .drop_nulls("parent_codes")
        .rename({"parent_codes": "parent_code"})
        .select("code", vocabulary_expr())
        .unique()
    )


def _subject_filters(counts: pl.DataFrame, max_rows: int) -> list[pl.Expr]:
    """Cut the subject-id axis into contiguous ranges of roughly ``max_rows`` rows each.

    Every subject lands in exactly one range, which is what lets distinct-subject counts add across
    them. The ranges are half-open and the outer two unbounded, so every possible id is covered
    whether or not it appears in ``counts``. Null subject ids fall in no range and ride with the
    first.

    A new range starts at the first subject whose starting row offset crosses the next multiple of
    ``max_rows``, so a range overshoots the budget by less than one subject -- the one unit that
    cannot be split.

    Args:
        counts: Rows per subject, as columns ``subject_id`` and ``n``.
        max_rows: The row budget per range.

    Returns:
        One filter expression per range.

    Examples:
        >>> frame = pl.DataFrame({"subject_id": [1, 2, 3, 4, 5, None, 9]})
        >>> counts = pl.DataFrame({"subject_id": [1, 2, 3, 4, 5], "n": [4, 4, 4, 4, 4]})
        >>> [frame.filter(f)["subject_id"].to_list() for f in _subject_filters(counts, 10)]
        [[1, 2, 3, None], [4, 5, 9]]

        A subject larger than the whole budget still stays whole:

        >>> counts = pl.DataFrame({"subject_id": [1, 2, 3], "n": [1, 50, 1]})
        >>> [frame.filter(f)["subject_id"].to_list() for f in _subject_filters(counts, 10)]
        [[1, 2, None], [3, 4, 5, 9]]

        When everything fits there is one range, and it filters nothing:

        >>> [frame.filter(f).height for f in _subject_filters(counts, 100)]
        [7]
    """
    starts = (
        counts.drop_nulls("subject_id")
        .sort("subject_id")
        .select("subject_id", n=pl.col("n").cast(pl.Int64))
        .select("subject_id", bucket=(pl.col("n").cum_sum() - pl.col("n")) // max_rows)
    )
    cuts = starts.group_by("bucket").agg(pl.col("subject_id").min()).sort("bucket")["subject_id"][1:]

    sid = pl.col("subject_id")
    if cuts.is_empty():
        return [pl.lit(True)]
    return [
        (sid < cuts[0]) | sid.is_null(),
        *((sid >= lo) & (sid < hi) for lo, hi in itertools.pairwise(cuts)),
        sid >= cuts[-1],
    ]


def _plan_chunks(shards: list[Path], available: set[str], max_rows: int | None) -> list[pl.LazyFrame]:
    """Partition the dataset into chunks of about ``max_rows`` rows that never split a subject.

    **Why chunk.** The per-subject and per-code aggregates count distinct ``(subject, time)`` and
    ``(subject, code)`` pairs, so their state grows with the rows read, and the streaming engine's
    buffering grows with it. Over a whole dataset peak memory is linear in the row count -- about
    70 bytes a row in practice, which exhausts a 121 GB machine on a 2.7-billion-row cohort.
    Collected one chunk at a time, it is bounded by the chunk instead.

    **Why never split a subject.** Per-subject counts and every distinct-subject count are only
    additive across chunks that share no subject.

    The plan, cheapest first:

    1. ``max_rows`` is ``None``, or the dataset fits in one chunk: one chunk, read in one pass.
    2. Every subject lives in a single shard, as MEDS ETLs produce: consecutive shards are grouped up
       to the budget, and any shard larger than the budget is split by subject-id range.
    3. Some subject spans shards. That is non-conformant, but this tool exists to read broken ETL
       output, so rather than give up the whole dataset is split by subject-id range instead.

    Telling 2 from 3 costs one pass over ``subject_id`` alone; the row counts are parquet metadata.

    Args:
        shards: The data shard paths.
        available: Column names present in at least one shard.
        max_rows: The row budget per chunk, or ``None`` for a single pass.

    Returns:
        Normalized lazy frames (see :func:`_base_frame`) that together hold every row exactly once.

    Examples:
        >>> spec = {"data": {
        ...     "0.parquet": {"subject_id": [1, 1, 2], "code": ["A", "B", "A"]},
        ...     "1.parquet": {"subject_id": [3, 3, 3, 4], "code": ["A", "A", "B", "B"]},
        ...     "2.parquet": {"subject_id": [5], "code": ["C"]},
        ... }}
        >>> def plan(max_rows):
        ...     with yaml_disk(spec) as d:
        ...         chunks = _plan_chunks(_shard_paths(d), {"subject_id", "code"}, max_rows)
        ...         return [sorted(c.collect()["subject_id"].to_list()) for c in chunks]

        With room to spare, the dataset is read in one pass:

        >>> plan(None)
        [[1, 1, 2, 3, 3, 3, 4, 5]]
        >>> plan(100)
        [[1, 1, 2, 3, 3, 3, 4, 5]]

        Otherwise consecutive shards are grouped up to the budget:

        >>> plan(5)
        [[1, 1, 2], [3, 3, 3, 4, 5]]

        And a shard too large for the budget on its own is split between its subjects:

        >>> plan(3)
        [[1, 1, 2], [3, 3, 3], [4], [5]]

        If a subject spans shards, no shard grouping is safe, so the cut is made by subject across
        the whole dataset. Subject 1 below is in both shards, and each chunk still holds all of it:

        >>> spec = {"data": {
        ...     "0.parquet": {"subject_id": [1, 2, 3], "code": ["A", "A", "A"]},
        ...     "1.parquet": {"subject_id": [1, 4], "code": ["B", "B"]},
        ... }}
        >>> plan(3)
        [[1, 1, 2], [3, 4]]
    """
    if max_rows is None:
        return [_base_frame(shards, available)]
    rows = [pl.scan_parquet(shard).select(pl.len()).collect().item() for shard in shards]
    if sum(rows) <= max_rows:
        return [_base_frame(shards, available)]

    counts = (
        pl.concat(
            [_base_frame([s], available).select("subject_id", shard=pl.lit(i)) for i, s in enumerate(shards)]
        )
        .group_by("shard", "subject_id")
        .agg(n=pl.len())
        .collect(engine="streaming")
    )
    if counts["subject_id"].is_duplicated().any():
        everything = _base_frame(shards, available)
        totals = counts.group_by("subject_id").agg(pl.col("n").sum())
        return [everything.filter(f) for f in _subject_filters(totals, max_rows)]

    chunks: list[pl.LazyFrame] = []
    group: list[Path] = []
    group_rows = 0
    for i, (shard, n) in enumerate(zip(shards, rows, strict=True)):
        if group and group_rows + n > max_rows:
            chunks.append(_base_frame(group, available))
            group, group_rows = [], 0
        if n > max_rows:
            whole_shard = _base_frame([shard], available)
            filters = _subject_filters(counts.filter(pl.col("shard") == i), max_rows)
            chunks.extend(whole_shard.filter(f) for f in filters)
            continue
        group.append(shard)
        group_rows += n
    if group:
        chunks.append(_base_frame(group, available))
    return chunks


def _chunk_aggregates(base: pl.LazyFrame, vocab: pl.DataFrame | None) -> dict[str, pl.DataFrame]:
    """Collect every combinable aggregate over one chunk, in a single batch.

    The queries are collected together so that polars can share the parquet scan and common subplans
    across them: the chunk is read once rather than once per statistic.

    Args:
        base: One chunk of the normalized four-column frame.
        vocab: Optional ``(code, vocabulary)`` mapping to attribute measurements to ontologies.

    Returns:
        A dict of collected frames keyed by aggregate name.
    """
    nv = pl.col("numeric_value")
    finite = _finite_numeric()

    queries: dict[str, pl.LazyFrame] = {
        "overall": base.select(
            n_measurements=pl.len(),
            n_subjects=pl.col("subject_id").n_unique(),
            n_unique_codes=pl.col("code").n_unique(),
            n_static_measurements=pl.col("time").is_null().sum(),
            n_null_subject_id=pl.col("subject_id").null_count(),
            n_null_code=pl.col("code").null_count(),
            n_empty_code=(pl.col("code").str.len_bytes() == 0).sum(),
            numeric_n_present=nv.is_not_null().sum(),
            numeric_n_nan=nv.is_nan().sum(),
            numeric_n_infinite=nv.is_infinite().sum(),
            numeric_n_finite=finite.count(),
            numeric_n_negative=(finite < 0).sum(),
            numeric_n_zero=(finite == 0).sum(),
            numeric_mean=finite.mean(),
            numeric_std=finite.std(),
        ),
        "per_subject": base.group_by("subject_id").agg(
            n_measurements=pl.len(),
            n_events=pl.col("time").n_unique(),
            n_unique_codes=pl.col("code").n_unique(),
            n_static=pl.col("time").is_null().sum(),
            t_min=pl.col("time").min(),
            t_max=pl.col("time").max(),
            has_birth=(pl.col("code") == meds.birth_code).any(),
            has_death=(pl.col("code") == meds.death_code).any(),
        ),
        "per_code": base.group_by("code").agg(
            n_measurements=pl.len(),
            n_subjects=pl.col("subject_id").n_unique(),
            numeric_n_finite=finite.count(),
            numeric_mean=finite.mean(),
            numeric_std=finite.std(),
        ),
        "by_year": base.drop_nulls("time")
        .group_by(pl.col("time").dt.year().alias("year"))
        .agg(n_measurements=pl.len(), n_subjects=pl.col("subject_id").n_unique()),
    }

    if vocab is not None:
        # A code with parents in two vocabularies is counted under both, so these columns
        # deliberately over-sum relative to the dataset total.
        queries["by_vocabulary"] = (
            base.join(vocab.lazy(), on="code", how="inner")
            .group_by("vocabulary")
            .agg(
                n_measurements=pl.len(),
                n_subjects=pl.col("subject_id").n_unique(),
                n_child_codes=pl.col("code").n_unique(),
                numeric_n_finite=finite.count(),
                numeric_mean=finite.mean(),
                numeric_std=finite.std(),
            )
        )
        queries["unmapped"] = base.join(vocab.lazy().select("code").unique(), on="code", how="anti").select(
            n_measurements=pl.len()
        )

    names = list(queries)
    collected = pl.collect_all([queries[n] for n in names], engine="streaming")
    return dict(zip(names, collected, strict=True))


def _pooled_moments() -> list[pl.Expr]:
    """Pool per-chunk ``numeric_mean`` and ``numeric_std`` into whole-dataset ones.

    The parallel form of Welford's update (Chan, Golub & LeVeque): each chunk contributes its sum of
    squared deviations, ``std**2 * (n - 1)``, plus a between-chunk term for how far its mean sits from
    the pooled mean. That is exact in real arithmetic, and in floating point agrees with a single pass
    far inside the six significant digits the document carries. When only one chunk holds any finite
    value -- every key in a single-chunk run, and most rare codes in a chunked one -- its moments pass
    through untouched.

    Works as a ``select`` over stacked partials, or inside ``group_by(...).agg``.

    Examples:
        >>> def moments(frame):
        ...     return frame.select(numeric_n_finite=pl.len(), numeric_mean=pl.col("x").mean(),
        ...                         numeric_std=pl.col("x").std())
        >>> parts = [pl.DataFrame({"x": xs}) for xs in ([1.0, 2.0], [4.0], [8.0, 16.0, 32.0])]
        >>> pooled = pl.concat([moments(p) for p in parts]).select(_pooled_moments())
        >>> single_pass = moments(pl.concat(parts)).drop("numeric_n_finite")
        >>> [abs(a - b) < 1e-12 for a, b in zip(pooled.row(0), single_pass.row(0))]
        [True, True]

        Chunks without a finite value contribute nothing, and none at all yields nulls, as a
        single pass over no values would:

        >>> parts = pl.DataFrame({"numeric_n_finite": [0, 1, 0], "numeric_mean": [None, 5.0, None],
        ...                       "numeric_std": [None, None, None]})
        >>> parts.select(_pooled_moments()).row(0)
        (5.0, None)
        >>> parts.head(1).select(_pooled_moments()).row(0)
        (None, None)
    """
    n = pl.col("numeric_n_finite").cast(pl.Float64)
    mean = pl.col("numeric_mean")
    std = pl.col("numeric_std")

    holds = n > 0
    n_holding = holds.sum()
    n_total = n.sum()
    pooled_mean = (n * mean).filter(holds).sum() / n_total
    m2 = (std.fill_null(0.0).pow(2) * (n - 1) + n * (mean - pooled_mean).pow(2)).filter(holds).sum()
    return [
        pl.when(n_holding == 1)
        .then(mean.filter(holds).first())
        .when(n_holding > 1)
        .then(pooled_mean)
        .alias("numeric_mean"),
        pl.when(n_holding == 1)
        .then(std.filter(holds).first())
        .when(n_holding > 1)
        .then((m2 / (n_total - 1)).sqrt())
        .alias("numeric_std"),
    ]


def _summed(*names: str) -> list[pl.Expr]:
    """Sum count columns across chunks, widened first so totals past 2**32 cannot wrap."""
    return [pl.col(name).cast(pl.UInt64).sum() for name in names]


def _combine_partials(
    partials: list[dict[str, pl.DataFrame]], vocab: pl.DataFrame | None
) -> dict[str, pl.DataFrame]:
    """Combine per-chunk aggregates into whole-dataset ones.

    Exact because no subject spans two chunks (see :func:`_plan_chunks`): per-subject rows
    concatenate, and every distinct-subject count sums. Two counts are distinct over *codes*, which
    do recur across chunks, so they are recomputed from the combined per-code frame instead -- the
    dataset's ``n_unique_codes`` and each vocabulary's ``n_child_codes``. Moments are pooled by
    :func:`_pooled_moments`.

    Args:
        partials: One dict of frames per chunk, as returned by :func:`_chunk_aggregates`.
        vocab: The ``(code, vocabulary)`` mapping, if any.

    Returns:
        A dict of frames with the same names and columns as each partial.

    Examples:
        Two chunks with one subject each, both measuring code ``A``:

        >>> frame = pl.LazyFrame(
        ...     {"subject_id": [1, 1, 2], "time": [None, None, None], "code": ["A", "B", "A"],
        ...      "numeric_value": [1.0, None, 3.0]},
        ...     schema_overrides={"time": pl.Datetime("us")})
        >>> parts = [_chunk_aggregates(frame.filter(pl.col("subject_id") == s), None) for s in (1, 2)]
        >>> combined = _combine_partials(parts, None)

        Subjects are disjoint, so subject counts add. Codes are not, so ``n_unique_codes`` is 2, not
        the 2 + 1 a sum would give:

        >>> combined["overall"].select("n_measurements", "n_subjects", "n_unique_codes").row(0)
        (3, 2, 2)
        >>> combined["per_code"].sort("code").select("code", "n_subjects", "numeric_mean").rows()
        [('A', 2, 2.0), ('B', 1, None)]
    """
    stacked = {name: pl.concat([part[name] for part in partials]) for name in partials[0]}

    per_code = (
        stacked["per_code"]
        .group_by("code")
        .agg(*_summed("n_measurements", "n_subjects", "numeric_n_finite"), *_pooled_moments())
    )
    not_summed = ("n_unique_codes", "numeric_mean", "numeric_std")
    overall_counts = [c for c in stacked["overall"].columns if c not in not_summed]
    combined = {
        "overall": stacked["overall"].select(
            *_summed(*overall_counts),
            *_pooled_moments(),
            n_unique_codes=pl.lit(per_code.height, pl.UInt64),
        ),
        "per_subject": stacked["per_subject"],
        "per_code": per_code,
        "by_year": stacked["by_year"].group_by("year").agg(*_summed("n_measurements", "n_subjects")),
    }
    if vocab is not None:
        n_child_codes = per_code.join(vocab, on="code").group_by("vocabulary").agg(n_child_codes=pl.len())
        combined["by_vocabulary"] = (
            stacked["by_vocabulary"]
            .group_by("vocabulary")
            .agg(*_summed("n_measurements", "n_subjects", "numeric_n_finite"), *_pooled_moments())
            .join(n_child_codes, on="vocabulary", how="left")
        )
        combined["unmapped"] = stacked["unmapped"].select(*_summed("n_measurements"))

    # Keep each frame's column order, so a combined frame reads exactly like a single chunk's.
    return {name: frame.select(partials[0][name].columns) for name, frame in combined.items()}


def _aggregate(
    chunks: list[pl.LazyFrame], cfg: ExtractConfig, vocab_map: pl.LazyFrame | None
) -> dict[str, pl.DataFrame]:
    """Run every data-scanning aggregate, chunk by chunk, and combine the results.

    Chunks are collected one after another, so only one chunk's aggregation state is ever resident;
    see :func:`_plan_chunks` for why that matters and :func:`_combine_partials` for why the result is
    the same as a single pass.

    Full-column quantiles (``numeric_detail="quantiles"``) cannot be assembled from parts, so they are
    collected afterwards over the whole dataset, one column at a time. Their memory does grow with the
    data, which is why they are opt-in.

    Args:
        chunks: The normalized four-column frame, partitioned as by :func:`_plan_chunks`.
        cfg: Extraction configuration.
        vocab_map: Optional ``(code, vocabulary)`` mapping to attribute measurements to ontologies.

    Returns:
        A dict of collected frames keyed by aggregate name.

    Examples:
        >>> spec = {"data": {"0.parquet": {
        ...     "subject_id": [1, 1, 2], "code": ["GLU", "GLU", "HR"],
        ...     "numeric_value": [1.0, 3.0, 60.0],
        ... }}}
        >>> with yaml_disk(spec) as d:
        ...     frames = _aggregate([_base_frame(_shard_paths(d), {"subject_id", "code", "numeric_value"})],
        ...                         ExtractConfig(), None)
        >>> sorted(frames)
        ['by_year', 'overall', 'per_code', 'per_subject']
        >>> frames["per_code"].sort("code").select("code", "n_measurements", "n_subjects")
        shape: (2, 3)
        ┌──────┬────────────────┬────────────┐
        │ code ┆ n_measurements ┆ n_subjects │
        │ ---  ┆ ---            ┆ ---        │
        │ str  ┆ u64            ┆ u64        │
        ╞══════╪════════════════╪════════════╡
        │ GLU  ┆ 2              ┆ 1          │
        │ HR   ┆ 1              ┆ 1          │
        └──────┴────────────────┴────────────┘
        >>> frames["overall"].select("n_measurements", "n_subjects", "numeric_mean").to_dicts()
        [{'n_measurements': 3, 'n_subjects': 2, 'numeric_mean': 21.333333333333332}]

        Split by subject into two chunks, the result is the same:

        >>> with yaml_disk(spec) as d:
        ...     base = _base_frame(_shard_paths(d), {"subject_id", "code", "numeric_value"})
        ...     chunked = _aggregate([base.filter(pl.col("subject_id") == s) for s in (1, 2)],
        ...                          ExtractConfig(), None)
        >>> chunked["overall"].select("n_measurements", "n_subjects", "numeric_mean").to_dicts()
        [{'n_measurements': 3, 'n_subjects': 2, 'numeric_mean': 21.333333333333332}]

        When code metadata supplies a ``(code, vocabulary)`` mapping, an extra aggregate attributes
        measurement volume and numeric moments to each external vocabulary. A code mapped into two
        vocabularies is counted under both, so these columns do not sum to the dataset total:

        >>> vocab_map = pl.LazyFrame({"code": ["GLU", "GLU", "HR"],
        ...                           "vocabulary": ["LOINC", "SNOMED", "LOINC"]})
        >>> with yaml_disk(spec) as d:
        ...     frames = _aggregate([_base_frame(_shard_paths(d), {"subject_id", "code", "numeric_value"})],
        ...                         ExtractConfig(), vocab_map)
        >>> frames["by_vocabulary"].sort("vocabulary").select(
        ...     "vocabulary", "n_measurements", "n_subjects", "n_child_codes")
        shape: (2, 4)
        ┌────────────┬────────────────┬────────────┬───────────────┐
        │ vocabulary ┆ n_measurements ┆ n_subjects ┆ n_child_codes │
        │ ---        ┆ ---            ┆ ---        ┆ ---           │
        │ str        ┆ u64            ┆ u64        ┆ u32           │
        ╞════════════╪════════════════╪════════════╪═══════════════╡
        │ LOINC      ┆ 3              ┆ 2          ┆ 2             │
        │ SNOMED     ┆ 2              ┆ 1          ┆ 1             │
        └────────────┴────────────────┴────────────┴───────────────┘

        A parallel aggregate covers the measurements whose code maps into no vocabulary at all,
        which is the signal that a concept-mapping table has stopped resolving:

        >>> frames["unmapped"].to_dicts()
        [{'n_measurements': 0}]
    """
    vocab = vocab_map.collect() if vocab_map is not None else None
    frames = _combine_partials([_chunk_aggregates(chunk, vocab) for chunk in chunks], vocab)

    if cfg.numeric_detail == "quantiles":
        whole = chunks[0] if len(chunks) == 1 else pl.concat(chunks)
        finite = _finite_numeric()
        frames["numeric_quantiles"] = whole.select(
            **{f"q{_qkey(q)}": finite.quantile(q, "linear") for q in cfg.quantiles}
        ).collect(engine="streaming")
        frames["time_quantiles"] = (
            whole.drop_nulls("time")
            .select(
                **{f"q{_qkey(q)}": pl.col("time").dt.epoch("us").quantile(q, "linear") for q in cfg.quantiles}
            )
            .collect(engine="streaming")
        )
    return frames


def _codes_block(per_code: pl.DataFrame, cfg: ExtractConfig) -> dict[str, Any]:
    """Summarize the code vocabulary without emitting any code.

    Three complementary views are reported, because they fail differently:

    * **Digests** answer "is this the same vocabulary?" exactly. The alphabetical digest changes if and
      only if the *set* of codes changed. The frequency-ordered digests additionally change if the
      relative ranking shifted, which makes them sensitive tripwires but noisy ones -- hence the
      thresholded variants, restricted to codes common enough that their ordering is stable.
    * **Occurrence distributions** answer "did the shape of the vocabulary change?" and survive
      benign additions to the tail.
    * **Concentration** answers "did the head-versus-tail profile change?", which moves sharply when
      a mapping bug collapses many codes into one or explodes one into many.

    What is deliberately *not* here is any partition of the vocabulary by the structure of the code
    strings themselves. Splitting on ``//`` would give a convenient "which part of the vocabulary
    moved" axis, but MEDS does not require that convention, so the axis would be meaningful on some
    datasets, absent on others, and -- for a flat vocabulary, where the prefix is the whole code --
    would republish the vocabulary outright. The equivalent axis is taken from ``parent_codes``
    instead, in :func:`_ontology_block`, where the groupings are declared rather than guessed.

    Args:
        per_code: Collected per-code aggregate.
        cfg: Extraction configuration.

    Returns:
        The ``codes`` block of the stats document.

    Examples:
        >>> per_code = pl.DataFrame({
        ...     "code": ["GLUCOSE", "SODIUM", "HR"],
        ...     "n_measurements": [100, 50, 10],
        ...     "n_subjects": [40, 30, 5],
        ...     "numeric_n_finite": [100, 0, 10],
        ...     "numeric_mean": [1.5, None, 60.0],
        ...     "numeric_std": [0.5, None, 5.0],
        ... })
        >>> block = _codes_block(per_code, ExtractConfig(min_cell_size=20))

        No code string survives anywhere in the output:

        >>> any(c in json.dumps(block) for c in ("GLUCOSE", "SODIUM", "HR"))
        False

        The rare code is excluded from the reportable count but still counted:

        >>> block["n_unique_codes"], block["n_codes_below_min_cell_size"]
        (3, 1)

        Digests are taken at every configured threshold, and each reports how many codes it covered so
        that a digest change can be read against a changing denominator:

        >>> sorted(block["digests"])
        ['alphabetical', 'by_measurement_frequency', 'by_subject_frequency',
         'numeric_moments', 'thresholded']
        >>> block["digests"]["thresholded"]["20"]["n_codes"]
        2
        >>> block["digests"]["thresholded"]["1000"]["n_codes"]
        0
    """
    codes = per_code["code"].to_list()
    n_measurements = per_code["n_measurements"].to_list()
    n_subjects = per_code["n_subjects"].to_list()

    thresholded: dict[str, Any] = {}
    for threshold in cfg.digest_subject_thresholds:
        subset = per_code.filter(pl.col("n_subjects") >= threshold)
        subset_codes = subset["code"].to_list()
        thresholded[str(threshold)] = {
            "n_codes": subset.height,
            "alphabetical": digest(sorted(subset_codes)),
            "by_measurement_frequency": digest(ordered_by(subset_codes, subset["n_measurements"].to_list())),
            "by_subject_frequency": digest(ordered_by(subset_codes, subset["n_subjects"].to_list())),
        }

    moment_rows = per_code.filter(
        (pl.col("n_subjects") >= cfg.min_cell_size)
        & (pl.col("numeric_n_finite") >= cfg.min_numeric_cell_size)
    ).sort("code")
    sig = cfg.moment_digest_sig_digits
    moment_lines = [
        f"{c}\t{format_moment(m, sig)}\t{format_moment(s, sig)}"
        for c, m, s in zip(
            moment_rows["code"].to_list(),
            moment_rows["numeric_mean"].to_list(),
            moment_rows["numeric_std"].to_list(),
            strict=True,
        )
    ]

    return {
        "n_unique_codes": per_code.height,
        "n_codes_below_min_cell_size": int((per_code["n_subjects"] < cfg.min_cell_size).sum()),
        "digests": {
            "alphabetical": digest(sorted(codes)),
            "by_measurement_frequency": digest(ordered_by(codes, n_measurements)),
            "by_subject_frequency": digest(ordered_by(codes, n_subjects)),
            "numeric_moments": {"n_codes": len(moment_lines), "digest": digest(moment_lines)},
            "thresholded": thresholded,
        },
        "occurrence": {
            "n_measurements": _distribution(per_code["n_measurements"].cast(pl.Float64), cfg),
            "n_subjects": _distribution(per_code["n_subjects"].cast(pl.Float64), cfg),
        },
        "concentration": _concentration(n_measurements),
    }


def _ontology_block(
    meds_dir: Path,
    by_vocabulary: pl.DataFrame | None,
    unmapped: pl.DataFrame | None,
    observed_codes: set[str],
    n_total_measurements: int | None,
    cfg: ExtractConfig,
) -> dict[str, Any]:
    """Summarize the external ontologies that ``parent_codes`` links into.

    This is the only axis along which the vocabulary is partitioned, and the reason it is the only
    one: ``parent_codes`` entries name concepts in *published* terminologies -- LOINC, SNOMED, ICD --
    so the grouping is declared by the dataset rather than guessed from the shape of its code
    strings, and the group names are properties of the terminology rather than of any patient. It
    therefore does double duty. It answers "*where* did the vocabulary change?", which is what
    someone debugging a failed comparison needs, and it is a strong ETL signal in its own right: a
    concept-mapping table that silently stops resolving shows up here as a vocabulary whose link
    count collapsed, or as a jump in unmapped measurement volume, while every count over the data
    itself stays exactly the same.

    Suppression still applies to the *statistics*: a vocabulary reached by too few subjects is rolled
    up rather than reported, because its measurement count would otherwise describe a tiny cohort.
    The ``coverage`` figures are global ratios and counts, so they are always reported.

    Args:
        meds_dir: The MEDS dataset root.
        by_vocabulary: Collected per-vocabulary data aggregate, or ``None`` if unavailable.
        unmapped: Collected aggregate over measurements whose code has no parents, or ``None``.
        observed_codes: Codes actually present in the data, for the coverage figures.
        n_total_measurements: Total measurements in the dataset, or ``None`` if unavailable. Mapped
            volume is derived from this rather than summed across vocabularies, because a code with
            parents in two vocabularies contributes to both and the sum would exceed the total.
        cfg: Extraction configuration.

    Returns:
        The ``ontology`` sub-block, or ``{"present": False}`` when there is nothing to read.

    Examples:
        >>> from pprint import pprint
        >>> spec = {"metadata": {"codes.parquet": {
        ...     "code": ["HR", "GLU", "X"],
        ...     "parent_codes": [["LOINC/8867-4"], ["LOINC/2345-7", "SNOMED/33747003"], None],
        ... }}}
        >>> by_vocab = pl.DataFrame({
        ...     "vocabulary": ["LOINC", "SNOMED"],
        ...     "n_measurements": [500, 20], "n_subjects": [50, 2], "n_child_codes": [2, 1],
        ...     "numeric_n_finite": [480, 0], "numeric_mean": [88.5, None], "numeric_std": [12.25, None],
        ... })
        >>> unmapped = pl.DataFrame([{"n_measurements": 40}])
        >>> with yaml_disk(spec) as d:
        ...     block = _ontology_block(d, by_vocab, unmapped, {"HR", "GLU", "X"}, 540,
        ...                             ExtractConfig(min_cell_size=20))
        >>> pprint({k: v for k, v in block.items() if k not in ("by_vocabulary", "coverage")})
        {'n_codes_in_multiple_vocabularies': 1,
         'n_codes_with_parents': 2,
         'n_distinct_parents': 3,
         'n_parent_links': 3,
         'n_parents_without_vocabulary': 0,
         'n_vocabularies': 2,
         'n_vocabularies_recognized': 2,
         'parents_per_code': {'n': 3, 'suppressed': True},
         'present': True,
         'vocabularies_per_code': {'n': 3, 'suppressed': True}}

        Coverage is where an unresolved mapping table shows up. These are dataset-level ratios, so
        they are never suppressed:

        >>> pprint(block["coverage"])
        {'frac_measurements_mapped': 0.925926,
         'frac_observed_codes_with_parents': 0.666667,
         'n_measurements_mapped': 500,
         'n_measurements_unmapped': 40,
         'n_observed_codes': 3,
         'n_observed_codes_with_parents': 2,
         'n_observed_codes_without_parents': 1}

        LOINC clears the subject floor and is reported by name, with digests of its key set and its
        numeric moments. SNOMED, reached by only two subjects, is rolled up -- its name is withheld
        along with its counts:

        >>> pprint(block["by_vocabulary"])
        {'<suppressed>': {'n_child_codes': 1,
                          'n_measurements': 20,
                          'n_vocabularies': 1},
         'LOINC': {'keys_digest': 'sha256:...',
                   'n_child_codes': 2,
                   'n_distinct_keys': 2,
                   'n_measurements': 500,
                   'n_subjects': 50,
                   'numeric': {'mean': 88.5, 'n_finite': 480, 'std': 12.25},
                   'recognized': True}}

        With no code metadata at all, the block records its own absence rather than vanishing, so a
        later comparison can flag metadata that disappeared:

        >>> with yaml_disk({"data": {}}) as d:
        ...     _ontology_block(d, None, None, set(), None, ExtractConfig())
        {'present': False}

        As does code metadata that exists but declares no ``parent_codes`` column:

        >>> with yaml_disk({"metadata": {"codes.parquet": {"code": ["HR"]}}}) as d:
        ...     _ontology_block(d, None, None, {"HR"}, 10, ExtractConfig())
        {'present': False}

        With ``emit_ontology_keys``, each reported vocabulary also lists its keys in full. Those
        name published terminology concepts, so this is off by default for output size rather than
        for privacy:

        >>> spec = {"metadata": {"codes.parquet": {
        ...     "code": ["HR"], "parent_codes": [["LOINC/8867-4", "LOINC/8310-5"]],
        ... }}}
        >>> volumes = pl.DataFrame([{"vocabulary": "LOINC", "n_measurements": 500,
        ...                          "n_subjects": 50, "n_child_codes": 1,
        ...                          "numeric_n_finite": 0, "numeric_mean": None,
        ...                          "numeric_std": None}])
        >>> with yaml_disk(spec) as d:
        ...     keyed = _ontology_block(d, volumes, None, {"HR"}, 500,
        ...                             ExtractConfig(min_cell_size=20, emit_ontology_keys=True))
        >>> keyed["by_vocabulary"]["LOINC"]["keys"]
        ['8310-5', '8867-4']
    """
    fp = meds_dir / meds.code_metadata_filepath
    if not fp.is_file():
        return {"present": False}
    lf = pl.scan_parquet(fp)
    if "parent_codes" not in lf.collect_schema():
        return {"present": False}

    per_code_parents = lf.select("code", n_parents=pl.col("parent_codes").list.len().fill_null(0)).collect()
    links = (
        lf.select("code", "parent_codes")
        .explode("parent_codes", empty_as_null=True)
        .drop_nulls("parent_codes")
        .rename({"parent_codes": "parent_code"})
        .select("code", "parent_code", vocabulary_expr(), key_expr())
        .collect()
    )
    per_code_vocabularies = links.group_by("code").agg(n_vocabularies=pl.col("vocabulary").n_unique())

    volumes = {r["vocabulary"]: r for r in by_vocabulary.to_dicts()} if by_vocabulary is not None else {}
    rows = []
    for vocabulary, group in links.group_by("vocabulary"):
        name = vocabulary[0]
        volume = volumes.get(name, {})
        keys = sorted(set(group["key"].to_list()))
        row = {
            "vocabulary": name,
            "recognized": is_recognized(name),
            "n_distinct_keys": len(keys),
            "n_child_codes": group["code"].n_unique(),
            "n_measurements": volume.get("n_measurements"),
            "n_subjects": volume.get("n_subjects"),
            "keys_digest": digest(keys),
        }
        if (volume.get("numeric_n_finite") or 0) >= cfg.min_numeric_cell_size:
            row["numeric"] = {
                "n_finite": volume["numeric_n_finite"],
                "mean": _round_sig(volume["numeric_mean"]),
                "std": _round_sig(volume["numeric_std"]),
            }
        if cfg.emit_ontology_keys:
            row["keys"] = keys
        rows.append(row)

    mapped_codes = set(links["code"].to_list())
    n_observed = len(observed_codes)
    n_observed_mapped = len(observed_codes & mapped_codes)
    n_unmapped = int(unmapped["n_measurements"][0]) if unmapped is not None else None
    # Derived, not summed: a code with parents in two vocabularies appears under both, so summing
    # `by_vocabulary` would double-count it and could push the fraction above 1.
    n_mapped = (
        None if n_unmapped is None or n_total_measurements is None else n_total_measurements - n_unmapped
    )

    return {
        "present": True,
        "n_codes_with_parents": int((per_code_parents["n_parents"] > 0).sum()),
        "n_parent_links": links.height,
        "n_distinct_parents": links["parent_code"].n_unique(),
        "n_parents_without_vocabulary": int((links["vocabulary"] == NO_VOCABULARY).sum()),
        "n_vocabularies": links["vocabulary"].n_unique(),
        "n_vocabularies_recognized": sum(1 for r in rows if r["recognized"]),
        "n_codes_in_multiple_vocabularies": int((per_code_vocabularies["n_vocabularies"] > 1).sum()),
        "parents_per_code": _distribution(per_code_parents["n_parents"].cast(pl.Float64), cfg),
        "vocabularies_per_code": _distribution(
            per_code_parents.join(per_code_vocabularies, on="code", how="left")["n_vocabularies"]
            .fill_null(0)
            .cast(pl.Float64),
            cfg,
        ),
        "coverage": {
            "n_observed_codes": n_observed,
            "n_observed_codes_with_parents": n_observed_mapped,
            "n_observed_codes_without_parents": n_observed - n_observed_mapped,
            "frac_observed_codes_with_parents": _round_sig(
                n_observed_mapped / n_observed if n_observed else None
            ),
            "n_measurements_mapped": n_mapped,
            "n_measurements_unmapped": n_unmapped,
            "frac_measurements_mapped": _round_sig(
                n_mapped / n_total_measurements if n_mapped is not None and n_total_measurements else None
            ),
        },
        "by_vocabulary": _suppress_keyed(
            rows, "vocabulary", cfg, ("n_measurements", "n_child_codes"), "n_vocabularies"
        ),
    }


def _code_metadata_block(
    meds_dir: Path,
    per_code: pl.DataFrame,
    by_vocabulary: pl.DataFrame | None,
    unmapped: pl.DataFrame | None,
    n_total_measurements: int | None,
    cfg: ExtractConfig,
) -> dict[str, Any]:
    """Summarize ``metadata/codes.parquet`` and its agreement with the observed data.

    MEDS requires every code in the data to appear in the code metadata. The two coverage counts here
    are the direct test of that invariant, and they catch a whole class of ETL bug -- a metadata join
    that dropped rows -- that no statistic over the data alone can see.

    Description *lengths* are summarized; description text never is.

    Examples:
        >>> from pprint import pprint
        >>> spec = {"metadata": {"codes.parquet": {
        ...     "code": ["HR", "GLU", "UNUSED"],
        ...     "description": ["Heart rate", None, "Never observed"],
        ...     "parent_codes": [["LOINC/8867-4"], None, None],
        ... }}}
        >>> observed = pl.DataFrame({"code": ["HR", "GLU", "NOT_IN_METADATA"]})
        >>> with yaml_disk(spec) as d:
        ...     block = _code_metadata_block(d, observed, None, None, None, ExtractConfig())
        >>> pprint({k: v for k, v in block.items() if k not in ("ontology", "description_length")})
        {'n_codes': 3,
         'n_data_codes_missing_from_metadata': 1,
         'n_metadata_codes_absent_from_data': 1,
         'n_with_description': 2,
         'n_with_parent_codes': 1,
         'present': True,
         'vocabulary_digest': 'sha256:...'}

        >>> with yaml_disk({"data": {}}) as d:
        ...     _code_metadata_block(d, observed, None, None, None, ExtractConfig())
        {'present': False}
    """
    fp = meds_dir / meds.code_metadata_filepath
    if not fp.is_file():
        return {"present": False}

    lf = pl.scan_parquet(fp)
    schema = lf.collect_schema()
    metadata_codes = lf.select("code").collect()["code"]
    observed_codes = set(per_code["code"].to_list())
    metadata_code_set = set(metadata_codes.to_list())

    block: dict[str, Any] = {
        "present": True,
        "n_codes": metadata_codes.len(),
        "n_with_description": (
            int(lf.select(pl.col("description").is_not_null().sum()).collect().item())
            if "description" in schema
            else None
        ),
        "n_with_parent_codes": (
            int(lf.select((pl.col("parent_codes").list.len().fill_null(0) > 0).sum()).collect().item())
            if "parent_codes" in schema
            else None
        ),
        "n_data_codes_missing_from_metadata": len(observed_codes - metadata_code_set),
        "n_metadata_codes_absent_from_data": len(metadata_code_set - observed_codes),
        "vocabulary_digest": (digest(sorted(metadata_code_set)) if cfg.code_metadata_only_digests else None),
    }
    if "description" in schema:
        lengths = lf.select(pl.col("description").str.len_chars()).collect()["description"]
        block["description_length"] = _distribution(lengths.cast(pl.Float64), cfg)
    block["ontology"] = _ontology_block(
        meds_dir, by_vocabulary, unmapped, observed_codes, n_total_measurements, cfg
    )
    return block


def _splits_block(meds_dir: Path, per_subject: pl.DataFrame, cfg: ExtractConfig) -> dict[str, Any]:
    """Summarize the subject-split assignment.

    Args:
        meds_dir: The MEDS dataset root.
        per_subject: Collected per-subject aggregate, used to attribute measurement volume to splits.
        cfg: Extraction configuration.

    Returns:
        The ``splits`` block.

    Examples:
        >>> from pprint import pprint
        >>> spec = {"metadata": {"subject_splits.parquet": {
        ...     "subject_id": [1, 2, 3, 4], "split": ["train", "train", "held_out", "pilot"],
        ... }}}
        >>> per_subject = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 5], "n_measurements": [10, 20, 30, 40],
        ... })
        >>> with yaml_disk(spec) as d:
        ...     pprint(_splits_block(d, per_subject, ExtractConfig(min_cell_size=1)))
        {'by_split': {'held_out': {'n_measurements': 30, 'n_subjects': 1},
                      'pilot': {'n_measurements': 0, 'n_subjects': 1},
                      'train': {'n_measurements': 30, 'n_subjects': 2}},
         'n_splits': 3,
         'n_subjects_in_data_without_split': 1,
         'n_subjects_in_splits_without_data': 1,
         'present': True}

        Subject 4 is assigned a split but absent from the data, and subject 5 is the reverse; both are
        counted rather than silently dropped, since either is an ETL bug.

        >>> with yaml_disk({"data": {}}) as d:
        ...     _splits_block(d, per_subject, ExtractConfig())
        {'present': False}
    """
    fp = meds_dir / meds.subject_splits_filepath
    if not fp.is_file():
        return {"present": False}

    splits = pl.read_parquet(fp).select("subject_id", "split")
    joined = splits.join(per_subject.select("subject_id", "n_measurements"), on="subject_id", how="left")
    rows = (
        joined.group_by("split")
        .agg(
            n_subjects=pl.col("subject_id").n_unique(),
            n_measurements=pl.col("n_measurements").fill_null(0).sum(),
        )
        .to_dicts()
    )
    split_subjects = set(splits["subject_id"].to_list())
    data_subjects = set(per_subject["subject_id"].to_list())
    return {
        "present": True,
        "n_splits": splits["split"].n_unique(),
        "n_subjects_in_data_without_split": len(data_subjects - split_subjects),
        "n_subjects_in_splits_without_data": len(split_subjects - data_subjects),
        "by_split": _suppress_keyed(rows, "split", cfg, ("n_measurements",), "n_splits"),
    }


def _counts_block(overall: pl.DataFrame, per_subject: pl.DataFrame) -> dict[str, Any]:
    """Assemble the headline volume counts.

    These are the numbers the motivating use case turns on: if the dataset version has not changed,
    an ETL revision should not move the subject or measurement count by more than a hair.

    Examples:
        >>> from pprint import pprint
        >>> overall = pl.DataFrame([{
        ...     "n_measurements": 100, "n_subjects": 10, "n_unique_codes": 5,
        ...     "n_static_measurements": 20, "n_null_subject_id": 0, "n_null_code": 0,
        ...     "n_empty_code": 0, "numeric_n_present": 60, "numeric_n_nan": 1,
        ...     "numeric_n_infinite": 0, "numeric_n_finite": 59, "numeric_n_negative": 2,
        ...     "numeric_n_zero": 3, "numeric_mean": 1.23456789, "numeric_std": 0.5,
        ... }])
        >>> per_subject = pl.DataFrame({
        ...     "subject_id": [1, 2], "n_events": [3, 4],
        ...     "has_birth": [True, False], "has_death": [False, False], "n_static": [1, 0],
        ... })
        >>> pprint(_counts_block(overall, per_subject))
        {'n_events': 7,
         'n_measurements': 100,
         'n_static_measurements': 20,
         'n_subjects': 10,
         'n_subjects_with_birth': 1,
         'n_subjects_with_death': 0,
         'n_subjects_with_static': 1,
         'n_unique_codes': 5,
         'numeric': {'mean': 1.23457,
                     'n_finite': 59,
                     'n_infinite': 0,
                     'n_nan': 1,
                     'n_negative': 2,
                     'n_present': 60,
                     'n_zero': 3,
                     'std': 0.5},
         'quality': {'n_empty_code': 0, 'n_null_code': 0, 'n_null_subject_id': 0}}
    """
    row = overall.to_dicts()[0]
    return {
        "n_subjects": row["n_subjects"],
        "n_measurements": row["n_measurements"],
        "n_events": int(per_subject["n_events"].sum()),
        "n_unique_codes": row["n_unique_codes"],
        "n_static_measurements": row["n_static_measurements"],
        "n_subjects_with_static": int((per_subject["n_static"] > 0).sum()),
        "n_subjects_with_birth": int(per_subject["has_birth"].fill_null(False).sum()),
        "n_subjects_with_death": int(per_subject["has_death"].fill_null(False).sum()),
        "numeric": {
            "n_present": row["numeric_n_present"],
            "n_finite": row["numeric_n_finite"],
            "n_nan": row["numeric_n_nan"],
            "n_infinite": row["numeric_n_infinite"],
            "n_negative": row["numeric_n_negative"],
            "n_zero": row["numeric_n_zero"],
            "mean": _round_sig(row["numeric_mean"]),
            "std": _round_sig(row["numeric_std"]),
        },
        "quality": {
            "n_null_subject_id": row["n_null_subject_id"],
            "n_null_code": row["n_null_code"],
            "n_empty_code": row["n_empty_code"],
        },
    }


def _per_subject_block(per_subject: pl.DataFrame, cfg: ExtractConfig) -> dict[str, Any]:
    """Summarize the per-subject distributions.

    Reported as distributions, never as extremes: the subject with the most measurements, or the
    longest record, is exactly the subject a re-identification attempt would start from.

    Examples:
        >>> cfg = ExtractConfig(min_numeric_cell_size=2, quantiles=(0.5,))
        >>> per_subject = pl.DataFrame({
        ...     "n_measurements": [10, 20], "n_events": [2, 4], "n_unique_codes": [3, 5],
        ...     "t_min": [datetime(2020, 1, 1), datetime(2021, 1, 1)],
        ...     "t_max": [datetime(2020, 1, 11), datetime(2021, 1, 3)],
        ... })
        >>> block = _per_subject_block(per_subject, cfg)
        >>> sorted(block)
        ['n_events', 'n_measurements', 'n_unique_codes', 'span_days']
        >>> block["n_measurements"]
        {'n': 2, 'mean': 15.0, 'std': 7.07107, 'quantiles': {'p50': 15.0}}
        >>> block["span_days"]["mean"]
        6.0
    """
    span_days = (per_subject["t_max"] - per_subject["t_min"]).dt.total_seconds().cast(pl.Float64) / 86400.0
    return {
        "n_measurements": _distribution(per_subject["n_measurements"].cast(pl.Float64), cfg),
        "n_events": _distribution(per_subject["n_events"].cast(pl.Float64), cfg),
        "n_unique_codes": _distribution(per_subject["n_unique_codes"].cast(pl.Float64), cfg),
        "span_days": _distribution(span_days, cfg),
    }


def _time_block(
    by_year: pl.DataFrame, time_quantiles: pl.DataFrame | None, cfg: ExtractConfig
) -> dict[str, Any]:
    """Summarize when the dataset's events happened.

    The calendar-year histogram is the workhorse: it is cheap, high-count, and immediately legible
    when something goes wrong, since a timezone or epoch-parsing regression relocates whole years of
    data at once. Years reached by too few subjects are suppressed, which also disposes of the stray
    year-1900 and year-2099 rows that bad date parsing produces -- those show up in the rollup count
    instead of as a reportable cell.

    Examples:
        >>> from pprint import pprint
        >>> by_year = pl.DataFrame({
        ...     "year": [2019, 2020, 1900],
        ...     "n_measurements": [500, 600, 2], "n_subjects": [50, 60, 1],
        ... })
        >>> pprint(_time_block(by_year, None, ExtractConfig(min_cell_size=20)))
        {'by_year': {'2019': {'n_measurements': 500, 'n_subjects': 50},
                     '2020': {'n_measurements': 600, 'n_subjects': 60},
                     '<suppressed>': {'n_measurements': 2, 'n_years': 1}},
         'n_measurements_with_time': 1102,
         'n_years_observed': 3}

        With ``numeric_detail="quantiles"``, full-column time quantiles are added as ISO strings:

        >>> tq = pl.DataFrame([{"qp50": 1577836800000000}])
        >>> _time_block(by_year, tq, ExtractConfig(quantiles=(0.5,)))["quantiles"]
        {'p50': '2020-01-01T00:00:00'}
    """
    rows = [
        {"year": str(r["year"]), "n_measurements": r["n_measurements"], "n_subjects": r["n_subjects"]}
        for r in by_year.sort("year").to_dicts()
    ]
    block: dict[str, Any] = {
        "n_measurements_with_time": int(by_year["n_measurements"].sum()),
        "n_years_observed": by_year.height,
        "by_year": _suppress_keyed(rows, "year", cfg, ("n_measurements",), "n_years"),
    }
    if time_quantiles is not None:
        row = time_quantiles.to_dicts()[0]
        block["quantiles"] = {
            _qkey(q): _iso(
                datetime.datetime.fromtimestamp(row[f"q{_qkey(q)}"] / 1e6, tz=datetime.UTC).replace(
                    tzinfo=None
                )
            )
            if row[f"q{_qkey(q)}"] is not None
            else None
            for q in cfg.quantiles
        }
    return block


def extract_summary_stats(
    meds_dir: str | Path,
    cfg: ExtractConfig | None = None,
    *,
    max_rows_per_chunk: int | None = DEFAULT_MAX_ROWS_PER_CHUNK,
) -> dict[str, Any]:
    """Extract the full summary-statistics document for a MEDS dataset.

    Args:
        meds_dir: Root of the MEDS dataset (the directory containing ``data/`` and ``metadata/``).
        cfg: Extraction configuration. Defaults to :class:`~meds_summary_stats.config.ExtractConfig`.
        max_rows_per_chunk: Roughly how many rows one aggregation pass may hold, which is what bounds
            peak memory; ``None`` reads the whole dataset in one pass. This decides only how the
            work is scheduled, never what it computes, so it is deliberately not part of ``cfg``
            and not recorded in the document.

    Returns:
        A JSON-serializable dict. See :func:`write_summary_stats` for the on-disk form.

    Raises:
        NotAMEDSDatasetError: If ``meds_dir`` contains no readable data shards.
        ValueError: If ``max_rows_per_chunk`` is not positive.

    Examples:
        >>> stats = extract_summary_stats(simple_static_MEDS)
        >>> sorted(stats)
        ['code_metadata', 'codes', 'config', 'counts', 'dataset', 'dataset_digest', 'extracted_at',
         'layout', 'meds_summary_stats_version', 'per_subject', 'splits', 'stats_schema_version',
         'time']

        The digest answers "is this the same dataset at all?", independently of every statistic:

        >>> stats["dataset_digest"]["method"], stats["dataset_digest"]["n_files"]
        ('files', 7)
        >>> stats["dataset_digest"]["combined"]
        'sha256:...'

        The sample dataset has six subjects across four shards:

        >>> stats["counts"]["n_subjects"], stats["counts"]["n_measurements"]
        (6, 62)
        >>> stats["layout"]["n_data_shards"], sorted(stats["layout"]["data_columns"])
        (4, ['code', 'numeric_value', 'subject_id', 'time'])

        It is far below any sane suppression floor, so every keyed cell is withheld and only rollups
        survive -- which is the behavior that matters: a small dataset yields a fingerprint that is
        still comparable but discloses nothing.

        >>> stats["splits"]["by_split"]
        {'<suppressed>': {'n_measurements': 62, 'n_splits': 3}}
        >>> stats["code_metadata"]["ontology"]["by_vocabulary"]
        {'<suppressed>': {'n_child_codes': 2, 'n_measurements': 32, 'n_vocabularies': 1}}
        >>> stats["codes"]["digests"]["alphabetical"]
        'sha256:...'

        The vocabulary is never partitioned by the shape of the code strings, so there is no
        namespace view to suppress in the first place -- only digests and aggregate distributions:

        >>> sorted(stats["codes"])
        ['concentration', 'digests', 'n_codes_below_min_cell_size', 'n_unique_codes', 'occurrence']

        Lowering the floor to 1 makes the same dataset fully legible, which is how the tool is meant
        to be used on data that is already public. The ontology view is the axis that says *where*
        something changed, and its groups come from the dataset's own ``parent_codes``:

        >>> stats = extract_summary_stats(simple_static_MEDS, ExtractConfig(min_cell_size=1))
        >>> sorted(stats["splits"]["by_split"])
        ['held_out', 'train', 'tuning']
        >>> stats["splits"]["by_split"]["train"]
        {'n_measurements': 44, 'n_subjects': 4}
        >>> sorted(stats["code_metadata"]["ontology"]["by_vocabulary"])
        ['LOINC']

        Coverage reports how much of the data the ontology actually reaches, which is how an
        unresolved mapping table announces itself:

        >>> coverage = stats["code_metadata"]["ontology"]["coverage"]
        >>> coverage["n_observed_codes"], coverage["n_observed_codes_without_parents"]
        (11, 9)
        >>> coverage["frac_measurements_mapped"]
        0.516129

        No code string appears anywhere in the document, at any threshold:

        >>> blob = json.dumps(stats)
        >>> any(code in blob for code in ("EYE_COLOR//BROWN", "MEDS_BIRTH", "HR", "TEMP"))
        False

        Memory is bounded by aggregating in chunks, which never changes the result:

        >>> chunked = extract_summary_stats(simple_static_MEDS, ExtractConfig(min_cell_size=1),
        ...                                 max_rows_per_chunk=5)
        >>> {k: v for k, v in chunked.items() if k != "extracted_at"} == {
        ...     k: v for k, v in stats.items() if k != "extracted_at"}
        True
        >>> extract_summary_stats(simple_static_MEDS, max_rows_per_chunk=0)
        Traceback (most recent call last):
            ...
        ValueError: max_rows_per_chunk must be >= 1 or None; got 0
    """
    if max_rows_per_chunk is not None and max_rows_per_chunk < 1:
        raise ValueError(f"max_rows_per_chunk must be >= 1 or None; got {max_rows_per_chunk}")
    meds_dir = Path(meds_dir)
    cfg = cfg or ExtractConfig()

    shards = _shard_paths(meds_dir)
    layout = _layout(meds_dir, shards)
    available = set(layout["data_columns"])
    chunks = _plan_chunks(shards, available, max_rows_per_chunk)

    vocab_map = _code_vocabulary_map(meds_dir)
    frames = _aggregate(chunks, cfg, vocab_map)

    per_subject = frames["per_subject"]
    per_code = frames["per_code"]
    dataset_present, dataset_metadata = _read_dataset_metadata(meds_dir)

    stats: dict[str, Any] = {
        "meds_summary_stats_version": __version__,
        "stats_schema_version": STATS_SCHEMA_VERSION,
        "extracted_at": _iso(datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)),
        "config": cfg.to_dict(),
        "dataset": {"present": dataset_present, **dataset_metadata},
        "dataset_digest": _dataset_digest_block(meds_dir, cfg),
        "layout": layout,
        "counts": _counts_block(frames["overall"], per_subject),
        "per_subject": _per_subject_block(per_subject, cfg),
        "time": _time_block(frames["by_year"], frames.get("time_quantiles"), cfg),
        "splits": _splits_block(meds_dir, per_subject, cfg),
        "codes": _codes_block(per_code, cfg),
        "code_metadata": _code_metadata_block(
            meds_dir,
            per_code,
            frames.get("by_vocabulary"),
            frames.get("unmapped"),
            int(frames["overall"]["n_measurements"][0]),
            cfg,
        ),
    }

    if (numeric_quantiles := frames.get("numeric_quantiles")) is not None:
        row = numeric_quantiles.to_dicts()[0]
        stats["counts"]["numeric"]["quantiles"] = {
            _qkey(q): _round_sig(row[f"q{_qkey(q)}"]) for q in cfg.quantiles
        }
    return stats


def write_summary_stats(stats: dict[str, Any], out_path: str | Path) -> Path:
    """Write a stats document to disk in its canonical form.

    The canonical form is UTF-8 JSON with sorted keys, two-space indentation, and a trailing newline.
    Sorting matters beyond tidiness: it makes the file stable under re-extraction and therefore
    reviewable as a plain ``git diff`` when it is checked in as a baseline.

    Args:
        stats: The document to write.
        out_path: Destination path. Parent directories are created.

    Returns:
        The path written.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = write_summary_stats({"b": 1, "a": {"d": 2, "c": 3}}, Path(tmp) / "nested" / "s.json")
        ...     print(fp.read_text(), end="")
        {
          "a": {
            "c": 3,
            "d": 2
          },
          "b": 1
        }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return out_path


def read_summary_stats(path: str | Path) -> dict[str, Any]:
    """Read a stats document from disk.

    Args:
        path: The file to read.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the file is not a stats document this version understands.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = write_summary_stats({"stats_schema_version": 1, "counts": {}}, Path(tmp) / "s.json")
        ...     sorted(read_summary_stats(fp))
        ['counts', 'stats_schema_version']

        A file from a future format version is rejected rather than misread:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = write_summary_stats({"stats_schema_version": 99}, Path(tmp) / "s.json")
        ...     read_summary_stats(fp)
        Traceback (most recent call last):
            ...
        ValueError: .../s.json has stats_schema_version 99; this build understands 1

        As is a JSON file that is not a stats document at all:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = write_summary_stats({"hello": "world"}, Path(tmp) / "s.json")
        ...     read_summary_stats(fp)
        Traceback (most recent call last):
            ...
        ValueError: .../s.json is not a summary-stats document (no stats_schema_version field)
    """
    path = Path(path)
    stats = json.loads(path.read_text())
    if not isinstance(stats, dict) or "stats_schema_version" not in stats:
        raise ValueError(f"{path} is not a summary-stats document (no stats_schema_version field)")
    if stats["stats_schema_version"] != STATS_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has stats_schema_version {stats['stats_schema_version']}; "
            f"this build understands {STATS_SCHEMA_VERSION}"
        )
    return stats
