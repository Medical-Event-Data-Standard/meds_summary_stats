"""Extraction of a privacy-safe summary fingerprint from a MEDS dataset.

The output is a single JSON document whose every leaf is a scalar, so it flattens cleanly to
``path -> value`` for comparison. Nothing in it identifies a subject, and -- by construction -- nothing
in it republishes the dataset's code vocabulary: codes are reported only through digests, aggregate
distributions, and namespace/ontology rollups that are suppressed below a subject-count floor.

The dataset is read in a small, fixed number of streaming passes regardless of its size; the only
frames materialized in memory are per-subject and per-code, both of which are orders of magnitude
smaller than the data itself.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

import meds
import polars as pl

from . import __version__
from .config import NO_NAMESPACE, NO_VOCABULARY, SUPPRESSED, ExtractConfig
from .digests import digest, format_moment, ordered_by
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


def namespace_expr() -> pl.Expr:
    """Expression bucketing each code into its ``//``-delimited namespace.

    Codes with no ``//`` are pooled under :data:`~meds_summary_stats.config.NO_NAMESPACE` rather than
    forming a namespace of their own. This matters: for a flat vocabulary such as ``HR`` or ``HEIGHT``,
    "the namespace" and "the code" are the same string, so bucketing by namespace would republish the
    vocabulary this package exists to keep unpublished.

    Examples:
        >>> df = pl.DataFrame({"code": ["LAB//GLUCOSE", "LAB//NA", "HR", "A//B//C"]})
        >>> df.select(ns=namespace_expr())["ns"].to_list()
        ['LAB', 'LAB', '<no-namespace>', 'A']
    """
    head = pl.col("code").str.splitn("//", 2).struct.field("field_0")
    return (
        pl.when(pl.col("code").str.contains("//", literal=True))
        .then(head)
        .otherwise(pl.lit(NO_NAMESPACE))
        .alias("namespace")
    )


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


def _aggregate(
    base: pl.LazyFrame, cfg: ExtractConfig, vocab_map: pl.LazyFrame | None
) -> dict[str, pl.DataFrame]:
    """Run every data-scanning aggregate in one batch.

    The queries are collected together so that polars can share the parquet scan and common subplans
    across them: the dataset is read a small fixed number of times rather than once per statistic.

    Args:
        base: The normalized four-column frame.
        cfg: Extraction configuration.
        vocab_map: Optional ``(code, vocabulary)`` mapping to attribute measurements to ontologies.

    Returns:
        A dict of collected frames keyed by aggregate name.

    Examples:
        >>> spec = {"data": {"0.parquet": {
        ...     "subject_id": [1, 1, 2], "code": ["LAB//A", "LAB//A", "HR"],
        ...     "numeric_value": [1.0, 3.0, 60.0],
        ... }}}
        >>> with yaml_disk(spec) as d:
        ...     frames = _aggregate(_base_frame(_shard_paths(d), {"subject_id", "code", "numeric_value"}),
        ...                         ExtractConfig(), None)
        >>> sorted(frames)
        ['by_namespace', 'by_year', 'overall', 'per_code', 'per_subject']
        >>> frames["per_code"].sort("code").select("code", "n_measurements", "n_subjects")
        shape: (2, 3)
        ┌────────┬────────────────┬────────────┐
        │ code   ┆ n_measurements ┆ n_subjects │
        │ ---    ┆ ---            ┆ ---        │
        │ str    ┆ u32            ┆ u32        │
        ╞════════╪════════════════╪════════════╡
        │ HR     ┆ 1              ┆ 1          │
        │ LAB//A ┆ 2              ┆ 1          │
        └────────┴────────────────┴────────────┘
        >>> frames["overall"].select("n_measurements", "n_subjects", "numeric_mean").to_dicts()
        [{'n_measurements': 3, 'n_subjects': 2, 'numeric_mean': 21.333333333333332}]
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
        "by_namespace": base.group_by(namespace_expr()).agg(
            n_measurements=pl.len(),
            n_subjects=pl.col("subject_id").n_unique(),
            n_codes=pl.col("code").n_unique(),
            numeric_n_finite=finite.count(),
            numeric_mean=finite.mean(),
            numeric_std=finite.std(),
        ),
        "by_year": base.drop_nulls("time")
        .group_by(pl.col("time").dt.year().alias("year"))
        .agg(n_measurements=pl.len(), n_subjects=pl.col("subject_id").n_unique()),
    }

    if vocab_map is not None:
        queries["by_vocabulary"] = (
            base.join(vocab_map, on="code", how="inner")
            .group_by("vocabulary")
            .agg(
                n_measurements=pl.len(),
                n_subjects=pl.col("subject_id").n_unique(),
                n_child_codes=pl.col("code").n_unique(),
            )
        )

    if cfg.numeric_detail == "quantiles":
        queries["numeric_quantiles"] = base.select(
            **{f"q{_qkey(q)}": finite.quantile(q, "linear") for q in cfg.quantiles}
        )
        queries["time_quantiles"] = base.drop_nulls("time").select(
            **{f"q{_qkey(q)}": pl.col("time").dt.epoch("us").quantile(q, "linear") for q in cfg.quantiles}
        )

    names = list(queries)
    collected = pl.collect_all([queries[n] for n in names], engine="streaming")
    return dict(zip(names, collected, strict=True))


def _codes_block(per_code: pl.DataFrame, by_namespace: pl.DataFrame, cfg: ExtractConfig) -> dict[str, Any]:
    """Summarize the code vocabulary without emitting any code.

    Three complementary views are reported, because they fail differently:

    * **Digests** answer "is this the same vocabulary?" exactly. The alphabetical digest changes if and
      only if the *set* of codes changed. The frequency-ordered digests additionally change if the
      relative ranking shifted, which makes them sensitive tripwires but noisy ones -- hence the
      thresholded variants, restricted to codes common enough that their ordering is stable.
    * **Occurrence distributions** answer "did the shape of the vocabulary change?" and survive
      benign additions to the tail.
    * **Namespace rollups** answer "*where* did it change?", which is what a person debugging a failed
      comparison actually needs.

    Args:
        per_code: Collected per-code aggregate.
        by_namespace: Collected per-namespace aggregate.
        cfg: Extraction configuration.

    Returns:
        The ``codes`` block of the stats document.

    Examples:
        >>> from pprint import pprint
        >>> per_code = pl.DataFrame({
        ...     "code": ["LAB//A", "LAB//B", "HR"],
        ...     "n_measurements": [100, 50, 10],
        ...     "n_subjects": [40, 30, 5],
        ...     "numeric_n_finite": [100, 0, 10],
        ...     "numeric_mean": [1.5, None, 60.0],
        ...     "numeric_std": [0.5, None, 5.0],
        ... })
        >>> by_ns = pl.DataFrame({
        ...     "namespace": ["LAB", "<no-namespace>"],
        ...     "n_measurements": [150, 10], "n_subjects": [50, 5], "n_codes": [2, 1],
        ...     "numeric_n_finite": [100, 10], "numeric_mean": [1.5, 60.0], "numeric_std": [0.5, 5.0],
        ... })
        >>> block = _codes_block(per_code, by_ns, ExtractConfig(min_cell_size=20))

        No code string survives anywhere in the output:

        >>> "LAB//A" in json.dumps(block) or "HR" in json.dumps(block)
        False

        The rare code is excluded from the reportable count but still counted:

        >>> block["n_unique_codes"], block["n_codes_below_min_cell_size"]
        (3, 1)

        The un-namespaced code is pooled, and its namespace entry is suppressed for having too few
        subjects -- so a flat vocabulary leaks nothing through the namespace view:

        >>> pprint(block["by_namespace"])
        {'<suppressed>': {'n_codes': 1, 'n_measurements': 10, 'n_namespaces': 1},
         'LAB': {'n_codes': 2,
                 'n_measurements': 150,
                 'n_subjects': 50,
                 'numeric': {'mean': 1.5, 'n_finite': 100, 'std': 0.5}}}

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

    namespace_rows = []
    for row in by_namespace.to_dicts():
        entry = {
            "namespace": row["namespace"],
            "n_measurements": row["n_measurements"],
            "n_subjects": row["n_subjects"],
            "n_codes": row["n_codes"],
        }
        if (row["numeric_n_finite"] or 0) >= cfg.min_numeric_cell_size:
            entry["numeric"] = {
                "n_finite": row["numeric_n_finite"],
                "mean": _round_sig(row["numeric_mean"]),
                "std": _round_sig(row["numeric_std"]),
            }
        namespace_rows.append(entry)

    return {
        "n_unique_codes": per_code.height,
        "n_codes_below_min_cell_size": int((per_code["n_subjects"] < cfg.min_cell_size).sum()),
        "n_codes_without_namespace": int(
            by_namespace.filter(pl.col("namespace") == NO_NAMESPACE)["n_codes"].sum()
        ),
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
        "by_namespace": _suppress_keyed(
            namespace_rows, "namespace", cfg, ("n_measurements", "n_codes"), "n_namespaces"
        ),
    }


def _ontology_block(
    meds_dir: Path,
    by_vocabulary: pl.DataFrame | None,
    cfg: ExtractConfig,
) -> dict[str, Any]:
    """Summarize the external ontologies that ``parent_codes`` links into.

    Parent codes name concepts in published terminologies -- LOINC, SNOMED, ICD -- so unlike the
    dataset's own vocabulary they carry no information about who is in the dataset. That makes them
    the one place this package can report identifiers rather than digests, and they are a strong ETL
    signal: a concept-mapping table that silently stops resolving shows up here as a vocabulary whose
    link count collapsed, which no aggregate count elsewhere would reveal.

    Suppression still applies to the *statistics*: a vocabulary reached by too few subjects is rolled
    up rather than reported, because its measurement count would otherwise describe a tiny cohort.

    Args:
        meds_dir: The MEDS dataset root.
        by_vocabulary: Collected per-vocabulary data aggregate, or ``None`` if unavailable.
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
        ... })
        >>> with yaml_disk(spec) as d:
        ...     block = _ontology_block(d, by_vocab, ExtractConfig(min_cell_size=20))
        >>> pprint({k: v for k, v in block.items() if k != "by_vocabulary"})
        {'frac_codes_with_parents': 0.666667,
         'n_codes_with_parents': 2,
         'n_distinct_parents': 3,
         'n_parent_links': 3,
         'n_parents_without_vocabulary': 0,
         'n_vocabularies': 2,
         'n_vocabularies_recognized': 2,
         'parents_per_code': {'n': 3, 'suppressed': True},
         'present': True}

        LOINC clears the subject floor and is reported by name, with a digest of its key set. SNOMED,
        reached by only two subjects, is rolled up -- its name is withheld along with its counts:

        >>> pprint(block["by_vocabulary"])
        {'<suppressed>': {'n_child_codes': 1,
                          'n_measurements': 20,
                          'n_vocabularies': 1},
         'LOINC': {'keys_digest': 'sha256:...',
                   'n_child_codes': 2,
                   'n_distinct_keys': 2,
                   'n_measurements': 500,
                   'n_subjects': 50,
                   'recognized': True}}

        With no code metadata at all, the block records its own absence rather than vanishing, so a
        later comparison can flag metadata that disappeared:

        >>> with yaml_disk({"data": {}}) as d:
        ...     _ontology_block(d, None, ExtractConfig())
        {'present': False}
    """
    fp = meds_dir / meds.code_metadata_filepath
    if not fp.is_file():
        return {"present": False}
    lf = pl.scan_parquet(fp)
    if "parent_codes" not in lf.collect_schema():
        return {"present": False}

    n_codes = lf.select(pl.len()).collect().item()
    per_code_parents = lf.select("code", n_parents=pl.col("parent_codes").list.len().fill_null(0)).collect()
    links = (
        lf.select("code", "parent_codes")
        .explode("parent_codes", empty_as_null=True)
        .drop_nulls("parent_codes")
        .rename({"parent_codes": "parent_code"})
        .select("code", "parent_code", vocabulary_expr(), key_expr())
        .collect()
    )

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
        if cfg.emit_ontology_keys:
            row["keys"] = keys
        rows.append(row)

    return {
        "present": True,
        "n_codes_with_parents": int((per_code_parents["n_parents"] > 0).sum()),
        "frac_codes_with_parents": _round_sig(
            int((per_code_parents["n_parents"] > 0).sum()) / n_codes if n_codes else None
        ),
        "n_parent_links": links.height,
        "n_distinct_parents": links["parent_code"].n_unique(),
        "n_parents_without_vocabulary": int((links["vocabulary"] == NO_VOCABULARY).sum()),
        "n_vocabularies": links["vocabulary"].n_unique(),
        "n_vocabularies_recognized": sum(1 for r in rows if r["recognized"]),
        "parents_per_code": _distribution(per_code_parents["n_parents"].cast(pl.Float64), cfg),
        "by_vocabulary": _suppress_keyed(
            rows, "vocabulary", cfg, ("n_measurements", "n_child_codes"), "n_vocabularies"
        ),
    }


def _code_metadata_block(
    meds_dir: Path,
    per_code: pl.DataFrame,
    by_vocabulary: pl.DataFrame | None,
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
        ...     block = _code_metadata_block(d, observed, None, ExtractConfig())
        >>> pprint({k: v for k, v in block.items() if k not in ("ontology", "description_length")})
        {'n_codes': 3,
         'n_data_codes_missing_from_metadata': 1,
         'n_metadata_codes_absent_from_data': 1,
         'n_with_description': 2,
         'n_with_parent_codes': 1,
         'present': True,
         'vocabulary_digest': 'sha256:...'}

        >>> with yaml_disk({"data": {}}) as d:
        ...     _code_metadata_block(d, observed, None, ExtractConfig())
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
    block["ontology"] = _ontology_block(meds_dir, by_vocabulary, cfg)
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


def extract_summary_stats(meds_dir: str | Path, cfg: ExtractConfig | None = None) -> dict[str, Any]:
    """Extract the full summary-statistics document for a MEDS dataset.

    Args:
        meds_dir: Root of the MEDS dataset (the directory containing ``data/`` and ``metadata/``).
        cfg: Extraction configuration. Defaults to :class:`~meds_summary_stats.config.ExtractConfig`.

    Returns:
        A JSON-serializable dict. See :func:`write_summary_stats` for the on-disk form.

    Raises:
        NotAMEDSDatasetError: If ``meds_dir`` contains no readable data shards.

    Examples:
        >>> stats = extract_summary_stats(simple_static_MEDS)
        >>> sorted(stats)
        ['code_metadata', 'codes', 'config', 'counts', 'dataset', 'extracted_at', 'layout',
         'meds_summary_stats_version', 'per_subject', 'splits', 'stats_schema_version', 'time']

        The sample dataset has six subjects across four shards:

        >>> stats["counts"]["n_subjects"], stats["counts"]["n_measurements"]
        (6, 62)
        >>> stats["layout"]["n_data_shards"], sorted(stats["layout"]["data_columns"])
        (4, ['code', 'numeric_value', 'subject_id', 'time'])

        It is far below any sane suppression floor, so every keyed cell is withheld and only rollups
        survive -- which is the behavior that matters: a small dataset yields a fingerprint that is
        still comparable but discloses nothing.

        >>> stats["codes"]["by_namespace"]
        {'<suppressed>': {'n_codes': 11, 'n_measurements': 62, 'n_namespaces': 3}}
        >>> stats["splits"]["by_split"]
        {'<suppressed>': {'n_measurements': 62, 'n_splits': 3}}
        >>> stats["codes"]["digests"]["alphabetical"]
        'sha256:...'

        Lowering the floor to 1 makes the same dataset fully legible, which is how the tool is meant
        to be used on data that is already public:

        >>> stats = extract_summary_stats(simple_static_MEDS, ExtractConfig(min_cell_size=1))
        >>> sorted(stats["codes"]["by_namespace"])
        ['<no-namespace>', 'ADMISSION', 'EYE_COLOR']
        >>> sorted(stats["splits"]["by_split"])
        ['held_out', 'train', 'tuning']
        >>> stats["splits"]["by_split"]["train"]
        {'n_measurements': 44, 'n_subjects': 4}

        No code string appears anywhere in the document, at any threshold:

        >>> blob = json.dumps(stats)
        >>> any(code in blob for code in ("EYE_COLOR//BROWN", "MEDS_BIRTH", "HR", "TEMP"))
        False
    """
    meds_dir = Path(meds_dir)
    cfg = cfg or ExtractConfig()

    shards = _shard_paths(meds_dir)
    layout = _layout(meds_dir, shards)
    available = set(layout["data_columns"])
    base = _base_frame(shards, available)

    vocab_map = _code_vocabulary_map(meds_dir)
    frames = _aggregate(base, cfg, vocab_map)

    per_subject = frames["per_subject"]
    per_code = frames["per_code"]
    dataset_present, dataset_metadata = _read_dataset_metadata(meds_dir)

    stats: dict[str, Any] = {
        "meds_summary_stats_version": __version__,
        "stats_schema_version": STATS_SCHEMA_VERSION,
        "extracted_at": _iso(datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)),
        "config": cfg.to_dict(),
        "dataset": {"present": dataset_present, **dataset_metadata},
        "layout": layout,
        "counts": _counts_block(frames["overall"], per_subject),
        "per_subject": _per_subject_block(per_subject, cfg),
        "time": _time_block(frames["by_year"], frames.get("time_quantiles"), cfg),
        "splits": _splits_block(meds_dir, per_subject, cfg),
        "codes": _codes_block(per_code, frames["by_namespace"], cfg),
        "code_metadata": _code_metadata_block(meds_dir, per_code, frames.get("by_vocabulary"), cfg),
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
