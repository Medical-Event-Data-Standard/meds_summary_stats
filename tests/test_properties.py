"""Property-based tests.

Doctests pin behavior on chosen examples; these pin the invariants that must hold on *every* input.
Three families:

1. **Privacy invariants.** Over arbitrary generated MEDS datasets, no code string may appear in the
    output and no reported cell may fall below the suppression floor. These are the claims the
    package makes, and an example-based test can only ever fail to falsify them.
2. **Digest and comparison algebra.** Determinism, order sensitivity, tie-breaking, and the
    threshold semantics that the tolerance policy relies on.
3. **Cross-implementation agreement.** The scalar and vectorized parsers of ``parent_codes`` must
    agree everywhere, since they are separate code paths for the same rule. Likewise, the
    per-subject measurement rate must agree with a row-by-row reading of its definition.
"""

from __future__ import annotations

import itertools
import json
import math
import string
from collections import defaultdict
from datetime import datetime, timedelta

import hypothesis.strategies as st
import meds
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from polars.testing import assert_frame_equal

from meds_summary_stats.compare import compare_summary_stats, relative_delta
from meds_summary_stats.config import SUPPRESSED, ExtractConfig
from meds_summary_stats.digests import digest, format_moment, ordered_by
from meds_summary_stats.extract import (
    _aggregate,
    _code_vocabulary_map,
    _distribution,
    _plan_chunks,
    _qkey,
    _shard_paths,
    extract_summary_stats,
)
from meds_summary_stats.flatten import escape, flatten, matches, specificity
from meds_summary_stats.ontology import key_expr, split_parent_code, vocabulary_expr

# Codes are drawn to look like real MEDS vocabularies: some carrying the `//` convention many ETLs
# use, some entirely flat. Nothing in the package may treat those two shapes differently -- MEDS
# does not require the convention -- so the strategy deliberately mixes them.
_CODE_ALPHABET = string.ascii_uppercase + string.digits + "_-."
codes = st.one_of(
    st.text(_CODE_ALPHABET, min_size=1, max_size=8),
    st.builds(
        "{}//{}".format,
        st.sampled_from(["LAB", "DIAGNOSIS", "MEDICATION", "ADMISSION"]),
        st.text(_CODE_ALPHABET, min_size=1, max_size=8),
    ),
)

parent_codes = st.text(string.ascii_letters + string.digits + "/_-.", min_size=0, max_size=20)

# Quantile levels for the extremes property: the defaults plus the deeper tails a user might
# configure, which need even more observations before they stop touching the extremes.
_QUANTILE_CHOICES = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)


@st.composite
def meds_datasets(draw, min_subjects: int = 1, max_subjects: int = 40):
    """Generate a small but structurally valid MEDS dataset as column maps.

    Returns:
        A ``(data_rows, code_rows)`` pair ready to be written as parquet.
    """
    n_subjects = draw(st.integers(min_subjects, max_subjects))
    vocabulary = draw(st.lists(codes, min_size=1, max_size=12, unique=True))
    n_shards = draw(st.integers(1, 3))

    rows = []
    for subject_id in range(1000, 1000 + n_subjects):
        for _ in range(draw(st.integers(1, 6))):
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": draw(
                        st.one_of(
                            st.none(),
                            st.datetimes(datetime(1990, 1, 1), datetime(2024, 12, 31)),
                        )
                    ),
                    "code": draw(st.sampled_from(vocabulary)),
                    "numeric_value": draw(
                        st.one_of(st.none(), st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
                    ),
                }
            )

    code_rows = [
        {
            "code": code,
            "description": None,
            "parent_codes": draw(st.lists(parent_codes, max_size=2)) or None,
        }
        for code in vocabulary
    ]
    return rows, code_rows, n_shards


def _rename(code: str, index: int) -> str:
    """Rewrite a code to a fresh, unique name.

    The whole string is replaced, including any ``//`` prefix. Nothing in the emitted document may
    depend on the internal structure of a code, so nothing needs preserving.

    Examples:
        >>> _rename("LAB//GLUCOSE", 0), _rename("HR", 1)
        ('RENAMED0', 'RENAMED1')
    """
    return f"RENAMED{index}"


def _write(tmp_path, rows, code_rows, n_shards, scatter: bool = False):
    """Write generated rows out as a sharded MEDS dataset.

    By default each subject is kept whole within one shard, as MEDS ETLs produce. With ``scatter``,
    rows are dealt round-robin across shards instead, so a subject spans several of them -- a
    non-conformant layout that extraction must still read correctly.
    """
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)

    frame = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
        },
        orient="row" if rows and isinstance(rows[0], tuple) else None,
    ).sort("subject_id", "time", nulls_last=False)

    subjects = frame["subject_id"].unique().sort().to_list()
    for shard in range(n_shards):
        if scatter:
            part = frame.gather_every(n_shards, offset=shard)
        else:
            part = frame.filter(pl.col("subject_id").is_in(subjects[shard::n_shards]))
        part.write_parquet(tmp_path / "data" / f"{shard}.parquet")

    pl.DataFrame(
        code_rows,
        schema={"code": pl.String, "description": pl.String, "parent_codes": pl.List(pl.String)},
    ).write_parquet(tmp_path / "metadata" / "codes.parquet")
    (tmp_path / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": "generated"}))
    return tmp_path


class TestPrivacyInvariants:
    """The guarantees the package exists to provide, over arbitrary datasets."""

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets())
    def test_renaming_the_vocabulary_does_not_change_the_document_shape(self, dataset, tmp_path_factory):
        """Renaming every code must leave the set of emitted paths untouched.

        This is the precise statement of "no code is ever emitted", and it is the formulation that
        works. A substring scan of the serialized document cannot: hypothesis finds within seconds
        that a code named ``000`` occurs inside almost every sha256 digest, a code named ``0``
        equals a shard name, and a code named ``S`` occurs inside the dtype string ``String``. None
        of those are leaks, and no amount of length-guarding separates the real cases from the
        coincidences.

        Rename-invariance has no such ambiguity. If a code reached a key or a string value, changing
        the code would change the document's *shape*; if it only reached a digest, the shape is
        identical and only values move. The whole code is replaced, including any ``//`` prefix:
        MEDS does not require that convention, so nothing in the document is permitted to depend on
        it, and a rename that preserved it would not test that.

        Run with the floor at 1 (suppression effectively disabled), because the guarantee must be
        structural. If it only held because the floor happened to be high, it would not be a
        guarantee at all.
        """
        rows, code_rows, n_shards = dataset
        vocabulary = sorted({r["code"] for r in code_rows})
        renames = {code: _rename(code, i) for i, code in enumerate(vocabulary)}

        cfg = ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)
        original = extract_summary_stats(
            _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards), cfg
        )
        renamed = extract_summary_stats(
            _write(
                tmp_path_factory.mktemp("meds"),
                [{**r, "code": renames[r["code"]]} for r in rows],
                [{**r, "code": renames[r["code"]]} for r in code_rows],
                n_shards,
            ),
            cfg,
        )

        original_paths = set(flatten(original)) - {"extracted_at"}
        renamed_paths = set(flatten(renamed)) - {"extracted_at"}
        assert original_paths == renamed_paths, (
            "renaming the vocabulary changed the document's shape, so a code reached a key: "
            f"{sorted(original_paths ^ renamed_paths)}"
        )

        # And the converse: the vocabulary must actually be *represented* somewhere, or the property
        # above would hold vacuously for a tool that ignored codes entirely.
        assert original["codes"]["digests"]["alphabetical"] != renamed["codes"]["digests"]["alphabetical"], (
            "renaming the vocabulary did not change its digest"
        )

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets(min_subjects=5), floor=st.integers(2, 30))
    def test_reported_cells_always_clear_the_floor(self, dataset, floor, tmp_path_factory):
        """Every keyed cell that survives must be backed by at least ``min_cell_size`` subjects."""
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        stats = extract_summary_stats(root, ExtractConfig(min_cell_size=floor))

        blocks = [
            stats["time"]["by_year"],
            stats["code_metadata"]["ontology"].get("by_vocabulary", {}),
        ]
        for block in blocks:
            for key, entry in block.items():
                if key == SUPPRESSED:
                    assert "n_subjects" not in entry
                    continue
                assert entry["n_subjects"] >= floor, (key, entry, floor)

    @settings(max_examples=300, deadline=None)
    @given(
        values=st.lists(
            st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=150
        ),
        floor=st.integers(1, 30),
        quantiles=st.lists(st.sampled_from(_QUANTILE_CHOICES), min_size=1, max_size=7, unique=True),
        up=st.floats(1, 1e6),
        down=st.floats(1, 1e6),
    )
    def test_no_quantile_depends_on_the_smallest_or_largest_value(self, values, floor, quantiles, up, down):
        """Invariant 3, stated as what it protects: the extremes cannot be read back out.

        Checking for keys named ``min`` and ``max`` is not enough. With linear interpolation, a
        quantile near the tail of a small sample is a weighted mix of the two most extreme values,
        and two such quantiles in the same interval pin both values down exactly -- at ``n = 20``,
        ``p95`` and ``p99`` give back the maximum to six significant digits.

        So the property is stated directly: making the largest value larger and the smallest value
        smaller must leave every emitted quantile unchanged. The converse is checked too, so the
        guarantee cannot be bought by withholding quantiles that were safe to report: each one
        withheld really would have moved.
        """
        cfg = ExtractConfig(min_numeric_cell_size=floor, quantiles=tuple(quantiles))
        ordered = sorted(values)
        raised, lowered = list(ordered), list(ordered)
        raised[-1] += up
        lowered[0] -= down
        both = list(raised)
        if len(both) > 1:
            both[0] -= down

        before = _distribution(pl.Series(ordered), cfg)
        if len(ordered) < floor:
            assert before == {"n": len(ordered), "suppressed": True}
            return

        after = _distribution(pl.Series(both), cfg)
        assert before["quantiles"] == after["quantiles"], (before, after)

        for q in cfg.quantiles:
            if _qkey(q) in before["quantiles"]:
                continue
            raw = pl.Series(ordered).quantile(q, "linear")
            moved = {pl.Series(raised).quantile(q, "linear"), pl.Series(lowered).quantile(q, "linear")}
            assert moved != {raw}, f"{_qkey(q)} withheld at n={len(ordered)} but touches neither extreme"

    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets(), up=st.floats(1, 1e5), days=st.integers(1, 3650))
    def test_full_column_quantiles_never_depend_on_the_extremes(self, dataset, up, days, tmp_path_factory):
        """The same guarantee for ``numeric_detail: quantiles``, which takes a separate code path.

        Full-column quantiles over ``numeric_value`` and ``time`` are computed over the whole
        dataset rather than by :func:`_distribution`, so they need their own check. Moving the
        largest and smallest finite value, and the latest and earliest timestamp, must change no
        emitted quantile.
        """
        rows, code_rows, n_shards = dataset
        cfg = ExtractConfig(min_cell_size=1, min_numeric_cell_size=1, numeric_detail="quantiles")
        moved = [dict(r) for r in rows]
        for field, shift in (("numeric_value", up), ("time", timedelta(days=days))):
            present = [i for i, r in enumerate(moved) if r[field] is not None]
            if not present:
                continue
            hi = max(present, key=lambda i: moved[i][field])
            lo = min(present, key=lambda i: moved[i][field])
            moved[hi][field] += shift
            if lo != hi:
                moved[lo][field] -= shift

        before = extract_summary_stats(
            _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards), cfg
        )
        after = extract_summary_stats(
            _write(tmp_path_factory.mktemp("meds"), moved, code_rows, n_shards), cfg
        )
        assert before["counts"]["numeric"]["quantiles"] == after["counts"]["numeric"]["quantiles"]
        assert before["time"]["quantiles"] == after["time"]["quantiles"]

    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets())
    def test_a_document_always_compares_equal_to_itself(self, dataset, tmp_path_factory):
        """Self-comparison must pass for every dataset.

        A false positive here is a false alarm in CI.
        """
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        stats = extract_summary_stats(root)
        report = compare_summary_stats(stats, stats)
        assert report.status == "pass", [f.path for f in report.findings]

    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets())
    def test_extraction_is_deterministic(self, dataset, tmp_path_factory):
        """Two extractions of one dataset must agree on every field but the timestamp."""
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        first, second = extract_summary_stats(root), extract_summary_stats(root)
        first.pop("extracted_at")
        second.pop("extracted_at")
        assert first == second


@st.composite
def meds_datasets_with_life_events(draw):
    """A generated dataset in which each subject may also carry a birth and a death row.

    ``meds_datasets`` never draws the MEDS life-event codes, and they are what
    ``per_subject.measurements_per_day`` treats specially, so they are added here at arbitrary times,
    including none at all and times far outside the subject's other rows.
    """
    rows, code_rows, n_shards = draw(meds_datasets())
    moments = st.one_of(st.none(), st.datetimes(datetime(1900, 1, 1), datetime(2030, 12, 31)))
    for subject_id in sorted({r["subject_id"] for r in rows}):
        for code in (meds.birth_code, meds.death_code):
            if draw(st.booleans()):
                rows.append(
                    {"subject_id": subject_id, "time": draw(moments), "code": code, "numeric_value": None}
                )
    return rows, code_rows, n_shards


def _reference_measurements_per_day(rows: list[dict]) -> list[float]:
    """Each subject's measurement rate, computed row by row from the documented definition.

    The rows that count are the timestamped ones whose code is neither MEDS life event. The rate is their
    number divided by the time from the earliest to the latest, in days, and a subject with no such span has
    no rate.
    """
    times = defaultdict(list)
    for row in rows:
        if row["time"] is not None and row["code"] not in (meds.birth_code, meds.death_code):
            times[row["subject_id"]].append(row["time"])
    rates = []
    for subject_times in times.values():
        span_days = (max(subject_times) - min(subject_times)) / timedelta(days=1)
        if span_days > 0:
            rates.append(len(subject_times) / span_days)
    return rates


class TestMeasurementsPerDay:
    """``per_subject.measurements_per_day`` must be exactly the distribution its definition describes."""

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets_with_life_events(), max_rows=st.one_of(st.none(), st.integers(1, 60)))
    def test_the_rate_matches_a_row_by_row_reference(self, dataset, max_rows, tmp_path_factory):
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        cfg = ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)
        block = extract_summary_stats(root, cfg, max_rows_per_chunk=max_rows)["per_subject"][
            "measurements_per_day"
        ]

        expected = pl.Series(_reference_measurements_per_day(rows), dtype=pl.Float64)
        # Subjects without a span are left out of `n` rather than given an infinite or zero rate.
        assert block["n"] == expected.len()
        if expected.is_empty():
            assert block == {"n": 0, "suppressed": True}
            return

        assert all(math.isfinite(v) and v > 0 for v in block["quantiles"].values()), block
        # Emitted values carry six significant digits; the absolute floor covers a std that is zero
        # in exact arithmetic but not in floating point.
        tol = {"rel": 1e-5, "abs": 1e-9 * float(expected.abs().max())}
        assert block["mean"] == pytest.approx(expected.mean(), **tol)
        if expected.len() > 1:
            assert block["std"] == pytest.approx(expected.std(), **tol)
        # Quantiles that would touch a subject's extreme value are withheld (see
        # `test_no_quantile_depends_on_the_smallest_or_largest_value`); every one that is emitted
        # must match the reference.
        level = {_qkey(q): q for q in cfg.quantiles}
        for key, value in block["quantiles"].items():
            assert value == pytest.approx(expected.quantile(level[key], "linear"), **tol)

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets_with_life_events(), floor=st.integers(2, 30))
    def test_quantiles_never_describe_fewer_subjects_than_the_floor(self, dataset, floor, tmp_path_factory):
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        block = extract_summary_stats(root, ExtractConfig(min_numeric_cell_size=floor))["per_subject"][
            "measurements_per_day"
        ]
        if block["n"] < floor:
            assert block == {"n": block["n"], "suppressed": True}
        else:
            assert set(block) == {"n", "mean", "std", "quantiles"}


class TestChunkedAggregation:
    """Aggregating in bounded chunks is how extraction keeps memory flat; it must change nothing.

    Distinct-subject counts only add across chunks that share no subject, so the planner has to keep every
    subject whole -- including on a non-conformant dataset that scatters a subject across shards, which is
    exactly the kind of broken output this tool exists to read.
    """

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets(), max_rows=st.integers(1, 60), scatter=st.booleans())
    def test_chunking_never_changes_an_aggregate(self, dataset, max_rows, scatter, tmp_path_factory):
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards, scatter=scatter)
        shards = _shard_paths(root)
        available = {"subject_id", "time", "code", "numeric_value"}
        cfg = ExtractConfig(numeric_detail="quantiles")
        vocab_map = _code_vocabulary_map(root)

        whole = _aggregate(_plan_chunks(shards, available, None), cfg, vocab_map)
        chunks = _plan_chunks(shards, available, max_rows)
        chunked = _aggregate(chunks, cfg, vocab_map)

        assert sorted(whole) == sorted(chunked)
        for name, expected in whole.items():
            keys = [k for k in ("subject_id", "code", "year", "vocabulary") if k in expected.columns]
            actual = chunked[name]
            if keys:
                expected, actual = expected.sort(keys), actual.sort(keys)
            # Moments pooled across chunks agree to rounding, not bit-for-bit; every count is exact
            # because the tolerance is far below one.
            assert_frame_equal(expected, actual, rel_tol=1e-9, abs_tol=1e-9)

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets(), max_rows=st.integers(1, 60), scatter=st.booleans())
    def test_a_chunk_overshoots_its_budget_by_less_than_one_subject(
        self, dataset, max_rows, scatter, tmp_path_factory
    ):
        """The budget is a bound, give or take the one subject that cannot be split."""
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards, scatter=scatter)
        available = {"subject_id", "time", "code", "numeric_value"}

        for chunk in _plan_chunks(_shard_paths(root), available, max_rows):
            sizes = chunk.group_by("subject_id").agg(n=pl.len()).collect()["n"]
            if sizes.len():
                assert sizes.sum() < max_rows + sizes.max(), (max_rows, sizes.to_list())


class TestDigests:
    """Digests are the vocabulary's only representation in the output; their algebra must be exact."""

    @given(items=st.lists(st.text(min_size=0, max_size=10).filter(lambda s: "\n" not in s), max_size=20))
    def test_digest_is_deterministic(self, items):
        assert digest(items) == digest(list(items))

    @given(
        items=st.lists(
            st.text(min_size=1, max_size=6).filter(lambda s: "\n" not in s), max_size=8, unique=True
        )
    )
    def test_digest_is_order_sensitive(self, items):
        """Reversing a distinct sequence must change the digest, or frequency ordering means nothing."""
        assume(len(items) > 1)
        assert digest(items) != digest(list(reversed(items)))

    @given(
        items=st.lists(
            st.text(min_size=1, max_size=6).filter(lambda s: "\n" not in s), max_size=10, unique=True
        )
    )
    def test_sorted_digest_is_permutation_invariant(self, items):
        """The alphabetical digest must depend on the code *set* and nothing else."""
        shuffled = sorted(items, key=lambda s: (len(s), s))
        assert digest(sorted(items)) == digest(sorted(shuffled))

    @given(
        pairs=st.lists(
            st.tuples(st.text(string.ascii_uppercase, min_size=1, max_size=4), st.integers(0, 1000)),
            max_size=12,
            unique_by=lambda p: p[0],
        )
    )
    def test_ordered_by_is_a_total_deterministic_order(self, pairs):
        """The ordering must be a permutation of the input and independent of input order."""
        names = [p[0] for p in pairs]
        weights = [p[1] for p in pairs]
        ordered = ordered_by(names, weights)

        assert sorted(ordered) == sorted(names)

        reordered = sorted(pairs, reverse=True)
        assert ordered == ordered_by([p[0] for p in reordered], [p[1] for p in reordered])

        weight_of = dict(pairs)
        for earlier, later in itertools.pairwise(ordered):
            assert (-weight_of[earlier], earlier) < (-weight_of[later], later)

    @given(
        value=st.floats(-1e12, 1e12, allow_nan=False, allow_infinity=False),
        digits=st.integers(1, 12),
    )
    def test_format_moment_is_idempotent(self, value, digits):
        """Rounding an already-rounded value must not move it, or digests would not be stable."""
        once = format_moment(value, digits)
        assert format_moment(float(once), digits) == once

    @given(value=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), digits=st.integers(3, 10))
    def test_format_moment_preserves_magnitude(self, value, digits):
        """The rounded value must stay within its own precision of the original."""
        rounded = float(format_moment(value, digits))
        assert abs(rounded - value) <= abs(value) * 10.0 ** (1 - digits) + 1e-30


class TestComparisonAlgebra:
    """Properties the tolerance policy depends on."""

    @given(
        baseline=st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False),
        current=st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False),
    )
    def test_relative_delta_is_zero_exactly_when_unchanged(self, baseline, current):
        assert (relative_delta(baseline, current) == 0.0) == (baseline == current)

    @given(
        baseline=st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False),
        current=st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False),
    )
    def test_relative_delta_is_antisymmetric_in_sign(self, baseline, current):
        """Swapping the arguments must flip the sign, so 'grew' and 'shrank' are never confused."""
        forward = relative_delta(baseline, current)
        backward = relative_delta(current, baseline)
        assert (forward > 0) == (backward < 0)

    @given(
        baseline=st.floats(1.0, 1e9, allow_nan=False, allow_infinity=False),
        factor=st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
    )
    def test_relative_delta_tracks_the_multiplicative_change(self, baseline, factor):
        """For counts above 1, the relative delta must equal ``factor - 1`` to floating precision."""
        assert relative_delta(baseline, baseline * factor) == pytest.approx(factor - 1, abs=1e-9)


class TestPathAlgebra:
    """Flattening and glob matching decide which rule governs which number."""

    @given(
        document=st.dictionaries(
            st.text(string.ascii_lowercase + "_", min_size=1, max_size=6),
            st.one_of(st.integers(), st.text(max_size=5), st.none(), st.booleans()),
            min_size=1,
            max_size=8,
        )
    )
    def test_flatten_preserves_every_leaf(self, document):
        flat = flatten(document)
        assert len(flat) == len(document)
        assert set(flat.values()) == set(document.values())

    def test_flattening_an_empty_document_yields_one_empty_leaf(self):
        """An empty document is itself the leaf.

        Documented here because it reads as a surprise.
        """
        assert flatten({}) == {"": {}}

    @given(
        keys=st.lists(st.text(string.ascii_lowercase + ".", min_size=1, max_size=6), max_size=8, unique=True)
    )
    def test_escaping_keeps_distinct_keys_distinct(self, keys):
        """Escaping must be injective, or two different split names could collide on one path."""
        assert len({escape(k) for k in keys}) == len(keys)

    @given(segments=st.lists(st.text(string.ascii_lowercase, min_size=1, max_size=5), min_size=1, max_size=4))
    def test_double_star_matches_everything(self, segments):
        """The catch-all must never miss, or a path would go unchecked."""
        assert matches(".".join(segments), "**")

    @given(segments=st.lists(st.text(string.ascii_lowercase, min_size=1, max_size=5), min_size=1, max_size=4))
    def test_a_literal_path_outranks_its_wildcards(self, segments):
        """A hand-written exact rule must always beat a broader one that also matches."""
        path = ".".join(segments)
        wildcarded = ".".join(["*"] * len(segments))
        assert matches(path, path)
        assert matches(path, wildcarded)
        assert specificity(path) > specificity(wildcarded)


class TestOntologyParsers:
    """The scalar and vectorized parsers of ``parent_codes`` are separate code paths for one rule."""

    @settings(max_examples=200)
    @given(parents=st.lists(parent_codes, min_size=1, max_size=8))
    def test_scalar_and_vectorized_parsers_agree(self, parents):
        frame = pl.DataFrame({"parent_code": parents})
        vectorized = frame.select(vocabulary_expr(), key_expr())
        expected = [split_parent_code(p) for p in parents]

        assert vectorized["vocabulary"].to_list() == [v for v, _ in expected]
        assert vectorized["key"].to_list() == [k for _, k in expected]

    @given(parent=parent_codes)
    def test_the_split_is_lossless_up_to_separators(self, parent):
        """Vocabulary and key together must account for the whole input, minus the separator run."""
        vocabulary, key = split_parent_code(parent)
        if vocabulary == "<no-vocabulary>":
            assert parent.lstrip("/") == key
        else:
            assert parent.startswith(vocabulary + "/")
            assert parent[len(vocabulary) :].lstrip("/") == key
