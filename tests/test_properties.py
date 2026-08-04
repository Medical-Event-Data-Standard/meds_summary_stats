"""Property-based tests.

Doctests pin behavior on chosen examples; these pin the invariants that must hold on *every* input.
Three families:

1. **Privacy invariants.** Over arbitrary generated MEDS datasets, no code string may appear in the
    output and no reported cell may fall below the suppression floor. These are the claims the
    package makes, and an example-based test can only ever fail to falsify them.
2. **Digest and comparison algebra.** Determinism, order sensitivity, tie-breaking, and the
    threshold semantics that the tolerance policy relies on.
3. **Cross-implementation agreement.** The scalar and vectorized parsers of ``parent_codes`` must
    agree everywhere, since they are separate code paths for the same rule.
"""

from __future__ import annotations

import itertools
import json
import string
from datetime import datetime

import hypothesis.strategies as st
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings

from meds_summary_stats.compare import compare_summary_stats, relative_delta
from meds_summary_stats.config import SUPPRESSED, ExtractConfig
from meds_summary_stats.digests import digest, format_moment, ordered_by
from meds_summary_stats.extract import extract_summary_stats
from meds_summary_stats.flatten import escape, flatten, matches, specificity
from meds_summary_stats.ontology import key_expr, split_parent_code, vocabulary_expr

# Codes are drawn to look like real MEDS vocabularies: some namespaced, some flat, with the
# separator characters that the namespace and vocabulary parsers key on.
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


def _write(tmp_path, rows, code_rows, n_shards):
    """Write generated rows out as a sharded MEDS dataset, keeping subjects whole per shard."""
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
        assigned = subjects[shard::n_shards]
        frame.filter(pl.col("subject_id").is_in(assigned)).write_parquet(
            tmp_path / "data" / f"{shard}.parquet"
        )

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
    def test_no_code_ever_appears_in_the_output(self, dataset, tmp_path_factory):
        """No code string may appear in the emitted document, at any suppression threshold.

        Run with the floor at 1 -- suppression effectively disabled -- because the guarantee must be
        structural. If it only held because the floor happened to be high, it would not be a
        guarantee at all.
        """
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)

        cfg = ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)
        blob = json.dumps(extract_summary_stats(root, cfg))
        vocabulary = {r["code"] for r in code_rows}
        leaked = {c for c in vocabulary if len(c) > 2 and c in blob}
        assert not leaked, f"leaked codes: {sorted(leaked)}"

    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets(min_subjects=5), floor=st.integers(2, 30))
    def test_reported_cells_always_clear_the_floor(self, dataset, floor, tmp_path_factory):
        """Every keyed cell that survives must be backed by at least ``min_cell_size`` subjects."""
        rows, code_rows, n_shards = dataset
        root = _write(tmp_path_factory.mktemp("meds"), rows, code_rows, n_shards)
        stats = extract_summary_stats(root, ExtractConfig(min_cell_size=floor))

        blocks = [
            stats["codes"]["by_namespace"],
            stats["time"]["by_year"],
            stats["code_metadata"]["ontology"].get("by_vocabulary", {}),
        ]
        for block in blocks:
            for key, entry in block.items():
                if key == SUPPRESSED:
                    assert "n_subjects" not in entry
                    continue
                assert entry["n_subjects"] >= floor, (key, entry, floor)

    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(dataset=meds_datasets())
    def test_a_document_always_compares_equal_to_itself(self, dataset, tmp_path_factory):
        """Self-comparison must pass for every dataset. A false positive here is a false alarm in CI."""
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
        """An empty document is itself the leaf. Documented here because it reads as a surprise."""
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
