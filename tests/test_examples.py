"""Tests that keep the worked example in ``examples/`` honest.

The example is committed in three layers -- the datasets, the fingerprints extracted from them, and
the comparison outcomes shown in the README -- and each layer can drift from the one below it. A
README showing a report the tool no longer produces is worse than no README, so all three are
regenerated here and checked against what is on disk.

Regenerate with ``uv run python examples/build_datasets.py && uv run python examples/build_stats.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from build_datasets import BUILDERS  # noqa: E402  (path set immediately above)
from build_stats import PLACEHOLDER_EXTRACTED_AT, PLACEHOLDER_VERSION, stats_for  # noqa: E402

from meds_summary_stats.compare import compare_summary_stats  # noqa: E402
from meds_summary_stats.extract import read_summary_stats  # noqa: E402
from meds_summary_stats.policy import ComparePolicy  # noqa: E402

DATASETS = sorted(BUILDERS)


@pytest.mark.parametrize("name", DATASETS)
def test_committed_datasets_match_the_builder(name: str, tmp_path: Path):
    """Each committed dataset must be exactly what ``build_datasets.py`` produces.

    Content equality, not byte equality: a new parquet writer version would change the bytes without changing
    the data, and failing on that would be noise.
    """
    rebuilt = BUILDERS[name](tmp_path / name)
    committed = EXAMPLES / "datasets" / name

    for relative in ("metadata/codes.parquet", "metadata/subject_splits.parquet"):
        assert pl.read_parquet(rebuilt / relative).equals(pl.read_parquet(committed / relative)), relative
    assert (rebuilt / "metadata/dataset.json").read_text() == (
        committed / "metadata/dataset.json"
    ).read_text()

    rebuilt_shards = sorted(p.relative_to(rebuilt) for p in (rebuilt / "data").rglob("*.parquet"))
    committed_shards = sorted(p.relative_to(committed) for p in (committed / "data").rglob("*.parquet"))
    assert rebuilt_shards == committed_shards
    for shard in rebuilt_shards:
        assert pl.read_parquet(rebuilt / shard).equals(pl.read_parquet(committed / shard)), shard


@pytest.mark.parametrize("name", DATASETS)
def test_committed_fingerprints_match_the_datasets(name: str):
    """Each committed fingerprint must be what extraction currently produces for its dataset."""
    assert stats_for(name) == read_summary_stats(EXAMPLES / "stats" / f"{name}.json")


@pytest.mark.parametrize("name", DATASETS)
def test_committed_fingerprints_use_the_placeholders(name: str):
    """The two run-varying fields must be pinned, or every release would diff the examples."""
    stats = read_summary_stats(EXAMPLES / "stats" / f"{name}.json")
    assert stats["extracted_at"] == PLACEHOLDER_EXTRACTED_AT
    assert stats["meds_summary_stats_version"] == PLACEHOLDER_VERSION


@pytest.fixture(scope="module")
def fingerprints() -> dict[str, dict]:
    """The three committed fingerprints, read once."""
    return {name: read_summary_stats(EXAMPLES / "stats" / f"{name}.json") for name in DATASETS}


def test_baseline_compares_clean_against_itself(fingerprints):
    """The healthy fingerprint must pass against itself, with a non-trivial number of checks."""
    report = compare_summary_stats(fingerprints["v1_baseline"], fingerprints["v1_baseline"])
    assert report.status == "pass"
    assert not report.failed
    assert report.findings == ()
    assert report.n_paths_checked > 150


def test_the_broken_etl_is_caught_bug_by_bug(fingerprints):
    """Each of the three planted bugs must surface, on a path that identifies it."""
    report = compare_summary_stats(fingerprints["v1_baseline"], fingerprints["v1_etl_regression"])
    assert report.status == "fail"
    assert report.failed

    errors = {f.path: f for f in report.findings if str(f.severity) == "error"}

    # Bug 1: subjects dropped on the floor. 160 -> 152 is 5%, ten times the 0.5% error threshold.
    assert "counts.n_subjects" in errors
    assert errors["counts.n_subjects"].baseline == 160
    assert errors["counts.n_subjects"].current == 152

    # Bug 2: a code renamed. The set-identity digest is the check that catches it, and the report
    # says so without ever naming either the old or the new code.
    assert "codes.digests.alphabetical" in errors
    assert "SODIUM" not in str(report.to_dict())
    assert "LAB//NA" not in str(report.to_dict())

    # Bug 3: TEMP emitted in Celsius. Per-vocabulary moments localize it to the LOINC-mapped codes.
    assert "code_metadata.ontology.by_vocabulary.LOINC.numeric.mean" in errors
    assert errors["code_metadata.ontology.by_vocabulary.LOINC.numeric.mean"].relative_delta < -0.1


def test_a_data_refresh_is_not_an_etl_regression(fingerprints):
    """The refreshed dataset must fail strictly and pass under ``relax``."""
    baseline, refreshed = fingerprints["v1_baseline"], fingerprints["v2_data_refresh"]

    strict = compare_summary_stats(baseline, refreshed)
    assert strict.failed, "a 37% jump in subjects should fail when the version is held fixed"

    relaxed = compare_summary_stats(
        baseline, refreshed, ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
    )
    assert relaxed.relaxed
    assert not relaxed.failed, [f.path for f in relaxed.findings if str(f.severity) == "error"]
    assert relaxed.status == "warn"

    # The vocabulary is unchanged, so set identity must hold even though volume moved a lot.
    assert baseline["codes"]["digests"]["alphabetical"] == refreshed["codes"]["digests"]["alphabetical"]


def test_the_refresh_and_the_regression_are_distinguishable(fingerprints):
    """The two failure modes must not look alike, or the check teaches nothing.

    Both move volume by roughly the same magnitude. What separates them is that the broken ETL also changed
    the vocabulary and shifted a distribution, while the refresh did neither.
    """
    regression = compare_summary_stats(fingerprints["v1_baseline"], fingerprints["v1_etl_regression"])
    refresh = compare_summary_stats(fingerprints["v1_baseline"], fingerprints["v2_data_refresh"])

    regression_paths = {f.path for f in regression.findings}
    refresh_paths = {f.path for f in refresh.findings}

    assert "codes.digests.alphabetical" in regression_paths
    assert "codes.digests.alphabetical" not in refresh_paths

    assert "code_metadata.ontology.by_vocabulary.LOINC.numeric.mean" in regression_paths
    assert "code_metadata.ontology.by_vocabulary.LOINC.numeric.mean" not in refresh_paths
