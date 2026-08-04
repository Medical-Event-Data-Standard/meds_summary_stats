"""End-to-end tests over real MEDS datasets on disk.

The API-level behavior of every function is covered by doctests. What lives here is the work a
doctest would obscure rather than illuminate: running the installed console scripts as subprocesses,
mutating a dataset on disk and checking that the intended regression -- and only that regression --
is caught, and asserting the privacy invariants over a dataset large enough for suppression to
actually bite.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from meds_summary_stats.cli import EXIT_INPUT_ERROR, EXIT_REGRESSION, compare_main, extract_main
from meds_summary_stats.compare import compare_summary_stats
from meds_summary_stats.config import SUPPRESSED, ExtractConfig
from meds_summary_stats.extract import extract_summary_stats, read_summary_stats, write_summary_stats
from meds_summary_stats.policy import ComparePolicy


@pytest.fixture
def dataset_copy(generated_sample_MEDS: Path, tmp_path: Path) -> Path:
    """A private, mutable copy of the generated sample dataset."""
    target = tmp_path / "dataset"
    shutil.copytree(generated_sample_MEDS, target)
    return target


def _mutate_shard(meds_dir: Path, fn) -> None:
    """Rewrite the first data shard through ``fn``."""
    shard = sorted((meds_dir / "data").rglob("*.parquet"))[0]
    fn(pl.read_parquet(shard)).write_parquet(shard)


def test_round_trip_is_stable(generated_sample_MEDS: Path, tmp_path: Path):
    """Extracting the same dataset twice must produce byte-identical files apart from the timestamp.

    This is the property the whole tool rests on: if repeated extraction of unchanged data were not
    reproducible, every comparison would be noise.
    """
    first = extract_summary_stats(generated_sample_MEDS)
    second = extract_summary_stats(generated_sample_MEDS)

    first.pop("extracted_at")
    second.pop("extracted_at")
    assert first == second

    a = write_summary_stats(first, tmp_path / "a.json")
    b = write_summary_stats(second, tmp_path / "b.json")
    assert a.read_text() == b.read_text()

    report = compare_summary_stats(read_summary_stats(a), read_summary_stats(b))
    assert report.status == "pass", report.findings
    assert report.n_paths_checked > 50


def test_shard_layout_does_not_change_the_statistics(generated_sample_MEDS: Path, tmp_path: Path):
    """Re-sharding a dataset must not move any statistic except the layout block.

    An ETL that changes its parallelism changes its shard count. That is not a data regression, and conflating
    the two would make the check useless on any pipeline whose worker count varies.
    """
    baseline = extract_summary_stats(generated_sample_MEDS)

    resharded = tmp_path / "resharded"
    (resharded / "data").mkdir(parents=True)
    shutil.copytree(generated_sample_MEDS / "metadata", resharded / "metadata")
    combined = pl.read_parquet(sorted((generated_sample_MEDS / "data").rglob("*.parquet")))
    combined.sort("subject_id", "time", nulls_last=False).write_parquet(resharded / "data" / "all.parquet")

    current = extract_summary_stats(resharded)
    report = compare_summary_stats(baseline, current)

    changed = {f.path for f in report.findings}
    assert changed, "expected the layout block itself to differ"
    assert all(p.startswith("layout.") for p in changed), sorted(changed)


@pytest.mark.parametrize(
    ("description", "mutate", "expected_path"),
    [
        (
            "dropping subjects",
            lambda df: df.filter(pl.col("subject_id") % 5 != 0),
            "counts.n_subjects",
        ),
        (
            "rescaling numeric values",
            lambda df: df.with_columns(numeric_value=pl.col("numeric_value") * 10),
            "counts.numeric.mean",
        ),
        (
            "nulling out the time column",
            lambda df: df.with_columns(time=pl.lit(None, pl.Datetime("us"))),
            "counts.n_static_measurements",
        ),
        (
            "narrowing the subject_id dtype",
            lambda df: df.with_columns(subject_id=pl.col("subject_id").cast(pl.Int32)),
            "layout.data_columns.subject_id",
        ),
    ],
)
def test_regressions_are_caught(
    dataset_copy: Path, generated_sample_MEDS: Path, description, mutate, expected_path
):
    """Each mutation is a real ETL failure mode, and each must surface as an error on its own path."""
    baseline = extract_summary_stats(generated_sample_MEDS)
    _mutate_shard(dataset_copy, mutate)
    current = extract_summary_stats(dataset_copy)

    report = compare_summary_stats(baseline, current)
    assert report.failed, f"{description} was not caught"
    error_paths = {f.path for f in report.findings if str(f.severity) == "error"}
    assert expected_path in error_paths, f"{description}: expected {expected_path}, got {sorted(error_paths)}"


def test_renaming_a_code_is_caught_without_naming_the_code(dataset_copy: Path, generated_sample_MEDS: Path):
    """A vocabulary change must be detected via digests, and must not leak the code in the report.

    This is the central tension of the design: the tool has to be sensitive to vocabulary drift while
    refusing to publish the vocabulary. The digest gives detection; the report must stay clean.
    """
    baseline = extract_summary_stats(generated_sample_MEDS)
    victim = pl.read_parquet(sorted((generated_sample_MEDS / "data").rglob("*.parquet"))[0])["code"][0]
    _mutate_shard(
        dataset_copy,
        lambda df: df.with_columns(
            code=pl.when(pl.col("code") == victim).then(pl.lit("RENAMED//CODE")).otherwise(pl.col("code"))
        ),
    )
    current = extract_summary_stats(dataset_copy)
    report = compare_summary_stats(baseline, current)

    assert report.failed
    assert "codes.digests.alphabetical" in {f.path for f in report.findings}

    blob = json.dumps(report.to_dict())
    assert victim not in blob
    assert "RENAMED//CODE" not in blob


def test_relax_distinguishes_data_change_from_etl_regression(dataset_copy: Path, generated_sample_MEDS: Path):
    """Under ``relax``, a version bump excuses volume drift but not a schema change."""
    baseline = extract_summary_stats(generated_sample_MEDS)
    baseline["dataset"]["dataset_version"] = "1.0"

    _mutate_shard(dataset_copy, lambda df: df.filter(pl.col("subject_id") % 3 != 0))
    current = extract_summary_stats(dataset_copy)
    current["dataset"]["dataset_version"] = "2.0"

    policy = ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
    relaxed = compare_summary_stats(baseline, current, policy)
    assert relaxed.relaxed
    assert not relaxed.failed, [f.path for f in relaxed.findings if str(f.severity) == "error"]

    strict = compare_summary_stats(baseline, current)
    assert strict.failed

    current["layout"]["data_columns"]["code"] = "Int64"
    still_caught = compare_summary_stats(baseline, current, policy)
    assert still_caught.failed
    assert {f.path for f in still_caught.findings if str(f.severity) == "error"} == {
        "layout.data_columns.code"
    }


def test_suppression_holds_over_a_realistic_dataset(generated_sample_MEDS: Path):
    """No reported cell may be backed by fewer subjects than the configured floor.

    Checked at a floor high enough to bite (the sample dataset has a few hundred subjects), so the assertion
    exercises the suppression path rather than trivially passing.
    """
    cfg = ExtractConfig(min_cell_size=100)
    stats = extract_summary_stats(generated_sample_MEDS, cfg)

    keyed_blocks = [
        stats["codes"]["by_namespace"],
        stats["splits"]["by_split"],
        stats["time"]["by_year"],
        stats["code_metadata"]["ontology"].get("by_vocabulary", {}),
    ]
    for block in keyed_blocks:
        for key, entry in block.items():
            if key == SUPPRESSED:
                assert "n_subjects" not in entry, "rollups must not report a pooled subject count"
                continue
            assert entry["n_subjects"] >= cfg.min_cell_size, (key, entry)

    assert any(SUPPRESSED in block for block in keyed_blocks), "floor of 100 should suppress something"


@pytest.mark.parametrize("dataset_name", ["generated_sample_MEDS", "simple_static_MEDS"])
def test_no_code_string_appears_in_the_output(dataset_name, request):
    """No code string may appear in the output, at any suppression threshold.

    Checked with suppression effectively disabled as well as at the default, because the guarantee is
    structural -- it must not depend on the floor happening to be high enough.
    """
    meds_dir = request.getfixturevalue(dataset_name)
    codes = set(pl.read_parquet(sorted((meds_dir / "data").rglob("*.parquet")))["code"].unique().to_list())

    for cfg in (ExtractConfig(), ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)):
        blob = json.dumps(extract_summary_stats(meds_dir, cfg))
        leaked = {c for c in codes if c in blob}
        assert not leaked, f"code strings leaked at min_cell_size={cfg.min_cell_size}: {sorted(leaked)}"


def test_no_subject_identifier_appears_in_the_output(simple_static_MEDS: Path):
    """No subject id may appear in the output.

    Tested against the static sample rather than the generated one: its subject ids are large,
    distinctive integers (239684, 1195293, ...), so a hit is real. The generated dataset numbers its
    subjects 0..N, which collide with every small count in the document and would make the check
    meaningless.
    """
    data = pl.read_parquet(sorted((simple_static_MEDS / "data").rglob("*.parquet")))
    subject_ids = set(data["subject_id"].unique().to_list())
    assert all(i > 1000 for i in subject_ids), "fixture assumption changed; revisit this test"

    for cfg in (ExtractConfig(), ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)):
        blob = json.dumps(extract_summary_stats(simple_static_MEDS, cfg))
        leaked = {i for i in subject_ids if str(i) in blob}
        assert not leaked, f"subject ids leaked at min_cell_size={cfg.min_cell_size}: {sorted(leaked)}"


def test_quantile_detail_adds_only_quantiles(generated_sample_MEDS: Path):
    """The expensive detail level must be additive, not a different set of numbers."""
    moments = extract_summary_stats(generated_sample_MEDS, ExtractConfig(numeric_detail="moments"))
    quantiles = extract_summary_stats(generated_sample_MEDS, ExtractConfig(numeric_detail="quantiles"))

    assert "quantiles" not in moments["counts"]["numeric"]
    assert "quantiles" in quantiles["counts"]["numeric"]
    assert "quantiles" in quantiles["time"]

    assert moments["counts"]["numeric"]["mean"] == quantiles["counts"]["numeric"]["mean"]
    assert moments["codes"]["digests"] == quantiles["codes"]["digests"]


class TestConsoleScripts:
    """The installed entry points, run as real subprocesses."""

    @staticmethod
    def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            args, capture_output=True, text=True, check=False, env={**os.environ, **(env or {})}
        )

    def test_extract_script_writes_a_readable_document(self, generated_sample_MEDS: Path, tmp_path: Path):
        out = tmp_path / "stats.json"
        result = self._run("extract_MEDS_summary_stats", str(generated_sample_MEDS), "-o", str(out))
        assert result.returncode == 0, result.stderr
        assert read_summary_stats(out)["counts"]["n_subjects"] > 0

    def test_extract_script_streams_to_stdout(self, generated_sample_MEDS: Path):
        result = self._run("extract_MEDS_summary_stats", str(generated_sample_MEDS))
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["stats_schema_version"] == 1

    def test_compare_script_exit_codes(self, generated_sample_MEDS: Path, tmp_path: Path):
        good = tmp_path / "good.json"
        assert extract_main([str(generated_sample_MEDS), "-o", str(good)]) == 0

        result = self._run("compare_MEDS_summary_stats", str(good), str(good))
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("PASS")

        bad = json.loads(good.read_text())
        bad["counts"]["n_subjects"] = int(bad["counts"]["n_subjects"] * 0.5)
        (tmp_path / "bad.json").write_text(json.dumps(bad))

        result = self._run("compare_MEDS_summary_stats", str(good), str(tmp_path / "bad.json"))
        assert result.returncode == EXIT_REGRESSION
        assert "counts.n_subjects" in result.stdout

        result = self._run("compare_MEDS_summary_stats", str(tmp_path / "nope.json"), str(good))
        assert result.returncode == EXIT_INPUT_ERROR

    def test_module_dispatch(self, generated_sample_MEDS: Path, tmp_path: Path):
        out = tmp_path / "stats.json"
        result = self._run(
            sys.executable, "-m", "meds_summary_stats", "extract", str(generated_sample_MEDS), "-o", str(out)
        )
        assert result.returncode == 0, result.stderr
        assert out.is_file()

        result = self._run(sys.executable, "-m", "meds_summary_stats", "compare", str(out), str(out))
        assert result.returncode == 0, result.stderr

        result = self._run(sys.executable, "-m", "meds_summary_stats")
        assert result.returncode == 2

    def test_github_step_summary_is_appended(self, generated_sample_MEDS: Path, tmp_path: Path):
        """The CI integration writes markdown to ``$GITHUB_STEP_SUMMARY`` without clobbering it."""
        stats = tmp_path / "stats.json"
        assert extract_main([str(generated_sample_MEDS), "-o", str(stats)]) == 0

        summary = tmp_path / "summary.md"
        summary.write_text("# Pre-existing content\n")
        result = self._run(
            "compare_MEDS_summary_stats",
            str(stats),
            str(stats),
            env={"GITHUB_STEP_SUMMARY": str(summary)},
        )
        assert result.returncode == 0, result.stderr
        text = summary.read_text()
        assert text.startswith("# Pre-existing content")
        assert "MEDS summary-stats comparison: pass" in text


def test_compare_main_writes_both_report_formats(generated_sample_MEDS: Path, tmp_path: Path):
    """The JSON report and the markdown report must both be produced when requested."""
    stats = tmp_path / "stats.json"
    assert extract_main([str(generated_sample_MEDS), "-o", str(stats)]) == 0

    perturbed = json.loads(stats.read_text())
    perturbed["counts"]["n_measurements"] = int(perturbed["counts"]["n_measurements"] * 1.5)
    (tmp_path / "perturbed.json").write_text(json.dumps(perturbed))

    code = compare_main(
        [
            str(stats),
            str(tmp_path / "perturbed.json"),
            "-o",
            str(tmp_path / "nested" / "report.json"),
            "--markdown",
            str(tmp_path / "nested" / "report.md"),
            "--quiet",
        ]
    )
    assert code == EXIT_REGRESSION

    report = json.loads((tmp_path / "nested" / "report.json").read_text())
    assert report["status"] == "fail"
    assert report["failed"] is True
    assert any(f["path"] == "counts.n_measurements" for f in report["findings"])
    assert "| Path | Baseline | Current | Detail |" in (tmp_path / "nested" / "report.md").read_text()
