"""End-to-end runs of ``baseline-freshness/check.sh``, the script behind the baseline-freshness action.

Each test builds a two-commit git repository (a base and a pull request on top of it) and runs the
script in it with the environment the composite action passes, then reads what it printed and what it
wrote to ``$GITHUB_OUTPUT`` and ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / "baseline-freshness" / "check.sh"
BASELINE = "tests/regression/full_summary_stats.json"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path, changes: dict[str, str]) -> Path:
    """A repository whose HEAD changes ``changes`` (path -> new content) on top of a base commit."""
    repo = tmp_path / "repo"
    for rel in ("src/etl.yaml", "README.md", BASELINE):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("base\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    for rel, content in changes.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "pr")
    return repo


def _check(repo: Path, tmp_path: Path, **env: str) -> tuple[str, str, str]:
    output, summary = tmp_path / "output", tmp_path / "summary"
    result = subprocess.run(
        ["bash", str(CHECK)],
        cwd=repo,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
            "REGENERATE_COMMAND": "baseline_MEDS_summary_stats full --input-dir $RAW",
            **env,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout, output.read_text(), summary.read_text() if summary.exists() else ""


def test_an_etl_change_without_a_new_baseline_warns(tmp_path: Path):
    repo = _repo(tmp_path, {"src/etl.yaml": "changed\n"})
    stdout, output, summary = _check(repo, tmp_path)
    assert "::warning title=Summary-stats baseline not regenerated::" in stdout
    assert f"This pull request changes the ETL but not {BASELINE}." in stdout
    assert output == f"stale={BASELINE}\n"
    assert "- `src/etl.yaml`" in summary
    assert f"- `{BASELINE}`" in summary
    assert "baseline_MEDS_summary_stats full --input-dir $RAW" in summary


def test_an_etl_change_with_a_new_baseline_passes(tmp_path: Path):
    repo = _repo(tmp_path, {"src/etl.yaml": "changed\n", BASELINE: "regenerated\n"})
    stdout, output, summary = _check(repo, tmp_path)
    assert f"The ETL changed and {BASELINE} was regenerated." in stdout
    assert output == "stale=\n"
    assert summary == ""


@pytest.mark.parametrize("changes", [{"README.md": "docs\n"}, {"src/notes.md": "docs\n"}])
def test_documentation_is_not_an_etl_change(tmp_path: Path, changes: dict[str, str]):
    repo = _repo(tmp_path, changes)
    stdout, output, summary = _check(repo, tmp_path)
    assert stdout == "No ETL changes.\n"
    assert output == "stale=\n"
    assert summary == ""


def test_several_baselines_and_custom_paths(tmp_path: Path):
    demo = "tests/regression/demo_summary_stats.json"
    repo = _repo(tmp_path, {"configs/spec.yaml": "changed\n", BASELINE: "regenerated\n"})
    stdout, output, _ = _check(repo, tmp_path, BASELINES=f"{BASELINE}\n{demo}\n", WATCH="^(src|configs)/")
    assert output == f"stale={demo}\n"
    assert f"changes the ETL but not {demo}" in stdout
    assert f"changes the ETL but not {BASELINE}" not in stdout
