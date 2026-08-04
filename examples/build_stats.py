"""Regenerate the committed example fingerprints under ``examples/stats/``.

These are ordinary ``extract_MEDS_summary_stats`` outputs, committed so that the README can show
real numbers and so ``tests/test_examples.py`` can prove those numbers are still what the tool
produces. They are also exactly what a baseline looks like in practice -- a small, aggregated,
non-sensitive JSON file that belongs in version control next to the ETL that produced it.

Two fields are overwritten with placeholders before writing: ``extracted_at`` and
``meds_summary_stats_version``. Both legitimately change on every run and both are ignored by the
comparison policy, so leaving the real values in would produce a diff on every release for no
reason. Nothing else is touched.

Run with ``uv run python examples/build_stats.py``.
"""

from __future__ import annotations

from pathlib import Path

from meds_summary_stats.config import ExtractConfig
from meds_summary_stats.extract import extract_summary_stats, write_summary_stats

EXAMPLES_DIR = Path(__file__).parent
DATASETS_DIR = EXAMPLES_DIR / "datasets"
STATS_DIR = EXAMPLES_DIR / "stats"

#: Placeholders for the two fields that change on every run. See the module docstring.
PLACEHOLDER_EXTRACTED_AT = "2026-01-01T00:00:00"
PLACEHOLDER_VERSION = "0.0.0-example"

#: The example uses the shipped defaults, so the fingerprints show what a real caller would get.
CONFIG = ExtractConfig()


def stats_for(dataset: str, cfg: ExtractConfig | None = None) -> dict:
    """Extract a fingerprint for one example dataset, with run-varying fields pinned.

    Args:
        dataset: Directory name under ``examples/datasets/``.
        cfg: Extraction config. Defaults to the shipped defaults, which is what the committed
            fingerprints use.

    Returns:
        The stats document, ready to be written or compared.
    """
    stats = extract_summary_stats(DATASETS_DIR / dataset, cfg or CONFIG)
    stats["extracted_at"] = PLACEHOLDER_EXTRACTED_AT
    stats["meds_summary_stats_version"] = PLACEHOLDER_VERSION
    return stats


def main() -> None:
    """Regenerate every example fingerprint in place."""
    STATS_DIR.mkdir(exist_ok=True)
    for dataset in sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir()):
        path = write_summary_stats(stats_for(dataset), STATS_DIR / f"{dataset}.json")
        print(f"{dataset}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
