# meds_summary_stats

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/meds_summary_stats)](https://pypi.org/project/meds_summary_stats/)
[![Tests](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/tests.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/Medical-Event-Data-Standard/meds_summary_stats.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats#license)

Fast, privacy-safe summary statistics over [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
datasets, for validating MEDS ETLs and detecting regressions in them over time.

> \[!WARNING\]
> This package is under active initial development; the stats surface and comparison semantics are
> not yet stable.

## What it does

1. **Extract** — walk a MEDS dataset once and emit a small, aggregated, unambiguously non-sensitive
    JSON "fingerprint" of it.
2. **Compare** — diff two such fingerprints against a configurable tolerance policy and flag
    regressions, with a machine-readable report and a CI-friendly exit code.

The intended workflow: an ETL emits a fingerprint at release time; a later ETL version re-extracts
and compares. If the underlying dataset version has not changed, the fingerprints should agree
within tolerance.

## License

MIT
