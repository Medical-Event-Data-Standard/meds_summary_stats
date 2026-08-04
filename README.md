# meds_summary_stats

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/meds_summary_stats)](https://pypi.org/project/meds_summary_stats/)
[![Tests](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/tests.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/Medical-Event-Data-Standard/meds_summary_stats.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Medical-Event-Data-Standard/meds_summary_stats#license)

Fast, privacy-safe summary statistics over [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
datasets, for validating ETLs and catching regressions in them over time.

You ship a new version of an ETL. The dataset version has not changed, so the subject count should
not have moved, the schema should be identical, and the code vocabulary should be the same set of
strings. This tool turns that expectation into a file you can commit and a check you can run in CI.

> [!WARNING]
> Under active initial development. The stats surface and comparison semantics are not yet stable;
> pin a version.

## Two commands

```bash
# Reduce a dataset to a small, aggregated, non-sensitive fingerprint.
extract_MEDS_summary_stats /data/mimic-iv-meds -o baseline.json

# ... later, after an ETL change ...
extract_MEDS_summary_stats /data/mimic-iv-meds -o current.json

# Diff the two against a tolerance policy. Exit code 1 if it regressed.
compare_MEDS_summary_stats baseline.json current.json
```

Both are also `python -m meds_summary_stats extract` / `... compare`, and both are importable
(`extract_summary_stats`, `compare_summary_stats`).

```python
>>> from meds_summary_stats.extract import extract_summary_stats
>>> from meds_summary_stats.compare import compare_summary_stats
>>> baseline = extract_summary_stats(simple_static_MEDS)
>>> current = extract_summary_stats(simple_static_MEDS)
>>> compare_summary_stats(baseline, current).status
'pass'

```

## What it emits

A single canonical JSON document. Roughly:

| Block           | Contents                                                                                                                   |
| --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `counts`        | subjects, measurements, events, unique codes, static rows, births, deaths, global numeric moments, data-quality invariants |
| `codes`         | vocabulary digests, occurrence distributions, concentration curves, per-namespace rollups                                  |
| `code_metadata` | metadata coverage, and per-external-vocabulary ontology linkage from `parent_codes`                                        |
| `per_subject`   | distributions of measurements, events, distinct codes, and record span per subject                                         |
| `time`          | calendar-year histogram, optional time quantiles                                                                           |
| `splits`        | per-split subject and measurement counts, plus data/split membership mismatches                                            |
| `layout`        | shard count, column names and dtypes, per-shard presence, dtype conflicts                                                  |
| `dataset`       | verbatim `metadata/dataset.json`                                                                                           |

[**`docs/stats-surface.md`**](docs/stats-surface.md) documents every field, with live output.

### The privacy contract

The output is designed to be committable to a public repository as a baseline. Three rules, enforced
structurally rather than by convention:

1. **No code string is ever emitted** — at any threshold, under any flag. The vocabulary appears
    only as digests (`sha256` over the sorted code list, over the frequency-ordered list, and over
    frequency-restricted subsets) and as aggregate distributions. That is enough to detect that the
    vocabulary changed, and never enough to reconstruct it. Codes without a `//` namespace are
    pooled under a sentinel rather than bucketed by themselves, so a flat vocabulary does not leak
    through the namespace view either.
2. **No small cells.** Any namespace, split, calendar-year, or ontology entry backed by fewer than
    `min_cell_size` distinct subjects (default 20) is dropped entirely — the key is never written,
    only a `<suppressed>` rollup of how many were dropped and their pooled volume.
3. **No extremes.** No `min`, no `max`, anywhere. Not on values, not on timestamps. The subject with
    the longest record is exactly where re-identification starts; quantiles replace extremes, and
    are themselves suppressed below `min_numeric_cell_size` observations.

The one exception is deliberate: `parent_codes` entries name concepts in published external
terminologies (`LOINC/8867-4`), which are properties of the terminology rather than of any patient.
Those vocabularies are reported by name, with their statistics still subject to suppression.

## How comparison works

Both documents flatten to `path -> value`. Each path resolves to exactly one rule — the most
specific glob that matches it — and the rule says how to compare:

| kind       | behavior                                                          |
| ---------- | ----------------------------------------------------------------- |
| `exact`    | any inequality is a finding (schema, dtypes, digests)             |
| `relative` | flag when `abs(new - old) / max(abs(old), 1)` exceeds a threshold |
| `absolute` | flag when the raw delta exceeds a threshold                       |
| `ignore`   | never compared (timestamps, tool version)                         |

Each rule has a `warn` and an `error` tier. The built-in policy is already a reasonable ETL
regression policy; a config file only states your overrides:

```yaml
on_dataset_version_change: relax
fail_on: error
rules:
  - {path: counts.n_subjects, kind: relative, warn: 0.0, error: 0.002}
  - {path: codes.digests.alphabetical, kind: exact, on_change: error}
  - {path: time.by_year.*.n_measurements, kind: relative, error: 0.05}
```

See [`examples/policy.yaml`](examples/policy.yaml) for a fully commented one.

### `on_dataset_version_change: relax`

The setting that makes the motivating workflow practical. The premise of the check is "same data,
new ETL". When the data itself moved — `dataset.dataset_version` differs — volume drift is expected
and flagging it is noise, but a schema or vocabulary regression is still a bug. `relax` draws that
line: everything numeric drops to informational, `exact` rules stay hard.

```python
>>> from copy import deepcopy
>>> from meds_summary_stats.policy import ComparePolicy
>>> baseline = {"stats_schema_version": 1, "config": {},
...             "dataset": {"dataset_version": "1.0"},
...             "layout": {"data_columns": {"code": "String"}},
...             "counts": {"n_subjects": 100_000}}
>>> refreshed = deepcopy(baseline)
>>> refreshed["dataset"]["dataset_version"] = "2.0"
>>> refreshed["counts"]["n_subjects"] = 140_000

Strictly, a 40% jump in subjects is a failure:

>>> compare_summary_stats(baseline, refreshed).status
'fail'

Under `relax`, it is expected -- the data changed:

>>> policy = ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
>>> compare_summary_stats(baseline, refreshed, policy).status
'pass'

But a schema change is never excused:

>>> broken = deepcopy(refreshed)
>>> broken["layout"]["data_columns"]["code"] = "Int64"
>>> report = compare_summary_stats(baseline, broken, policy)
>>> report.status, [f.path for f in report.findings if str(f.severity) == "error"]
('fail', ['layout.data_columns.code'])

```

### Reading a report

```python
>>> from meds_summary_stats.report import render_text
>>> regressed = deepcopy(baseline)
>>> regressed["counts"]["n_subjects"] = 98_000
>>> print(render_text(compare_summary_stats(baseline, regressed)))
FAIL - 3 paths checked, 0 ignored; 1 error.
<BLANKLINE>
ERROR (1)
  counts.n_subjects
      100000 -> 98000 (-2.00%), over relative threshold 0.005

```

`compare_MEDS_summary_stats` also writes a machine-readable report (`--output`) and a
GitHub-flavored markdown one (`--markdown`), and appends the markdown to `$GITHUB_STEP_SUMMARY`
when it is running inside Actions.

## In CI

A composite GitHub Action, usable from any repository:

```yaml
  - name: Check for summary-stat regressions
    uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    with:
      meds-dir: build/meds
      baseline: baselines/mimic-iv-3.1.json
      policy: .github/meds-stats-policy.yaml
      package-spec: meds_summary_stats==0.1.0
```

[**`docs/github-action.md`**](docs/github-action.md) covers the reusable-workflow form, advisory
rollout with `fail-on: never`, reacting to the outcome, label-gated baseline refresh, and the full
input/output/exit-code tables.

## Performance

One `scan_parquet` per shard, normalized and concatenated, then a handful of aggregates collected
together on the polars streaming engine so the common scan is shared. The dataset is read a small
fixed number of times regardless of its size, and the only frames materialized are per-subject and
per-code — both orders of magnitude smaller than the data.

The default `numeric-detail: moments` needs no sorting. `quantiles` adds full-column quantiles over
`numeric_value` and `time`, which does require materializing those columns; it is opt-in for that
reason. Per-subject and per-code quantiles are always computed, since those frames are small.

## Installation

```bash
uv add meds_summary_stats # or: uvx --from meds_summary_stats extract_MEDS_summary_stats --help
```

Requires Python 3.12+.

## Contributing

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md). Briefly: `uv sync --group dev`, then `uv run pytest -v`.
Tests are predominantly doctests — the API-level behavior of essentially every function is pinned by
examples in its docstring, with `tests/` reserved for end-to-end runs and
[hypothesis](https://hypothesis.readthedocs.io/) properties (notably, that no code string ever
escapes into the output for *any* dataset).

## License

MIT
