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

## A worked example

Three complete MEDS datasets are committed under [`examples/datasets/`](examples/datasets), so
everything below runs against a fresh clone with no setup. Every output on this page is executed as
a doctest, so it is what the tool actually prints.

```python
>>> print_directory(EXAMPLES / "datasets" / "v1_baseline")
├── data
│   ├── held_out
│   │   └── 0.parquet
│   ├── train
│   │   └── 0.parquet
│   └── tuning
│       └── 0.parquet
└── metadata
    ├── codes.parquet
    ├── dataset.json
    └── subject_splits.parquet

```

160 subjects, 2258 measurements, a ten-code vocabulary — half of it mapped into LOINC, ICD10CM, and
SNOMED via `parent_codes`, half unmapped. [`examples/build_datasets.py`](examples/build_datasets.py)
is the readable source of truth for what is in them.

### 1. Extract a baseline

```bash
extract_MEDS_summary_stats examples/datasets/v1_baseline -o examples/stats/v1_baseline.json
```

That fingerprint is committed at [`examples/stats/v1_baseline.json`](examples/stats/v1_baseline.json)
— 8 KB of JSON, safe to keep in version control next to the ETL that produced it. The headline
block:

```python
>>> from meds_summary_stats.extract import read_summary_stats
>>> baseline = read_summary_stats(EXAMPLES / "stats" / "v1_baseline.json")
>>> print(json.dumps(baseline["counts"], indent=2, sort_keys=True))
{
  "n_events": 2258,
  "n_measurements": 2258,
  "n_static_measurements": 160,
  "n_subjects": 160,
  "n_subjects_with_birth": 160,
  "n_subjects_with_death": 0,
  "n_subjects_with_static": 160,
  "n_unique_codes": 10,
  "numeric": {
    "mean": 103.9,
    "n_finite": 1037,
    "n_infinite": 0,
    "n_nan": 0,
    "n_negative": 0,
    "n_present": 1037,
    "n_zero": 0,
    "std": 26.3615
  },
  "quality": {
    "n_empty_code": 0,
    "n_null_code": 0,
    "n_null_subject_id": 0
  }
}

```

Splits survive the default suppression floor of 20 subjects, so they are reported by name:

```python
>>> print(json.dumps(baseline["splits"]["by_split"], indent=2, sort_keys=True))
{
  "held_out": {
    "n_measurements": 315,
    "n_subjects": 24
  },
  "train": {
    "n_measurements": 1593,
    "n_subjects": 112
  },
  "tuning": {
    "n_measurements": 350,
    "n_subjects": 24
  }
}

```

Alongside the statistics, the fingerprint carries a checksum of the dataset itself — a sha256 over
the bytes of every file the MEDS schema defines. That answers the prior question, "is this even the
same dataset?", without reference to any statistic:

```python
>>> baseline["dataset_digest"]["combined"]
'sha256:...'
>>> baseline["dataset_digest"]["n_files"], baseline["dataset_digest"]["data"]["n_files"]
(6, 3)

```

Files outside the schema are ignored, so task labels and caches in the dataset root cannot affect
it, and `metadata/dataset.json` is digested separately because its `created_at` changes on every
run. It is a digest of bytes, so it is also sensitive to re-encoding — see
[`docs/stats-surface.md`](docs/stats-surface.md#dataset_digest--is-this-the-same-dataset-at-all)
for what it is and is not invariant to, and for how to promote it from a warning to a hard error
once you know your ETL is byte-reproducible.

Re-extracting the same dataset reproduces it exactly, which is the property the whole tool rests on:

```python
>>> from meds_summary_stats.compare import compare_summary_stats
>>> from meds_summary_stats.report import render_text
>>> print(render_text(compare_summary_stats(baseline, baseline)))
PASS - 197 paths checked, 4 ignored, no findings.

```

### 2. A broken ETL, same source data

[`examples/datasets/v1_etl_regression`](examples/datasets/v1_etl_regression) is the *same* source
data — `dataset_version` still `1.0` — put through an ETL carrying three independent bugs: `TEMP` is
emitted in Celsius instead of Fahrenheit, `LAB//SODIUM` is silently renamed, and every twentieth
subject is dropped.

```bash
extract_MEDS_summary_stats examples/datasets/v1_etl_regression -o current.json
compare_MEDS_summary_stats examples/stats/v1_baseline.json current.json # exit 1
```

```python
>>> regressed = read_summary_stats(EXAMPLES / "stats" / "v1_etl_regression.json")
>>> report = compare_summary_stats(baseline, regressed)
>>> report.status, report.failed, report.counts
('fail', True, {'error': 44, 'info': 0, 'warning': 22})

```

All three bugs are caught, each on a path that names the problem:

```python
>>> findings = {f.path: f for f in report.findings}

>>> print(findings["counts.n_subjects"].message)          # dropped subjects
160 -> 152 (-5.00%), over relative threshold 0.005

>>> findings["codes.digests.alphabetical"].kind           # renamed code
'exact'

>>> loinc = findings["code_metadata.ontology.by_vocabulary.LOINC.numeric.mean"]
>>> loinc.baseline, loinc.current                         # Fahrenheit -> Celsius
(103.9, 88.1595)

```

Note what the second one does *not* say. The vocabulary change is detected by a digest, so neither
the old code nor the new one appears anywhere in the report:

```python
>>> blob = json.dumps(report.to_dict())
>>> "SODIUM" in blob or "LAB//NA" in blob
False

```

And note what the third one *does*. The per-code moments digest says only "some code's distribution
moved"; the per-vocabulary moments narrow it to the LOINC-mapped measurements. That is as specific
as a report can get without naming a code.

### 3. A genuine data refresh

[`examples/datasets/v2_data_refresh`](examples/datasets/v2_data_refresh) bumps `dataset_version` to
`2.0` and has 220 subjects instead of 160 — same schema, same vocabulary, more data. Strictly, that
fails:

```python
>>> refreshed = read_summary_stats(EXAMPLES / "stats" / "v2_data_refresh.json")
>>> compare_summary_stats(baseline, refreshed).failed
True

```

Which is the wrong answer, and why `relax` exists. It tells the comparison that a version bump
excuses volume drift but nothing else:

```bash
compare_MEDS_summary_stats examples/stats/v1_baseline.json current.json \
	--on-dataset-version-change relax # exit 0
```

```python
>>> from meds_summary_stats.policy import ComparePolicy
>>> policy = ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
>>> relaxed = compare_summary_stats(baseline, refreshed, policy)
>>> relaxed.status, relaxed.failed, relaxed.counts
('warn', False, {'error': 0, 'info': 60, 'warning': 11})

```

The 60 volume findings drop to informational. What survives as a warning is the *ordering* of the
vocabulary by frequency, which genuinely did shift as the cohort grew, plus the file checksums,
which are of course different files:

```python
>>> sorted({f.path.split(".")[0] for f in relaxed.findings if str(f.severity) == "warning"})
['codes', 'dataset_digest']
>>> sorted(f.path for f in relaxed.findings
...        if str(f.severity) == "warning" and f.path.startswith("codes."))
['codes.digests.by_measurement_frequency', 'codes.digests.by_subject_frequency',
 'codes.digests.numeric_moments.digest']

```

Crucially, `relax` does not blind the check. The set of codes is unchanged here, and if the refresh
*had* also broken the vocabulary or the schema, that would still be a hard error:

```python
>>> baseline["codes"]["digests"]["alphabetical"] == refreshed["codes"]["digests"]["alphabetical"]
True
>>> from copy import deepcopy
>>> also_broken = deepcopy(refreshed)
>>> also_broken["layout"]["data_columns"]["subject_id"] = "Int32"
>>> broken_report = compare_summary_stats(baseline, also_broken, policy)
>>> broken_report.failed
True
>>> [f.path for f in broken_report.findings if str(f.severity) == "error"]
['layout.data_columns.subject_id']

```

The two failure modes are distinguishable, which is the whole point: a data refresh moves volume, a
broken ETL moves volume *and* the vocabulary *and* a distribution.

## What it emits

A single canonical JSON document. Roughly:

| Block           | Contents                                                                                                                   |
| --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `counts`        | subjects, measurements, events, unique codes, static rows, births, deaths, global numeric moments, data-quality invariants |
| `codes`         | vocabulary digests, occurrence distributions, concentration curves                                                         |
| `code_metadata` | metadata coverage, plus per-external-vocabulary statistics and mapping coverage from `parent_codes`                        |
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
    vocabulary changed, and never enough to reconstruct it. Nor is the vocabulary ever partitioned
    by the *structure* of the code strings: splitting on the common `//` convention is not done,
    both because MEDS does not require it and because for a flat vocabulary the prefix is the whole
    code, so the list of prefixes would be the list of codes.
2. **No small cells.** Any split, calendar-year, or ontology-vocabulary entry backed by fewer than
    `min_cell_size` distinct subjects (default 20) is dropped entirely — the key is never written,
    only a `<suppressed>` rollup of how many were dropped and their pooled volume.
3. **No extremes.** No `min`, no `max`, anywhere. Not on values, not on timestamps. The subject with
    the longest record is exactly where re-identification starts; quantiles replace extremes, and
    are themselves suppressed below `min_numeric_cell_size` observations.

The one exception is deliberate, and it is where the deeper vocabulary inspection lives:
`parent_codes` entries name concepts in published external terminologies (`LOINC/8867-4`), which are
properties of the terminology rather than of any patient — and, decisively, the grouping is
*declared by the dataset* rather than guessed from string shape. So that block reports vocabularies
by name, with per-vocabulary volume, subject counts, numeric moments, and key-set digests (still
subject to suppression), plus dataset-level mapping coverage. It is the only axis that says *where*
the vocabulary moved, and it catches a regression nothing else can: when a concept-mapping join
silently stops resolving, every count over the data is unchanged and only
`ontology.coverage.frac_measurements_mapped` falls.

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

Peak memory is bounded by a row budget, not by the size of the dataset. The per-subject and
per-code aggregates count distinct `(subject, time)` and `(subject, code)` pairs, and over a whole
dataset that state grows with every row read, so they are run in chunks of about
`--max-rows-per-chunk` rows (default 10 million), collected one after another and combined exactly.
Within a chunk the aggregates are collected together on the polars streaming engine, so the chunk is
read once.

Combining is exact only if no subject is split between chunks, so chunks follow subjects. MEDS
shards hold disjoint sets of subjects, so consecutive shards are grouped up to the budget and an
oversized shard is cut between subjects. A dataset where one subject spans shards is cut by subject
across all shards instead. That layout is non-conformant, but the tool still has to read it. The
budget only schedules the work: it is not recorded in the output, and the statistics are the same
for any value of it.

The default `numeric-detail: moments` needs no sorting. `quantiles` adds full-column quantiles over
`numeric_value` and `time`, which cannot be combined from chunks and so does materialize those two
columns; it is opt-in for that reason. Per-subject and per-code quantiles are always computed, since
those frames are small.

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
