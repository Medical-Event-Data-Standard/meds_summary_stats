# Using this in CI

This repository ships a composite GitHub Action so that an ETL repo can gate its own pull requests
on "the summary statistics have not moved". Two ways to consume it:

- **The composite action** (`Medical-Event-Data-Standard/meds_summary_stats@v0`) — a step inside a
    job that has already produced the dataset. Use this when building the dataset is part of a
    larger job.
- **The reusable workflow** (`.../.github/workflows/meds-stats-regression.yaml@v0`) — a whole job.
    Use this when you want separate logs, a separate runner, and its own summary panel.

Both write a markdown report to the job summary and upload the stats and report as artifacts.

> [!IMPORTANT]
> The action runs wherever your data already is. It does not fetch, upload, or transmit the dataset.
> The only thing it produces is the aggregated fingerprint described in
> [`stats-surface.md`](stats-surface.md), which is why that fingerprint is safe to commit to a
> public repository as a baseline. Read that page before committing one.

## Getting a baseline

A comparison needs something to compare against. Generate it once from a known-good run and commit
it:

```bash
extract_MEDS_summary_stats /path/to/meds/dataset -o baselines/mimic-iv-3.1.json
git add baselines/mimic-iv-3.1.json && git commit -m "Add MEDS summary-stats baseline"
```

Or in CI, by running the action with no `baseline:` input and downloading the artifact.

To see what a baseline looks like before committing one of your own, three are checked in under
[`examples/stats/`](../examples/stats), extracted from the datasets in
[`examples/datasets/`](../examples/datasets). The [README walkthrough](../README.md#a-worked-example)
runs the whole flow against them.

## Example: gate an ETL pull request

The common case. The ETL runs on a small fixture or a licensed dataset available to the runner, and
the job fails if the output has drifted from the committed baseline.

```yaml
name: ETL regression

on: pull_request

permissions:
  contents: read

jobs:
  meds-stats:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v6.0.2
        with:
          persist-credentials: false

      - name: Run the ETL
        run: ./scripts/run_etl.sh --output build/meds

      - name: Check for summary-stat regressions
        uses: Medical-Event-Data-Standard/meds_summary_stats@v0
        with:
          meds-dir: build/meds
          baseline: baselines/mimic-iv-3.1.json
          policy: .github/meds-stats-policy.yaml
          package-spec: meds_summary_stats==0.1.0
```

Pinning `package-spec` matters more than it looks: an unpinned tool that changes how it computes a
statistic will report a regression in your ETL that is really a change in the checker.

## Example: the dataset version changed

When the source data is refreshed, volume drift is expected and flagging it is noise — but an ETL
bug is still an ETL bug. `on-dataset-version-change: relax` draws that line: everything numeric
drops to informational, while schema and vocabulary identity stay hard failures.

```yaml
  - name: Check for summary-stat regressions
    uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    with:
      meds-dir: build/meds
      baseline: baselines/current.json
      on-dataset-version-change: relax
```

This only takes effect when `dataset.dataset_version` actually differs between the two files, so it
is safe to leave on permanently: a same-version comparison is still enforced strictly.

## Example: advisory rollout

Adding this check to a repo that has never had it will surface drift nobody has been watching.
Start with `fail-on: never` — the report is still produced, the job still goes green — read a few
weeks of reports, tune the policy, then switch to `error`.

```yaml
  - uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    with:
      meds-dir: build/meds
      baseline: baselines/current.json
      fail-on: never
```

## Example: as a reusable workflow

```yaml
name: ETL regression

on: pull_request

permissions:
  contents: read

jobs:
  meds-stats:
    uses: >-
      Medical-Event-Data-Standard/meds_summary_stats/.github/workflows/meds-stats-regression.yaml@v0
    with:
      build-command: ./scripts/run_etl.sh --output build/meds
      meds-dir: build/meds
      baseline: baselines/mimic-iv-3.1.json
      policy: .github/meds-stats-policy.yaml
```

## Example: acting on the outcome

The action exposes its verdict as step outputs, so a following step can react without re-reading the
report.

```yaml
  - name: Check for summary-stat regressions
    id: stats
    uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    continue-on-error: true
    with:
      meds-dir: build/meds
      baseline: baselines/current.json

  - name: Open an issue on regression
    if: steps.stats.outputs.failed == 'true'
    env:
      GH_TOKEN: ${{ github.token }}
      STATUS: ${{ steps.stats.outputs.status }}
      REPORT: ${{ steps.stats.outputs.report-path }}
    run: |
      gh issue create \
        --title "MEDS summary-stat regression ($STATUS)" \
        --body-file "$REPORT"
```

Note the `continue-on-error: true`: without it the job stops at the failing step and the follow-up
never runs.

## Example: refreshing the baseline when the change is intentional

Sometimes the drift is the point — you fixed a mapping bug and the numbers are supposed to move. Let
a `refresh-baseline` label on the pull request regenerate the baseline instead of failing.

```yaml
env:
  # A `refresh-baseline` label on the PR says the drift is intentional.
  LABELS: ${{ toJSON(github.event.pull_request.labels.*.name) }}

steps:
  - name: Check for summary-stat regressions
    if: "!contains(env.LABELS, 'refresh-baseline')"
    uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    with:
      meds-dir: build/meds
      baseline: baselines/current.json

  - name: Regenerate the baseline
    if: contains(env.LABELS, 'refresh-baseline')
    uses: Medical-Event-Data-Standard/meds_summary_stats@v0
    with:
      meds-dir: build/meds
      output: baselines/current.json
      artifact-name: updated-baseline
```

The regenerated baseline lands in the `updated-baseline` artifact for a human to download, inspect,
and commit. Committing it automatically from CI would defeat the purpose — a regression checker that
updates its own expectations never fails.

## Inputs

| Input                       | Default                          | Meaning                                                                        |
| --------------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| `meds-dir`                  | *required*                       | Root of the MEDS dataset.                                                      |
| `baseline`                  | `""`                             | Baseline stats JSON. Empty means extract-only.                                 |
| `output`                    | `meds-summary-stats.json`        | Where to write the extracted stats.                                            |
| `report`                    | `meds-summary-stats-report.json` | Where to write the JSON comparison report.                                     |
| `markdown`                  | `meds-summary-stats-report.md`   | Where to write the markdown report.                                            |
| `extract-config`            | `""`                             | YAML extraction config.                                                        |
| `policy`                    | `""`                             | YAML comparison policy. See [`examples/policy.yaml`](../examples/policy.yaml). |
| `min-cell-size`             | *(20)*                           | Suppression floor, overriding the config file.                                 |
| `numeric-detail`            | *(`moments`)*                    | `moments` or `quantiles`.                                                      |
| `fail-on`                   | `error`                          | `error`, `warning`, or `never`.                                                |
| `on-dataset-version-change` | *(`warn`)*                       | `ignore`, `warn`, `error`, or `relax`.                                         |
| `package-spec`              | `meds_summary_stats`             | What to install. Pin it.                                                       |
| `python-version`            | `3.12`                           | Interpreter version.                                                           |
| `upload-artifacts`          | `true`                           | Upload stats and reports as an artifact.                                       |
| `artifact-name`             | `meds-summary-stats`             | Artifact name. Must be unique within a workflow run.                           |

## Outputs

| Output                               | Meaning                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `stats-path`                         | Path to the extracted stats JSON.                                        |
| `report-path`                        | Path to the JSON comparison report; empty when extract-only.             |
| `status`                             | `pass`, `warn`, or `fail` — what was found.                              |
| `failed`                             | `true` if the findings breached `fail-on` — what the policy makes of it. |
| `n_errors`, `n_warnings`             | Finding counts by severity.                                              |
| `dataset_version_changed`, `relaxed` | Whether the version differed, and whether that relaxed the run.          |

`status` and `failed` are separate on purpose. Under `fail-on: never`, `status` can be `fail` while
`failed` is `false`; under `fail-on: warning`, `status` can be `warn` while `failed` is `true`. Gate
your CI on `failed` and report `status` to humans.

## Exit codes

Useful if you invoke the CLI directly rather than through the action.

| Code | Meaning                               |
| ---- | ------------------------------------- |
| 0    | Comparison passed the policy.         |
| 1    | Comparison breached `fail-on`.        |
| 2    | Malformed command line.               |
| 3    | An input could not be read or parsed. |

The action distinguishes these: exit 1 fails the job with a regression message, while anything above
1 fails it as a tool error, because "the checker is broken" and "your ETL regressed" call for
different responses.

## Running without the action

Nothing here requires GitHub Actions. In any CI system:

```bash
uvx --from meds_summary_stats==0.1.0 extract_MEDS_summary_stats build/meds -o current.json
uvx --from meds_summary_stats==0.1.0 compare_MEDS_summary_stats \
	baselines/current.json current.json \
	--config .ci/meds-stats-policy.yaml \
	--output report.json
```

The comparison writes its verdict to `$GITHUB_OUTPUT` and its markdown to `$GITHUB_STEP_SUMMARY`
only when those variables are set, so outside Actions it just prints a table and sets an exit code.
