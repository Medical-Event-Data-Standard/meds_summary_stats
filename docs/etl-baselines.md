# Committed baselines in an ETL repository

An ETL repository can commit one summary-stats fingerprint per kind of dataset it builds (`full` for
the release, `demo` for an open demo, ...) and regenerate it in every pull request that changes the
output, so the diff of the fingerprint shows what the change did, even to a credentialed release that
reviewers cannot open. This page covers the three pieces that make that a few lines of configuration
instead of a script copied into every repository:

- `baseline_MEDS_summary_stats`, which builds, extracts, compares and updates in one command;
- the built-in `etl-regression` comparison policy;
- the `baseline-freshness` action, which reminds a pull request that changes the ETL but not a baseline
    CI cannot rebuild.

## Configure the builds

The repository describes its builds in its `pyproject.toml`:

```toml
[tool.meds_summary_stats.baselines]
# directory = "tests/regression"   # the default; each baseline is <directory>/<kind>_summary_stats.json
# policy = "etl-regression"        # the default; a built-in policy, or a path relative to the project
# Refuse a build without these values in its metadata/dataset.json (dotted paths), e.g. when the last
# stage of the ETL may soft-fail:
require_dataset_metadata = { "sumeds_enrichment.enriched" = true }

[tool.meds_summary_stats.baselines.kinds.full]
build = "uv run --frozen meds-extract-run spec=MY-ETL do_download=false input_dir={input_dir} output_dir={output_dir}"

[tool.meds_summary_stats.baselines.kinds.demo]
build = "uv run --frozen meds-extract-run spec=MY-ETL dataset_key=demo do_download=false input_dir={input_dir} output_dir={output_dir}"
# Optional: how to build without --input-dir, for a dataset the ETL can fetch itself.
build_without_input = "uv run --frozen meds-extract-run spec=MY-ETL dataset_key=demo output_dir={output_dir}"
```

Each `build` is a shell command, run from the project root with the project's own environment, that
writes the MEDS dataset (the directory holding `data/` and `metadata/`) to `{output_dir}`.
`{input_dir}` and `{output_dir}` are replaced by the shell-quoted paths. Anything else a build needs
(an environment variable, a `skip_stages=[...]`, a Hydra override) is part of the command.

## Regenerate or check a baseline

Run it from the repository root. The version is pinned where it is called, because a baseline can only
be compared with fingerprints of the same `stats_schema_version`:

```bash
# Build the kind with this checkout, print how the statistics moved, and overwrite the baseline
uvx --from meds_summary_stats==X.Y.Z baseline_MEDS_summary_stats full --input-dir "$RAW_INPUT_DIR"
# The same from a finished build
uvx --from meds_summary_stats==X.Y.Z baseline_MEDS_summary_stats full --meds-dir "$MEDS_DIR"
# A kind with build_without_input needs neither
uvx --from meds_summary_stats==X.Y.Z baseline_MEDS_summary_stats demo
# Compare instead of updating: writes nothing, exits 1 if the policy flags an error
uvx --from meds_summary_stats==X.Y.Z baseline_MEDS_summary_stats demo --meds-dir "$MEDS_DIR" --compare
```

A build goes to `--output-dir` (default `<tmp>/<project>-<kind>-<commit>`), which must not exist yet.
The build's output goes to stderr; the comparison report is the only thing on stdout. The exit codes
are the other commands': `0` success, `1` a regression (`--compare` only), `2` a malformed command line,
`3` unusable settings or inputs, or a failed build.

## The `etl-regression` policy

With the source release pinned, anything that moves in the fingerprint is something the ETL did, so
`etl-regression` treats the dataset's content as an error: counts, the code vocabulary, metadata
coverage, splits and dtype conflicts. Byte-level digests only warn (a dependency bump can rewrite the
parquet without changing the data), and provenance that changes on every run (creation time, ETL
version, `dataset_version`, suMEDS's enrichment time and version) is ignored. It extends the built-in
defaults, and a path takes its most specific rule, so the defaults written for single paths still
apply: `counts.n_subjects`, `n_events` and `n_measurements` error only beyond 0.5%, for instance. The
policy file, with the reason for every rule, is
[`src/meds_summary_stats/policies/etl-regression.yaml`](../src/meds_summary_stats/policies/etl-regression.yaml).

It is the default of `baseline_MEDS_summary_stats`, and every other place that takes a policy file
takes its name too: `compare_MEDS_summary_stats OLD NEW --config etl-regression`, or `policy: etl-regression` in the action. A repository that needs other rules writes only its overrides:

```yaml
extends: etl-regression
rules:
  - path: splits.**
    kind: exact
    on_change: warning
```

and points `policy` in `pyproject.toml` (and the action's `policy:` input) at that file.

## Check a demo build in CI

When CI can build a kind (an open demo), compare it with the composite action:

```yaml
  - name: Summary-stats regression check
    uses: Medical-Event-Data-Standard/meds_summary_stats@vX.Y.Z
    with:
      meds-dir: ${{ env.MEDS_DEMO_OUTPUT_DIR }}
      baseline: tests/regression/demo_summary_stats.json
      policy: etl-regression
      package-spec: meds_summary_stats==X.Y.Z
      fail-on: error
```

## Remind pull requests to regenerate a credentialed baseline

CI cannot rebuild a credentialed release, so whoever changes the ETL regenerates its baseline where the
data lives. The `baseline-freshness` action warns when a pull request changes the ETL but not the
baseline. It warns rather than fails, because not every contributor has the data and not every change
moves the output:

```yaml
name: Summary stats

on:
  pull_request:

permissions:
  contents: read

jobs:
  baseline-freshness:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v6.0.2
        with:
          fetch-depth: 2 # the pull_request merge commit and its base
          persist-credentials: false
      - uses: >-
          Medical-Event-Data-Standard/meds_summary_stats/baseline-freshness@vX.Y.Z
        with:
          regenerate-command: >-
            uvx --from meds_summary_stats==X.Y.Z baseline_MEDS_summary_stats full
            --input-dir "$RAW_INPUT_DIR"
```

| Input                | Default                                    | Meaning                                                           |
| -------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| `baselines`          | `tests/regression/full_summary_stats.json` | Baselines, one per line, that a change to the ETL must regenerate |
| `watch`              | `^src/`                                    | Extended regex of the paths whose changes can move the output     |
| `ignore`             | `\.md$`                                    | Extended regex of the paths among those that cannot               |
| `base-ref`           | `HEAD^1`                                   | The commit to diff against                                        |
| `regenerate-command` | `baseline_MEDS_summary_stats full ...`     | The command the job summary shows                                 |

Its `stale` output lists the baselines not regenerated, space-separated.

## Migrating from `scripts/summary_stats.sh`

The MEDS ETLs each carried a copy of the same script, policy and workflow. Each variant maps onto the
settings above:

| In the script                                                       | In `pyproject.toml`                                                                            |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `KINDS="full demo"` and the `build()` cases                         | one `kinds.<name>.build` per case, with `"$input"`/`"$output"` as `{input_dir}`/`{output_dir}` |
| `${input:+do_download=false} ${input:+"input_dir=$input"}` (a demo) | `build` with them, and `build_without_input` without                                           |
| the `sumeds_enrichment.enriched` guard                              | `require_dataset_metadata = { "sumeds_enrichment.enriched" = true }`                           |
| `tests/regression/policy.yaml`, identical to the shared one         | nothing: `etl-regression` is the default                                                       |
| `STATS_SPEC="meds_summary_stats==X"`                                | the version in the `uvx --from` call and the action's `package-spec`                           |

and `.github/workflows/summary-stats.yaml` becomes the workflow in the previous section. One
difference: the guard now also applies to `--compare`, where an unenriched build would only have
produced a long list of findings.
