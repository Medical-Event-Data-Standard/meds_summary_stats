# Agent Guidelines for McDermott Health AI Lab Projects

This file is read by AI coding agents (Claude Code, Cursor, Copilot, Codex CLI, Aider, Gemini CLI,
and others). `CLAUDE.md` is a symlink to this file.

## Read `CONTRIBUTORS.md` first

**The project's conventions live in [`CONTRIBUTORS.md`](CONTRIBUTORS.md): build system (uv),
testing, code style, issue/PR workflow, repository conventions.** Follow them. The notes below are
agent-specific additions and reminders that build on those conventions — they are not a substitute.

## Agent-specific tooling

- **Use `gh` CLI for all GitHub operations** (PRs, issues, code search, actions logs). Do **not**
  use the GitHub MCP server — `gh` is faster, more reliable, and uses far fewer tokens for the
  same operations.
- **Doctest namespace is pre-populated.** The project root `conftest.py` registers `Path`,
  `datetime`, `tempfile`, plus `yaml_to_disk` and `print_directory` / `PrintConfig` (if installed)
  into the doctest namespace. You rarely need explicit imports inside doctests.

## Working style

- **TDD.** When fixing bugs or adding features, write a failing test first, confirm the failure
  captures the intended behavior, then implement the fix.
- **Pre-commit hooks auto-fix many issues** (formatting, import sorting, EOF newlines, markdown
  format). Run `uv run pre-commit run --all-files` before committing — it will modify files. Stage
  the result.
- **Doctests are first-class.** Prefer adding API-validating tests as doctests in docstrings or
  markdown files; standalone `tests/**/test_*.py` files are for cases where a doctest would be
  excessively long or unclear.

## What not to do

- Do not use `pip install` directly — use `uv sync` or `uv add` (see CONTRIBUTORS.md "Build
  System: uv").
- Do not skip or disable pre-commit hooks (no `--no-verify`).
- Do not add broad `# noqa` or `# type: ignore` comments without a specific rule code and a
  one-line justification.
- Do not modify CI workflows (`.github/workflows/*.yaml`) without discussing with maintainers
  first. Workflow security is enforced by `zizmor` (see `.github/zizmor.yml`); changes there often
  surface real security findings.
- Do not commit data files or secrets. `gitleaks` runs as a pre-commit hook but treat it as a
  backstop, not a substitute for care.

## Project-specific conventions

### The privacy contract is the product

Three invariants hold everywhere in `extract.py`, and each is covered by a hypothesis property in
`tests/test_properties.py`:

1. **No code string is ever emitted.** Not at any suppression threshold, not under any flag. If you
    add a field, ask whether an adversary could read a code out of it.
    **Do not partition the vocabulary by the shape of the code strings.** Splitting on the `//`
    convention is tempting and is deliberately absent: MEDS does not require it, so the axis would
    be present on some datasets and silently missing on others, and for a flat vocabulary the
    prefix *is* the code, so the prefix list would be the vocabulary. `parent_codes` is the only
    sanctioned grouping axis, because those groups are declared rather than inferred.
    `tests/test_integration.py::test_the_slash_slash_convention_is_never_interpreted` guards this.
2. **No keyed cell below `min_cell_size` distinct subjects.** Route new keyed blocks through
    `_suppress_keyed` rather than filtering by hand.
3. **No `min` or `max`, anywhere.** Use quantiles. This is not negotiable for a new statistic.

Parent codes from `metadata/codes.parquet` are the one exception: they name published external
terminologies, so vocabulary names are safe to emit. Their *statistics* are still suppressed.

### Changing the emitted document

Any change to the meaning of an existing field, or to a digest encoding, is a breaking change:
bump `STATS_SCHEMA_VERSION` in `extract.py`. Comparison refuses to diff across versions, which is
the point -- a silently-reinterpreted field would show up as unexplained findings in someone's CI.

Purely additive fields do not need a bump, but do need a default rule in `DEFAULT_RULES`
(`policy.py`) if the catch-all `**` rule is wrong for them.

### Digests must stay reproducible

`digests.py` pins its encoding deliberately. Floating-point summation order varies with shard and
thread count, which is why moments are rounded via `format_moment` before hashing. If you find
yourself wanting to hash a raw float, you want `format_moment`.

### Testing

- Doctests carry essentially all API-level coverage; `tests/` holds only end-to-end runs and
    hypothesis properties. A new function should arrive with doctests, not a new test file.
- `conftest.py` puts `pl`, `pa`, `json`, `SUPPRESSED`, and the two `meds-testing-helpers` dataset
    fixtures (`simple_static_MEDS`, `simple_static_MEDS_dataset_with_task`) into the doctest
    namespace, alongside the auto-registered `yaml_disk`, `print_directory`, and `PrintConfig`.
- Build datasets inline with `yaml_disk`; it writes `.parquet` from a column dict, a row-dict list,
    or a `pyarrow.Table` (use the last when a doctest needs to control dtypes).
- `docs/*.md` is collected by pytest. Examples there are executed, so they cannot drift.
- `action.yml` is exercised by `.github/workflows/action-selftest.yaml`, not by pytest. Changes to
    the composite action need a look at that workflow.
