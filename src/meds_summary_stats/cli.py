"""Command-line entry points.

Two commands, registered as console scripts in ``pyproject.toml``:

* ``extract_MEDS_summary_stats`` / ``python -m meds_summary_stats extract``
* ``compare_MEDS_summary_stats`` / ``python -m meds_summary_stats compare``

Both ``main`` functions take an explicit argument list and return an exit code rather than calling
``sys.exit``, which is what lets the whole CLI surface be exercised from doctests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .compare import compare_summary_stats
from .config import ExtractConfig
from .extract import (
    NotAMEDSDatasetError,
    extract_summary_stats,
    read_summary_stats,
    write_summary_stats,
)
from .policy import ComparePolicy
from .report import render_markdown, render_text

#: Exit code for a comparison that failed its policy. Distinct from ``2``, which argparse uses for
#: a malformed command line, and from ``3``, which signals the inputs could not be read.
EXIT_REGRESSION = 1
EXIT_INPUT_ERROR = 3

#: Defaults shown in ``--help``. A ``slots=True`` dataclass replaces its class attributes with slot
#: descriptors, so the defaults have to be read off an instance rather than off the class.
_EXTRACT_DEFAULTS = ExtractConfig()


def _extract_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the extract command.

    Examples:
        >>> args = _extract_parser().parse_args(["/data/mimic", "-o", "stats.json"])
        >>> args.meds_dir, args.output, args.min_cell_size
        (PosixPath('/data/mimic'), PosixPath('stats.json'), 20)

        >>> _extract_parser().parse_args(["/data/mimic"]).output is None
        True
    """
    parser = argparse.ArgumentParser(
        prog="extract_MEDS_summary_stats",
        description="Extract privacy-safe summary statistics from a MEDS dataset.",
    )
    parser.add_argument("meds_dir", type=Path, help="Root of the MEDS dataset.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Where to write the stats JSON. Default: stdout."
    )
    parser.add_argument("-c", "--config", type=Path, default=None, help="YAML extraction config.")
    parser.add_argument(
        "--min-cell-size",
        type=int,
        default=_EXTRACT_DEFAULTS.min_cell_size,
        help="Minimum distinct subjects for a keyed cell to be reported (default: %(default)s).",
    )
    parser.add_argument(
        "--numeric-detail",
        choices=("moments", "quantiles"),
        default=_EXTRACT_DEFAULTS.numeric_detail,
        help=(
            "'moments' reports counts/mean/std only; 'quantiles' adds full-column quantiles over "
            "numeric_value and time, at meaningfully higher cost (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--emit-ontology-keys",
        action="store_true",
        help="List every key observed in each external vocabulary referenced by parent_codes.",
    )
    return parser


def _resolve_extract_config(args: argparse.Namespace) -> ExtractConfig:
    """Merge the YAML config, if any, with command-line overrides.

    Flags win over the file, but only when the user actually passed them: an unspecified flag sitting
    at its default must not silently override a value the config file set on purpose.

    Examples:
        >>> parser = _extract_parser()
        >>> _resolve_extract_config(parser.parse_args(["/d", "--min-cell-size", "50"])).min_cell_size
        50

        With a config file and no conflicting flag, the file wins:

        >>> with yaml_disk({"c.yaml": {"min_cell_size": 77}}) as root:
        ...     args = parser.parse_args(["/d", "-c", str(root / "c.yaml")])
        ...     _resolve_extract_config(args).min_cell_size
        77

        With both, the explicit flag wins:

        >>> with yaml_disk({"c.yaml": {"min_cell_size": 77}}) as root:
        ...     args = parser.parse_args(["/d", "-c", str(root / "c.yaml"), "--min-cell-size", "5"])
        ...     _resolve_extract_config(args).min_cell_size
        5
    """
    base = ExtractConfig.from_yaml(args.config) if args.config else ExtractConfig()
    overrides = {}
    if args.min_cell_size != _EXTRACT_DEFAULTS.min_cell_size:
        overrides["min_cell_size"] = args.min_cell_size
    if args.numeric_detail != _EXTRACT_DEFAULTS.numeric_detail:
        overrides["numeric_detail"] = args.numeric_detail
    if args.emit_ontology_keys:
        overrides["emit_ontology_keys"] = True
    return ExtractConfig.from_dict({**base.to_dict(), **overrides})


def extract_main(argv: list[str] | None = None) -> int:
    """Run the extract command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp) / "stats.json"
        ...     extract_main([str(simple_static_MEDS), "-o", str(out)])
        ...     stats = json.loads(out.read_text())
        0
        >>> stats["counts"]["n_subjects"]
        6

        A directory that is not a MEDS dataset exits with the input-error code and a message on
        stderr, rather than a traceback:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     extract_main([tmp, "-o", str(Path(tmp) / "s.json")])
        3

        Configuration flows through to the output, and is recorded in it:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp) / "stats.json"
        ...     extract_main([str(simple_static_MEDS), "-o", str(out), "--min-cell-size", "1"])
        ...     stats = json.loads(out.read_text())
        0
        >>> stats["config"]["min_cell_size"]
        1
        >>> sorted(stats["splits"]["by_split"])
        ['held_out', 'train', 'tuning']
    """
    args = _extract_parser().parse_args(argv)
    try:
        cfg = _resolve_extract_config(args)
        stats = extract_summary_stats(args.meds_dir, cfg)
    except (NotAMEDSDatasetError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if args.output is None:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        write_summary_stats(stats, args.output)
        print(f"Wrote summary stats to {args.output}", file=sys.stderr)
    return 0


def _compare_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the compare command.

    Examples:
        >>> args = _compare_parser().parse_args(["old.json", "new.json", "--fail-on", "warning"])
        >>> args.baseline, args.current, args.fail_on
        (PosixPath('old.json'), PosixPath('new.json'), 'warning')
    """
    parser = argparse.ArgumentParser(
        prog="compare_MEDS_summary_stats",
        description="Compare two MEDS summary-stats files and flag regressions.",
    )
    parser.add_argument("baseline", type=Path, help="The reference (known-good) stats file.")
    parser.add_argument("current", type=Path, help="The stats file under test.")
    parser.add_argument("-c", "--config", type=Path, default=None, help="YAML comparison policy.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Where to write the JSON report.")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="Where to write a markdown report for a CI summary."
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default=None,
        help="Minimum severity that fails the run. Overrides the policy file.",
    )
    parser.add_argument(
        "--on-dataset-version-change",
        choices=("ignore", "warn", "error", "relax"),
        default=None,
        help=(
            "What to do when the two files describe different dataset versions. 'relax' downgrades "
            "numeric findings to informational while keeping schema and vocabulary checks hard."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the text report on stdout.")
    return parser


def _resolve_policy(args: argparse.Namespace) -> ComparePolicy:
    """Merge the YAML policy, if any, with command-line overrides.

    Examples:
        >>> parser = _compare_parser()
        >>> _resolve_policy(parser.parse_args(["a", "b"])).fail_on
        <Severity.ERROR: 'error'>
        >>> _resolve_policy(parser.parse_args(["a", "b", "--fail-on", "warning"])).fail_on
        <Severity.WARNING: 'warning'>

        ``never`` clears the threshold entirely, so the run reports findings but always exits
        zero -- useful for a first, advisory rollout of the check:

        >>> _resolve_policy(parser.parse_args(["a", "b", "--fail-on", "never"])).fail_on is None
        True

        >>> args = parser.parse_args(["a", "b", "--on-dataset-version-change", "relax"])
        >>> _resolve_policy(args).on_dataset_version_change
        <DatasetVersionChangeAction.RELAX: 'relax'>
    """
    raw: dict[str, object] = {}
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.fail_on is not None:
        raw["fail_on"] = args.fail_on
    if args.on_dataset_version_change is not None:
        raw["on_dataset_version_change"] = args.on_dataset_version_change
    return ComparePolicy.from_dict(raw)


def compare_main(argv: list[str] | None = None) -> int:
    """Run the compare command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` if the comparison passed, ``1`` if it failed its policy, ``3`` if an input could not
        be read.

    Examples:
        >>> import io, contextlib
        >>> def run(*args):
        ...     buf = io.StringIO()
        ...     with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        ...         code = compare_main(list(args))
        ...     return code, buf.getvalue()

        Extract the same dataset twice and the comparison passes:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     _ = extract_main([str(simple_static_MEDS), "-o", str(tmp / "a.json")])
        ...     _ = extract_main([str(simple_static_MEDS), "-o", str(tmp / "b.json")])
        ...     code, out = run(str(tmp / "a.json"), str(tmp / "b.json"))
        >>> code
        0
        >>> print(out, end="")
        PASS - ... paths checked, ... ignored, no findings.

        Perturb the baseline and the run fails, naming the offending path and writing a machine-
        readable report:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     _ = extract_main([str(simple_static_MEDS), "-o", str(tmp / "a.json")])
        ...     stats = json.loads((tmp / "a.json").read_text())
        ...     stats["counts"]["n_subjects"] = 5
        ...     _ = (tmp / "b.json").write_text(json.dumps(stats))
        ...     code, out = run(str(tmp / "a.json"), str(tmp / "b.json"),
        ...                     "-o", str(tmp / "report.json"))
        ...     report = json.loads((tmp / "report.json").read_text())
        >>> code
        1
        >>> report["status"], report["findings"][0]["path"]
        ('fail', 'counts.n_subjects')

        ``--fail-on never`` reports the same findings but exits zero:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     _ = extract_main([str(simple_static_MEDS), "-o", str(tmp / "a.json")])
        ...     stats = json.loads((tmp / "a.json").read_text())
        ...     stats["counts"]["n_subjects"] = 5
        ...     _ = (tmp / "b.json").write_text(json.dumps(stats))
        ...     code, out = run(str(tmp / "a.json"), str(tmp / "b.json"), "--fail-on", "never")
        >>> code
        0
        >>> out.splitlines()[0].startswith("FAIL")
        True
        >>> out.splitlines()[-1]
        'Errors were found, but fail_on=never, so the run is not failed.'

        A missing input file is an input error, not a regression:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     run(str(Path(tmp) / "nope.json"), str(Path(tmp) / "also-nope.json"))[0]
        3
    """
    args = _compare_parser().parse_args(argv)
    try:
        policy = _resolve_policy(args)
        baseline = read_summary_stats(args.baseline)
        current = read_summary_stats(args.current)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    report = compare_summary_stats(baseline, current, policy)

    if not args.quiet:
        print(render_text(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as f:
            f.write(markdown)

    return EXIT_REGRESSION if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``python -m meds_summary_stats <extract|compare> ...``.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        A process exit code.

    Examples:
        >>> main([])
        usage: python -m meds_summary_stats {extract,compare} ...
        2
        >>> main(["frobnicate"])
        usage: python -m meds_summary_stats {extract,compare} ...
        2
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("extract", "compare"):
        print("usage: python -m meds_summary_stats {extract,compare} ...")
        return 2
    command, *rest = argv
    return extract_main(rest) if command == "extract" else compare_main(rest)
