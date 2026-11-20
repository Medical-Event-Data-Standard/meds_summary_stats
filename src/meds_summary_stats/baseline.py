"""Committed baselines for an ETL repository: build, extract, compare, and update in one command.

An ETL repository commits one fingerprint per kind of dataset it builds (``full`` for the release,
``demo`` for an open demo, ...) and regenerates it in every pull request that changes the output, so
the diff shows what changed. ``baseline_MEDS_summary_stats KIND`` does the whole round trip:

1. build the dataset with the repository's own command (or take an existing build, ``--meds-dir``);
2. check what the repository requires of every baseline build (``require_dataset_metadata``);
3. extract its summary statistics;
4. print how they moved against the committed baseline, under the repository's comparison policy;
5. overwrite the baseline, or with ``--compare`` write nothing and exit ``1`` on a regression.

The repository describes its builds in its ``pyproject.toml``, so no script has to be copied into it:

.. code-block:: toml

    [tool.meds_summary_stats.baselines]
    directory = "tests/regression"   # default; each baseline is <directory>/<kind>_summary_stats.json
    policy = "etl-regression"        # default; a built-in policy, or a path relative to the project
    require_dataset_metadata = { "sumeds_enrichment.enriched" = true }   # optional

    [tool.meds_summary_stats.baselines.kinds.full]
    build = "uv run meds-extract-run spec=X input_dir={input_dir} output_dir={output_dir}"

    [tool.meds_summary_stats.baselines.kinds.demo]
    build = "uv run meds-extract-run spec=X dataset_key=demo input_dir={input_dir} output_dir={output_dir}"
    # How to build it without --input-dir, e.g. a public demo the ETL downloads itself.
    build_without_input = "uv run meds-extract-run spec=X dataset_key=demo output_dir={output_dir}"

See ``docs/etl-baselines.md`` for the full reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .extract import NotAMEDSDatasetError, extract_summary_stats, write_summary_stats

#: Where the settings live in ``pyproject.toml``.
SETTINGS_TABLE = ("tool", "meds_summary_stats", "baselines")

#: Exit codes, as the other commands use them: ``1`` is a regression, ``2`` a malformed command line,
#: ``3`` an input that could not be read or a build that failed.
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_INPUT_ERROR = 3

#: Defaults of the settings. Module constants, because a ``slots=True`` dataclass replaces its class
#: attributes with slot descriptors.
DEFAULT_DIRECTORY = "tests/regression"
DEFAULT_POLICY = "etl-regression"


class BaselineError(Exception):
    """A problem with the settings, the inputs or the build, reported as a message rather than a trace."""


@dataclasses.dataclass(frozen=True, slots=True)
class Kind:
    """How a repository builds one kind of dataset.

    Attributes:
        name: The kind, e.g. ``full`` or ``demo``. Its baseline is ``<directory>/<name>_summary_stats.json``.
        build: The shell command that builds it from ``--input-dir``, run from the project root.
            ``{input_dir}`` and ``{output_dir}`` are replaced by the shell-quoted paths; the dataset
            must be written to ``{output_dir}`` (the directory that holds ``data/`` and ``metadata/``).
        build_without_input: The command used when no ``--input-dir`` is given, for a dataset the
            build can fetch itself (a public demo). Without it, ``--input-dir`` or ``--meds-dir`` is
            required.

    Examples:
        >>> Kind.from_dict("demo", {"build": "etl {input_dir} {output_dir}"})
        Kind(name='demo', build='etl {input_dir} {output_dir}', build_without_input=None)
        >>> Kind.from_dict("demo", {"build": "etl {output_dir}", "bulid_without_input": "x"})
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo: unknown key(s) ['bulid_without_input'];
        expected ['build', 'build_without_input']
        >>> Kind.from_dict("demo", "etl {output_dir}")
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo must be a table with a 'build' command
        >>> Kind.from_dict("demo", {"build": " "})
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo.build must be a non-empty command string
        >>> Kind.from_dict("demo", {"build_without_input": "etl {output_dir}"})
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo needs a 'build' command
        >>> Kind.from_dict("demo", {"build": "etl {input_dir}"})
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo.build must write the dataset to
        {output_dir}
        >>> Kind.from_dict("demo", {"build": "etl {output_dir}",
        ...                         "build_without_input": "etl {input_dir} {output_dir}"})
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: kinds.demo.build_without_input runs without an
        input directory, so it cannot use {input_dir}
    """

    name: str
    build: str
    build_without_input: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Kind:
        """Parse one ``kinds.<name>`` table."""
        known = ["build", "build_without_input"]
        if not isinstance(raw, dict):
            raise BaselineError(f"kinds.{name} must be a table with a 'build' command")
        if unknown := sorted(set(raw) - set(known)):
            raise BaselineError(f"kinds.{name}: unknown key(s) {unknown}; expected {known}")
        for key in known:
            if key in raw and not (isinstance(raw[key], str) and raw[key].strip()):
                raise BaselineError(f"kinds.{name}.{key} must be a non-empty command string")
        if "build" not in raw:
            raise BaselineError(f"kinds.{name} needs a 'build' command")
        for key in known:
            if key in raw and "{output_dir}" not in raw[key]:
                raise BaselineError(f"kinds.{name}.{key} must write the dataset to {{output_dir}}")
        if "{input_dir}" in raw.get("build_without_input", ""):
            raise BaselineError(
                f"kinds.{name}.build_without_input runs without an input directory, so it cannot use "
                "{input_dir}"
            )
        return cls(name=name, build=raw["build"], build_without_input=raw.get("build_without_input"))

    def command(self, output_dir: Path, input_dir: Path | None) -> str:
        """The shell command that builds this kind into ``output_dir``.

        Examples:
            >>> kind = Kind("demo", "etl in={input_dir} out={output_dir}", "etl out={output_dir}")
            >>> kind.command(Path("/tmp/my out"), Path("/data/raw"))
            "etl in=/data/raw out='/tmp/my out'"
            >>> kind.command(Path("/tmp/out"), None)
            'etl out=/tmp/out'
            >>> Kind("full", "etl {input_dir} {output_dir}").command(Path("/tmp/out"), None)
            Traceback (most recent call last):
                ...
            meds_summary_stats.baseline.BaselineError: building 'full' needs --input-dir (the raw
            data), or pass --meds-dir to summarize an existing build
        """
        if input_dir is None and self.build_without_input is None:
            raise BaselineError(
                f"building '{self.name}' needs --input-dir (the raw data), or pass --meds-dir to "
                "summarize an existing build"
            )
        template = self.build if input_dir is not None else self.build_without_input
        command = template.replace("{output_dir}", shlex.quote(str(output_dir)))
        if input_dir is not None:
            command = command.replace("{input_dir}", shlex.quote(str(input_dir)))
        return command


@dataclasses.dataclass(frozen=True, slots=True)
class BaselineSettings:
    """A repository's ``[tool.meds_summary_stats.baselines]`` settings.

    Attributes:
        project_dir: The directory holding ``pyproject.toml``. Paths are relative to it, and builds run
            in it.
        kinds: The kinds of dataset the repository builds, by name.
        directory: Where the baselines are, relative to ``project_dir``.
        policy: The comparison policy: the name of a built-in one (see
            :data:`~meds_summary_stats.policy.PRESETS`) or a path relative to ``project_dir``.
        require_dataset_metadata: Values every baseline build must have in its
            ``metadata/dataset.json``, by dotted path. A build that lacks one is refused before
            anything is extracted, e.g. ``{"sumeds_enrichment.enriched": True}`` for an ETL whose last
            stage may soft-fail.

    Examples:
        >>> toml = '''
        ... [tool.meds_summary_stats.baselines]
        ... require_dataset_metadata = { "sumeds_enrichment.enriched" = true }
        ... [tool.meds_summary_stats.baselines.kinds.full]
        ... build = "etl {input_dir} {output_dir}"
        ... '''
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "pyproject.toml").write_text(toml)
        ...     settings = BaselineSettings.from_project(Path(d))
        >>> list(settings.kinds), settings.directory, settings.policy, settings.require_dataset_metadata
        (['full'], 'tests/regression', 'etl-regression', {'sumeds_enrichment.enriched': True})
        >>> settings.baseline_path("full").relative_to(settings.project_dir)
        PosixPath('tests/regression/full_summary_stats.json')

        A repository without the table, or with a mistyped key, is told so:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "pyproject.toml").write_text("[project]")
        ...     BaselineSettings.from_project(Path(d))
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: .../pyproject.toml has no
        [tool.meds_summary_stats.baselines] table; see docs/etl-baselines.md
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "pyproject.toml").write_text('''
        ... [tool.meds_summary_stats.baselines]
        ... kind = {}
        ... ''')
        ...     BaselineSettings.from_project(Path(d))
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: [tool.meds_summary_stats.baselines]: unknown key(s)
        ['kind']; expected ['directory', 'kinds', 'policy', 'require_dataset_metadata']
        >>> def settings_from(toml):
        ...     with tempfile.TemporaryDirectory() as d:
        ...         _ = (Path(d) / "pyproject.toml").write_text(toml)
        ...         return BaselineSettings.from_project(Path(d))
        >>> settings_from("[tool.meds_summary_stats.baselines]")
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: [tool.meds_summary_stats.baselines] needs at least one
        kinds.<name> table
        >>> settings_from('''
        ... [tool.meds_summary_stats.baselines]
        ... require_dataset_metadata = ["sumeds_enrichment.enriched"]
        ... [tool.meds_summary_stats.baselines.kinds.full]
        ... build = "etl {output_dir}"
        ... ''')
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: require_dataset_metadata must map dotted dataset.json
        paths to values
        >>> with tempfile.TemporaryDirectory() as d:
        ...     BaselineSettings.from_project(Path(d))
        Traceback (most recent call last):
            ...
        meds_summary_stats.baseline.BaselineError: no pyproject.toml in ...; pass --project
    """

    project_dir: Path
    kinds: dict[str, Kind]
    directory: str = DEFAULT_DIRECTORY
    policy: str = DEFAULT_POLICY
    require_dataset_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_project(cls, project_dir: Path) -> BaselineSettings:
        """Read the settings from ``project_dir/pyproject.toml``."""
        pyproject = Path(project_dir) / "pyproject.toml"
        if not pyproject.is_file():
            raise BaselineError(f"no pyproject.toml in {project_dir}; pass --project")
        raw: Any = tomllib.loads(pyproject.read_text())
        for key in SETTINGS_TABLE:
            raw = raw.get(key) if isinstance(raw, dict) else None
        if not isinstance(raw, dict):
            raise BaselineError(
                f"{pyproject} has no [{'.'.join(SETTINGS_TABLE)}] table; see docs/etl-baselines.md"
            )
        known = ["directory", "kinds", "policy", "require_dataset_metadata"]
        if unknown := sorted(set(raw) - set(known)):
            raise BaselineError(f"[{'.'.join(SETTINGS_TABLE)}]: unknown key(s) {unknown}; expected {known}")
        kinds = raw.get("kinds", {})
        if not isinstance(kinds, dict) or not kinds:
            raise BaselineError(f"[{'.'.join(SETTINGS_TABLE)}] needs at least one kinds.<name> table")
        require = raw.get("require_dataset_metadata", {})
        if not isinstance(require, dict):
            raise BaselineError("require_dataset_metadata must map dotted dataset.json paths to values")
        return cls(
            project_dir=Path(project_dir).resolve(),
            kinds={name: Kind.from_dict(name, spec) for name, spec in kinds.items()},
            directory=str(raw.get("directory", DEFAULT_DIRECTORY)),
            policy=str(raw.get("policy", DEFAULT_POLICY)),
            require_dataset_metadata=dict(require),
        )

    def baseline_path(self, kind: str) -> Path:
        """The committed baseline of ``kind``."""
        return self.project_dir / self.directory / f"{kind}_summary_stats.json"

    def policy_spec(self) -> str:
        """The policy as ``compare_MEDS_summary_stats --config`` takes it.

        A built-in name stays a name unless the project has a file of that name; anything else is a
        path relative to the project.

        Examples:
            >>> s = BaselineSettings(project_dir=Path("/repo"), kinds={})
            >>> s.policy_spec()
            'etl-regression'
            >>> dataclasses.replace(s, policy="tests/regression/policy.yaml").policy_spec()
            '/repo/tests/regression/policy.yaml'
        """
        from .policy import PRESETS

        if self.policy in PRESETS and not (self.project_dir / self.policy).exists():
            return self.policy
        return str(self.project_dir / self.policy)

    def check_dataset_metadata(self, meds_dir: Path) -> None:
        """Refuse a build whose ``metadata/dataset.json`` lacks a required value.

        Examples:
            >>> require = {"sumeds_enrichment.enriched": True}
            >>> s = BaselineSettings(Path("."), {}, require_dataset_metadata=require)
            >>> with tempfile.TemporaryDirectory() as d:
            ...     (Path(d) / "metadata").mkdir()
            ...     meta = Path(d) / "metadata" / "dataset.json"
            ...     _ = meta.write_text(json.dumps({"sumeds_enrichment": {"enriched": True}}))
            ...     s.check_dataset_metadata(Path(d))
            ...     _ = meta.write_text(json.dumps({"dataset_name": "X"}))
            ...     s.check_dataset_metadata(Path(d))
            Traceback (most recent call last):
                ...
            meds_summary_stats.baseline.BaselineError: .../metadata/dataset.json has
            sumeds_enrichment.enriched = None, but this repository's baselines require True
            (require_dataset_metadata in pyproject.toml)
        """
        if not self.require_dataset_metadata:
            return
        dataset_json = Path(meds_dir) / "metadata" / "dataset.json"
        dataset = json.loads(dataset_json.read_text())
        for path, expected in self.require_dataset_metadata.items():
            value: Any = dataset
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value != expected:
                raise BaselineError(
                    f"{dataset_json} has {path} = {value!r}, but this repository's baselines require "
                    f"{expected!r} (require_dataset_metadata in pyproject.toml)"
                )


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Examples:
        >>> args = _parser().parse_args(["full", "--input-dir", "/raw", "--compare"])
        >>> args.kind, args.input_dir, args.meds_dir, args.compare
        ('full', PosixPath('/raw'), None, True)
        >>> _parser().parse_args(["full", "--input-dir", "/raw", "--meds-dir", "/m"])
        Traceback (most recent call last):
            ...
        SystemExit: 2
    """
    parser = argparse.ArgumentParser(
        prog="baseline_MEDS_summary_stats",
        description=(
            "Regenerate, or with --compare check, a summary-stats baseline committed in an ETL "
            "repository, as configured in its pyproject.toml ([tool.meds_summary_stats.baselines])."
        ),
    )
    parser.add_argument("kind", help="Which baseline: a kinds.<name> of the settings, e.g. full or demo.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", type=Path, default=None, help="Build from this raw data.")
    source.add_argument(
        "--meds-dir", type=Path, default=None, help="Summarize this existing build instead of building."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to build (must not exist). Default: <tmp>/<project>-<kind>-<git commit>.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with the baseline and exit 1 on a regression, instead of updating it.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="The ETL repository (the directory holding pyproject.toml). Default: the current directory.",
    )
    return parser


def _default_output_dir(settings: BaselineSettings, kind: str) -> Path:
    """``<tmp>/<project>-<kind>-<short commit>``, or without the commit outside a git checkout.

    Examples:
        >>> import subprocess
        >>> with tempfile.TemporaryDirectory() as d:
        ...     repo = Path(d) / "MY_ETL"
        ...     repo.mkdir()
        ...     outside = _default_output_dir(BaselineSettings(repo, {}), "full")
        ...     _ = subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        ...     _ = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@e.org", "commit",
        ...                         "-q", "--allow-empty", "-m", "x"], cwd=repo, check=True)
        ...     inside = _default_output_dir(BaselineSettings(repo, {}), "full")
        >>> outside.name, outside.parent == Path(tempfile.gettempdir())
        ('MY_ETL-full', True)
        >>> inside.name.startswith("MY_ETL-full-"), len(inside.name) > len("MY_ETL-full-")
        (True, True)
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=settings.project_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    name = "-".join(filter(None, (settings.project_dir.name, kind, commit)))
    return Path(tempfile.gettempdir()) / name


def _build(settings: BaselineSettings, kind: Kind, input_dir: Path | None, output_dir: Path | None) -> Path:
    """Run ``kind``'s build into a fresh directory and return it."""
    output_dir = output_dir or _default_output_dir(settings, kind.name)
    if output_dir.exists():
        raise BaselineError(f"{output_dir} already exists; pass it as --meds-dir to reuse a finished build")
    command = kind.command(output_dir.resolve(), input_dir.resolve() if input_dir else None)
    print(f"Building the {kind.name} dataset into {output_dir}:\n  {command}", file=sys.stderr)
    # The build's own output goes to stderr, so that nothing but this command's report is on stdout.
    # Streamed when stderr is a real file; captured and forwarded when it is not (e.g. redirected in a
    # test).
    try:
        stderr_fd: int | None = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        stderr_fd = None
    if stderr_fd is None:
        result = subprocess.run(
            command, shell=True, cwd=settings.project_dir, capture_output=True, text=True, check=False
        )
        sys.stderr.write(result.stdout + result.stderr)
    else:
        result = subprocess.run(command, shell=True, cwd=settings.project_dir, stdout=stderr_fd, check=False)
    if result.returncode != 0:
        raise BaselineError(f"the build exited with code {result.returncode}")
    return output_dir


def baseline_main(argv: list[str] | None = None) -> int:
    """Run the baseline command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` if ``--compare`` found a regression, ``2`` for a malformed command line,
        ``3`` if the settings or inputs are unusable or the build failed.

    Examples:
        A repository whose "build" copies a finished MEDS dataset, so the round trip runs here:

        >>> import contextlib, io, shutil
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = baseline_main([*args, "--project", str(project)])
        ...     return code, out.getvalue(), err.getvalue()
        >>> tmp = tempfile.TemporaryDirectory()
        >>> project = Path(tmp.name) / "repo"
        >>> project.mkdir()
        >>> _ = (project / "pyproject.toml").write_text(f'''
        ... [tool.meds_summary_stats.baselines.kinds.demo]
        ... build = "cp -R {{input_dir}} {{output_dir}}"
        ... build_without_input = "cp -R {simple_static_MEDS} {{output_dir}}"
        ... ''')

        There is nothing to compare with yet:

        >>> code, out, err = run("demo", "--compare", "--meds-dir", str(simple_static_MEDS))
        >>> code, err.strip().endswith("demo_summary_stats.json")
        (3, True)

        The first run builds the dataset and writes the baseline. The build's output goes to stderr,
        streamed when stderr is a file:

        >>> log = Path(tmp.name) / "build.log"
        >>> with log.open("w") as f, contextlib.redirect_stderr(f):
        ...     baseline_main(["demo", "--output-dir", str(Path(tmp.name) / "build1"),
        ...                    "--project", str(project)])
        0
        >>> (project / "tests/regression/demo_summary_stats.json").is_file()
        True
        >>> print(log.read_text().splitlines()[-1])
        Wrote tests/regression/demo_summary_stats.json

        ``--compare`` checks a build against it and writes nothing:

        >>> code, out, err = run("demo", "--compare", "--meds-dir", str(Path(tmp.name) / "build1"))
        >>> code, out.splitlines()[0]
        (0, 'PASS - ... paths checked, ... ignored, no findings.')

        A build that differs fails the comparison, under the ``etl-regression`` policy by default:

        >>> changed = Path(tmp.name) / "changed"
        >>> _ = shutil.copytree(simple_static_MEDS, changed)
        >>> (changed / "data" / "held_out" / "0.parquet").unlink()
        >>> code, out, err = run("demo", "--compare", "--meds-dir", str(changed))
        >>> code, out.splitlines()[0].startswith("FAIL")
        (1, True)

        Updating from it prints the same findings, never fails, and overwrites the baseline:

        >>> code, out, err = run("demo", "--meds-dir", str(changed))
        >>> code, out.splitlines()[0].startswith("FAIL"), err.splitlines()[-1]
        (0, True, 'Wrote tests/regression/demo_summary_stats.json')
        >>> run("demo", "--compare", "--meds-dir", str(changed))[0]
        0

        An unreadable baseline is not overwritten:

        >>> baseline = project / "tests/regression/demo_summary_stats.json"
        >>> _ = baseline.write_text("not json")
        >>> run("demo", "--meds-dir", str(changed))[0], baseline.read_text()
        (3, 'not json')

        Mistakes are reported as messages with the input-error code:

        >>> code, out, err = run("full")
        >>> code, err.strip()
        (3, "error: unknown kind 'full'; this repository's kinds are ['demo']")
        >>> code, out, err = run("demo", "--output-dir", str(Path(tmp.name) / "build1"))
        >>> code, err.strip().endswith("already exists; pass it as --meds-dir to reuse a finished build")
        (3, True)
        >>> code, out, err = run("demo", "--input-dir", str(Path(tmp.name) / "nope"),
        ...                      "--output-dir", str(Path(tmp.name) / "build2"))
        >>> code, err.splitlines()[-1]
        (3, 'error: the build exited with code 1')
        >>> code, out, err = run("demo", "--meds-dir", tmp.name)
        >>> code, err.splitlines()[-1].endswith("is not a MEDS dataset (no metadata/dataset.json)")
        (3, True)
        >>> tmp.cleanup()
    """
    from .cli import compare_main

    args = _parser().parse_args(argv)
    try:
        settings = BaselineSettings.from_project(args.project)
        if args.kind not in settings.kinds:
            raise BaselineError(
                f"unknown kind '{args.kind}'; this repository's kinds are {sorted(settings.kinds)}"
            )
        kind = settings.kinds[args.kind]
        meds_dir = args.meds_dir or _build(settings, kind, args.input_dir, args.output_dir)
        if not (meds_dir / "metadata" / "dataset.json").is_file():
            raise BaselineError(f"{meds_dir} is not a MEDS dataset (no metadata/dataset.json)")
        settings.check_dataset_metadata(meds_dir)
        baseline = settings.baseline_path(kind.name)
        if args.compare and not baseline.is_file():
            raise BaselineError(f"no baseline at {baseline}")
        print(f"Summarizing {meds_dir}", file=sys.stderr)
        stats = extract_summary_stats(meds_dir)
    except (BaselineError, NotAMEDSDatasetError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    policy = settings.policy_spec()
    with tempfile.TemporaryDirectory() as tmp:
        current = Path(tmp) / f"{kind.name}_summary_stats.json"
        write_summary_stats(stats, current)
        if args.compare:
            return compare_main([str(baseline), str(current), "--config", policy])
        if baseline.is_file():
            print("Changes against the previous baseline:", file=sys.stderr)
            code = compare_main([str(baseline), str(current), "--config", policy, "--fail-on", "never"])
            if code != 0:
                return code
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(current.read_bytes())
    print(f"Wrote {baseline.relative_to(settings.project_dir)}", file=sys.stderr)
    return 0
