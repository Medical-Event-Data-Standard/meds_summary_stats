"""Comparison of two extracted stats documents against a tolerance policy."""

from __future__ import annotations

import dataclasses
import datetime
import math
import re
from typing import Any

from .extract import STATS_SCHEMA_VERSION
from .flatten import flatten
from .policy import ComparePolicy, DatasetVersionChangeAction, Kind, Rule, Severity

#: Paths that describe the extraction itself rather than the dataset, and so must agree exactly
#: before any other comparison is meaningful.
_CONFIG_PREFIX = "config."

#: The exact shape extraction writes a timestamp in: ``datetime.isoformat()`` of a naive datetime.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?")

_SECONDS_PER_DAY = 86_400


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Finding:
    """A single flagged difference.

    Attributes:
        path: The flattened stats path.
        severity: How seriously to take it.
        kind: The comparison that produced it, or ``"added"`` / ``"removed"`` for key changes.
        baseline: The baseline value, if any.
        current: The current value, if any.
        delta: Signed absolute difference, for numeric comparisons.
        relative_delta: Signed relative difference, for numeric comparisons.
        threshold: The threshold that was exceeded, if any.
        message: A one-line human-readable summary.
    """

    path: str
    severity: Severity
    kind: str
    baseline: Any = None
    current: Any = None
    delta: float | None = None
    relative_delta: float | None = None
    threshold: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON, dropping fields that do not apply.

        Examples:
            >>> from pprint import pprint
            >>> pprint(Finding(path="counts.n_subjects", severity=Severity.ERROR, kind="relative",
            ...                baseline=100, current=90, delta=-10.0, relative_delta=-0.1,
            ...                threshold=0.005, message="dropped").to_dict())
            {'baseline': 100,
             'current': 90,
             'delta': -10.0,
             'kind': 'relative',
             'message': 'dropped',
             'path': 'counts.n_subjects',
             'relative_delta': -0.1,
             'severity': 'error',
             'threshold': 0.005}

            Inapplicable numeric fields are omitted rather than emitted as nulls, so an ``exact``
            finding does not carry three meaningless keys:

            >>> pprint(Finding(path="layout.data_columns.code", severity=Severity.ERROR,
            ...                kind="exact", baseline="String", current="Int64",
            ...                message="changed").to_dict())
            {'baseline': 'String',
             'current': 'Int64',
             'kind': 'exact',
             'message': 'changed',
             'path': 'layout.data_columns.code',
             'severity': 'error'}
        """
        out = {
            "path": self.path,
            "severity": str(self.severity),
            "kind": self.kind,
            "baseline": self.baseline,
            "current": self.current,
            "message": self.message,
        }
        for name in ("delta", "relative_delta", "threshold"):
            if (value := getattr(self, name)) is not None:
                out[name] = value
        return dict(sorted(out.items()))


def _is_number(value: Any) -> bool:
    """Whether a value can participate in a numeric comparison.

    Booleans are excluded deliberately: ``True`` is numerically ``1``, so a flag flipping from false
    to true would otherwise register as a 100% relative change rather than as the categorical
    difference it is.

    Examples:
        >>> [_is_number(v) for v in (1, 1.5, True, None, "3", float("nan"))]
        [True, True, False, False, False, True]
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


def _timestamp(value: Any) -> datetime.datetime | None:
    """Parse a timestamp written by extraction, or return ``None`` for anything else.

    Deliberately narrower than :meth:`datetime.datetime.fromisoformat`, which also accepts dates,
    week dates, and basic-format strings such as ``"20200101"``. Under that parser a version string
    or a code count could be read as a date and silently compared as a span of days. Only the shape
    extraction emits -- ``YYYY-MM-DDTHH:MM:SS`` with optional microseconds and no offset -- counts.

    Examples:
        >>> _timestamp("2021-12-19T19:17:48")
        datetime.datetime(2021, 12, 19, 19, 17, 48)
        >>> _timestamp("2021-12-19T19:17:48.500000")
        datetime.datetime(2021, 12, 19, 19, 17, 48, 500000)
        >>> [_timestamp(v) for v in ("20200101", "2020-01-01", "2020-13-01T00:00:00", 2020, None)]
        [None, None, None, None, None]
    """
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


def relative_delta(baseline: float, current: float) -> float:
    """Signed relative change from ``baseline`` to ``current``.

    The denominator is ``max(|baseline|, 1)`` rather than ``|baseline|``. Flooring it at 1 keeps a
    baseline of zero from dividing by zero, and keeps sub-unit baselines (fractions and rates, which
    this document is full of) from turning a negligible absolute move into an enormous ratio. Above
    1 the floor does nothing, so ordinary counts compare on ordinary relative terms.

    Args:
        baseline: The baseline value.
        current: The current value.

    Returns:
        The signed relative change.

    Examples:
        >>> relative_delta(1000, 1010)
        0.01
        >>> relative_delta(1000, 990)
        -0.01

        Small counts genuinely do move by large fractions, and are reported that way:

        >>> relative_delta(2, 3)
        0.5

        Zero baselines are handled without a division by zero:

        >>> relative_delta(0, 0), relative_delta(0, 5)
        (0.0, 5.0)

        Non-finite values compare as "changed if not identical", since arithmetic on them is
        meaningless:

        >>> relative_delta(float("nan"), float("nan"))
        0.0
        >>> relative_delta(1.0, float("inf"))
        inf
    """
    if not math.isfinite(baseline) or not math.isfinite(current):
        if (math.isnan(baseline) and math.isnan(current)) or baseline == current:
            return 0.0
        return math.inf
    return (current - baseline) / max(abs(baseline), 1.0)


def _severity_for(rule: Rule, magnitude: float) -> Severity | None:
    """Classify a numeric difference against a rule's thresholds.

    Examples:
        >>> rule = Rule(path="x", kind=Kind.RELATIVE, warn=0.01, error=0.05)
        >>> _severity_for(rule, 0.001) is None
        True
        >>> _severity_for(rule, 0.02), _severity_for(rule, 0.5)
        (<Severity.WARNING: 'warning'>, <Severity.ERROR: 'error'>)

        Thresholds are exclusive, so a difference exactly at the limit passes:

        >>> _severity_for(rule, 0.01) is None
        True

        A rule with only an error tier never emits warnings:

        >>> _severity_for(Rule(path="x", kind=Kind.RELATIVE, error=0.05), 0.02) is None
        True
    """
    if rule.error is not None and magnitude > rule.error:
        return Severity.ERROR
    if rule.warn is not None and magnitude > rule.warn:
        return Severity.WARNING
    return None


def _compare_collection(path: str, baseline: list, current: list, rule: Rule) -> Finding | None:
    """Compare two list-valued leaves by membership rather than by position.

    :func:`~meds_summary_stats.flatten.flatten` keeps lists of scalars whole precisely so this can
    happen: the interesting facts about a column list are which members appeared and which
    disappeared, and both are lost if the list is compared index by index.

    Args:
        path: The flattened path.
        baseline: The baseline list.
        current: The current list.
        rule: The governing rule.

    Returns:
        A finding naming the added and removed members, or ``None`` if the lists are equal.

    Examples:
        >>> rule = Rule(path="**", kind=Kind.EXACT)
        >>> _compare_collection("layout.extra_data_columns", ["a", "b"], ["a", "b"], rule) is None
        True

        One added column is one finding that names it, not one per shifted index:

        >>> f = _compare_collection("layout.extra_data_columns", ["b", "c"], ["a", "b", "c"], rule)
        >>> f.kind, f.message
        ('set', "added 'a'")

        Additions and removals are reported together:

        >>> f = _compare_collection("layout.code_metadata_columns", ["itemid", "x"], ["y", "x"], rule)
        >>> print(f.message)
        added 'y'; removed 'itemid'

        Same members in a different order is a real change for an ordered config, so it is still
        reported -- but it is described as what it is:

        >>> f = _compare_collection("config.quantiles", [0.1, 0.9], [0.9, 0.1], rule)
        >>> print(f.message)
        reordered, same members

        Unhashable members (never produced by extraction, but policies are user-facing) fall back
        to whole-value equality rather than crashing:

        >>> f = _compare_collection("x", [[1]], [[2]], rule)
        >>> f.kind, f.message
        ('exact', 'changed from [[1]] to [[2]]')
    """
    if baseline == current:
        return None

    severity = rule.on_change if rule.kind is Kind.EXACT else Severity.ERROR
    try:
        before, after = set(baseline), set(current)
    except TypeError:
        return Finding(
            path=path,
            severity=severity,
            kind="exact",
            baseline=baseline,
            current=current,
            message=f"changed from {baseline!r} to {current!r}",
        )

    added = sorted(after - before, key=repr)
    removed = sorted(before - after, key=repr)
    if added or removed:
        parts = []
        if added:
            parts.append("added " + ", ".join(repr(v) for v in added))
        if removed:
            parts.append("removed " + ", ".join(repr(v) for v in removed))
        message = "; ".join(parts)
    else:
        message = "reordered, same members"

    return Finding(
        path=path,
        severity=severity,
        kind="set",
        baseline=baseline,
        current=current,
        message=message,
    )


def _compare_leaf(path: str, baseline: Any, current: Any, rule: Rule) -> Finding | None:
    """Compare one path's two values under its rule.

    Args:
        path: The flattened path.
        baseline: The baseline value.
        current: The current value.
        rule: The governing rule.

    Returns:
        A finding, or ``None`` if the values agree within tolerance.

    Examples:
        Exact rules flag any inequality and say nothing otherwise:

        >>> rule = Rule(path="**", kind=Kind.EXACT)
        >>> _compare_leaf("layout.data_columns.code", "String", "String", rule) is None
        True
        >>> f = _compare_leaf("layout.data_columns.code", "String", "Int64", rule)
        >>> f.severity, f.message
        (<Severity.ERROR: 'error'>, "changed from 'String' to 'Int64'")

        Relative rules scale with the baseline:

        >>> rule = Rule(path="**", kind=Kind.RELATIVE, warn=0.001, error=0.01)
        >>> _compare_leaf("counts.n_measurements", 1_000_000, 1_002_000, rule).severity
        <Severity.WARNING: 'warning'>
        >>> _compare_leaf("counts.n_measurements", 1_000_000, 1_000_000, rule) is None
        True
        >>> f = _compare_leaf("counts.n_measurements", 1_000_000, 900_000, rule)
        >>> f.severity, f.relative_delta, f.threshold
        (<Severity.ERROR: 'error'>, -0.1, 0.01)

        Ignored paths never produce findings, whatever they contain:

        >>> _compare_leaf("extracted_at", "a", "b", Rule(path="**", kind=Kind.IGNORE)) is None
        True

        A numeric rule applied to non-numeric values falls back to exact comparison rather than
        crashing -- policies are user-written and will inevitably point a ``relative`` rule at a
        string:

        >>> f = _compare_leaf("dataset.dataset_name", "mimic", "eicu",
        ...                   Rule(path="**", kind=Kind.RELATIVE, error=0.01))
        >>> f.kind, f.severity
        ('exact', <Severity.ERROR: 'error'>)

        A value that changes type between documents is likewise a categorical change:

        >>> f = _compare_leaf("counts.n_subjects", 100, None,
        ...                   Rule(path="**", kind=Kind.RELATIVE, error=0.01))
        >>> f.kind, f.message
        ('exact', 'changed from 100 to None')

        Two list-valued leaves are diffed by membership -- see :func:`_compare_collection`:

        >>> f = _compare_leaf("layout.extra_data_columns", ["b"], ["a", "b"],
        ...                   Rule(path="**", kind=Kind.EXACT))
        >>> f.kind, f.message
        ('set', "added 'a'")

        Under an ``absolute`` rule, two timestamps compare as a distance in days, so the timestamp
        quantiles get a tolerance like every other quantile. The default policy warns on any
        movement and errors beyond one day:

        >>> policy = ComparePolicy.default()
        >>> rule = policy.rule_for("time.quantiles.p99")
        >>> rule.path, rule.kind
        ('time.quantiles.*', <Kind.ABSOLUTE: 'absolute'>)
        >>> f = _compare_leaf("time.quantiles.p99", "2021-12-19T19:17:48", "2021-12-19T19:54:24", rule)
        >>> f.severity, f.kind, f.threshold
        (<Severity.WARNING: 'warning'>, 'absolute', 0.0)
        >>> print(f.message)
        2021-12-19T19:17:48 -> 2021-12-19T19:54:24 (+0.0254167 days), over absolute threshold 0.0 days
        >>> f = _compare_leaf("time.quantiles.p50", "2019-07-30T17:01:00", "2019-08-05T12:50:00", rule)
        >>> f.severity, round(f.delta, 4)
        (<Severity.ERROR: 'error'>, 5.8257)
        >>> _compare_leaf("time.quantiles.p50", "2019-07-30T17:01:00", "2019-07-30T17:01:00", rule) is None
        True

        Only the exact shape extraction writes counts as a timestamp. ``"20200101"`` is valid ISO
        8601 too, but read as a date it would turn a version string into a span of days, so it falls
        back to exact comparison like any other string:

        >>> f = _compare_leaf("dataset.dataset_version", "20200101", "20200102",
        ...                   Rule(path="**", kind=Kind.ABSOLUTE, error=1.0))
        >>> f.kind, f.severity
        ('exact', <Severity.ERROR: 'error'>)
    """
    if rule.kind is Kind.IGNORE:
        return None

    if isinstance(baseline, list) and isinstance(current, list):
        return _compare_collection(path, baseline, current, rule)

    if rule.kind is Kind.ABSOLUTE and (before := _timestamp(baseline)) and (after := _timestamp(current)):
        # A relative threshold on a timestamp would be relative to the Unix epoch, which means
        # nothing; an absolute distance has a natural unit, and that unit is days.
        delta = (after - before).total_seconds() / _SECONDS_PER_DAY
        severity = _severity_for(rule, abs(delta))
        if severity is None:
            return None
        threshold = rule.error if severity is Severity.ERROR else rule.warn
        return Finding(
            path=path,
            severity=severity,
            kind=str(rule.kind),
            baseline=baseline,
            current=current,
            delta=delta,
            threshold=threshold,
            message=f"{baseline} -> {current} ({delta:+.6g} days), over absolute threshold {threshold} days",
        )

    if rule.kind is Kind.EXACT or not (_is_number(baseline) and _is_number(current)):
        if baseline == current or (
            _is_number(baseline) and _is_number(current) and math.isnan(baseline) and math.isnan(current)
        ):
            return None
        severity = rule.on_change if rule.kind is Kind.EXACT else Severity.ERROR
        return Finding(
            path=path,
            severity=severity,
            kind="exact",
            baseline=baseline,
            current=current,
            message=f"changed from {baseline!r} to {current!r}",
        )

    delta = float(current) - float(baseline) if math.isfinite(current - baseline) else math.inf
    rel = relative_delta(float(baseline), float(current))
    magnitude = abs(delta) if rule.kind is Kind.ABSOLUTE else abs(rel)
    severity = _severity_for(rule, magnitude)
    if severity is None:
        return None
    threshold = rule.error if severity is Severity.ERROR else rule.warn
    unit = "" if rule.kind is Kind.ABSOLUTE else f" ({rel:+.2%})"
    return Finding(
        path=path,
        severity=severity,
        kind=str(rule.kind),
        baseline=baseline,
        current=current,
        delta=delta,
        relative_delta=rel,
        threshold=threshold,
        message=f"{baseline} -> {current}{unit}, over {rule.kind.value} threshold {threshold}",
    )


def _check_preconditions(baseline: dict[str, Any], current: dict[str, Any]) -> list[Finding]:
    """Verify the two documents are comparable at all.

    Comparing documents extracted under different configs is worse than useless: digests taken at
    different subject thresholds are different quantities, and their inequality carries no
    information. The same goes for a format version bump. Both are reported as errors on their own
    terms rather than showing up as a scatter of unexplained findings.

    Examples:
        >>> base = {"stats_schema_version": 1, "config": {"min_cell_size": 20}}
        >>> _check_preconditions(base, base)
        []

        >>> findings = _check_preconditions(base, {"stats_schema_version": 1,
        ...                                        "config": {"min_cell_size": 10}})
        >>> findings[0].path, findings[0].severity
        ('config.min_cell_size', <Severity.ERROR: 'error'>)
        >>> print(findings[0].message)
        extraction config differs (20 -> 10); the two documents are not comparable

        >>> _check_preconditions({"stats_schema_version": 1}, {"stats_schema_version": 2})[0].message
        'stats format versions differ (1 -> 2); re-extract both with the same build'
    """
    findings: list[Finding] = []
    if baseline.get("stats_schema_version") != current.get("stats_schema_version"):
        findings.append(
            Finding(
                path="stats_schema_version",
                severity=Severity.ERROR,
                kind="precondition",
                baseline=baseline.get("stats_schema_version"),
                current=current.get("stats_schema_version"),
                message=(
                    f"stats format versions differ ({baseline.get('stats_schema_version')} -> "
                    f"{current.get('stats_schema_version')}); re-extract both with the same build"
                ),
            )
        )

    base_cfg = flatten(baseline.get("config", {}), "config")
    curr_cfg = flatten(current.get("config", {}), "config")
    for path in sorted(set(base_cfg) | set(curr_cfg)):
        before, after = base_cfg.get(path), curr_cfg.get(path)
        if before != after:
            findings.append(
                Finding(
                    path=path,
                    severity=Severity.ERROR,
                    kind="precondition",
                    baseline=before,
                    current=after,
                    message=(
                        f"extraction config differs ({before} -> {after}); "
                        "the two documents are not comparable"
                    ),
                )
            )
    return findings


def _dataset_version_finding(
    baseline: dict[str, Any], current: dict[str, Any], policy: ComparePolicy
) -> Finding | None:
    """Report a dataset-version change according to the policy.

    Examples:
        >>> from .policy import ComparePolicy
        >>> base = {"dataset": {"dataset_version": "1.0"}}
        >>> curr = {"dataset": {"dataset_version": "2.0"}}
        >>> _dataset_version_finding(base, base, ComparePolicy.default()) is None
        True
        >>> f = _dataset_version_finding(base, curr, ComparePolicy.default())
        >>> f.severity, f.message
        (<Severity.WARNING: 'warning'>, "dataset_version changed from '1.0' to '2.0'")

        Under ``relax`` the change is reported only as information, because the caller has said in
        advance that a data change makes volume drift expected:

        >>> policy = ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
        >>> _dataset_version_finding(base, curr, policy).severity
        <Severity.INFO: 'info'>

        Under ``ignore`` the change is not reported at all:

        >>> ignoring = ComparePolicy.from_dict({"on_dataset_version_change": "ignore"})
        >>> _dataset_version_finding(base, curr, ignoring) is None
        True
    """
    before = (baseline.get("dataset") or {}).get("dataset_version")
    after = (current.get("dataset") or {}).get("dataset_version")
    if before == after:
        return None
    action = policy.on_dataset_version_change
    if action is DatasetVersionChangeAction.IGNORE:
        return None
    severity = {
        DatasetVersionChangeAction.WARN: Severity.WARNING,
        DatasetVersionChangeAction.ERROR: Severity.ERROR,
        DatasetVersionChangeAction.RELAX: Severity.INFO,
    }[action]
    return Finding(
        path="dataset.dataset_version",
        severity=severity,
        kind="dataset_version",
        baseline=before,
        current=after,
        message=f"dataset_version changed from {before!r} to {after!r}",
    )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ComparisonReport:
    """The result of comparing two stats documents.

    ``status`` describes what was found; ``failed`` says what the policy makes of it. They are kept
    separate so that ``fail_on: never`` still reports honestly that there are errors, and
    ``fail_on: warning`` can fail a run whose worst finding is only a warning.

    Attributes:
        status: ``"pass"`` (no findings), ``"warn"`` (findings, none of them errors), or ``"fail"``
            (at least one error).
        failed: Whether the worst finding reached the policy's ``fail_on`` threshold.
        fail_on: The threshold that was applied, for reporting.
        findings: Every finding, most severe first.
        n_paths_checked: How many leaf paths were compared.
        n_paths_ignored: How many were skipped by an ``ignore`` rule.
        dataset_version_changed: Whether the two documents describe different dataset versions.
        relaxed: Whether numeric findings were downgraded due to a dataset-version change.
    """

    status: str
    failed: bool
    fail_on: str
    findings: tuple[Finding, ...]
    n_paths_checked: int
    n_paths_ignored: int
    dataset_version_changed: bool
    relaxed: bool

    @property
    def counts(self) -> dict[str, int]:
        """Findings tallied by severity.

        Examples:
            >>> findings = (Finding(path="a", severity=Severity.ERROR, kind="exact"),
            ...             Finding(path="b", severity=Severity.WARNING, kind="exact"))
            >>> report = ComparisonReport(status="fail", failed=True, fail_on="error",
            ...                           n_paths_checked=3, n_paths_ignored=0,
            ...                           dataset_version_changed=False, relaxed=False,
            ...                           findings=findings)
            >>> report.counts
            {'error': 1, 'info': 0, 'warning': 1}
        """
        return {
            str(s): sum(1 for f in self.findings if f.severity is s)
            for s in (Severity.ERROR, Severity.INFO, Severity.WARNING)
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to plain JSON.

        Examples:
            >>> report = ComparisonReport(status="pass", failed=False, fail_on="error",
            ...                           findings=(), n_paths_checked=42,
            ...                           n_paths_ignored=5, dataset_version_changed=False,
            ...                           relaxed=False)
            >>> from pprint import pprint
            >>> pprint(report.to_dict())
            {'counts': {'error': 0, 'info': 0, 'warning': 0},
             'dataset_version_changed': False,
             'fail_on': 'error',
             'failed': False,
             'findings': [],
             'n_paths_checked': 42,
             'n_paths_ignored': 5,
             'relaxed': False,
             'status': 'pass'}
        """
        return {
            "status": self.status,
            "failed": self.failed,
            "fail_on": self.fail_on,
            "counts": self.counts,
            "n_paths_checked": self.n_paths_checked,
            "n_paths_ignored": self.n_paths_ignored,
            "dataset_version_changed": self.dataset_version_changed,
            "relaxed": self.relaxed,
            "findings": [f.to_dict() for f in self.findings],
        }


def compare_summary_stats(
    baseline: dict[str, Any],
    current: dict[str, Any],
    policy: ComparePolicy | None = None,
) -> ComparisonReport:
    """Compare two stats documents under a tolerance policy.

    Args:
        baseline: The reference document (the known-good extraction).
        current: The document under test.
        policy: The tolerance policy. Defaults to :meth:`ComparePolicy.default`.

    Returns:
        A report whose ``status`` is ``"pass"``, ``"warn"``, or ``"fail"``.

    Examples:
        >>> from copy import deepcopy
        >>> base = {
        ...     "stats_schema_version": 1,
        ...     "config": {"min_cell_size": 20},
        ...     "dataset": {"dataset_version": "1.0"},
        ...     "layout": {"data_columns": {"code": "String", "subject_id": "Int64"}},
        ...     "counts": {"n_subjects": 100_000, "n_measurements": 5_000_000},
        ... }

        A document compared with itself passes, and reports how much it actually checked:

        >>> report = compare_summary_stats(base, deepcopy(base))
        >>> report.status, report.n_paths_checked, report.findings
        ('pass', 5, ())

        The default policy sets the subject count's warn threshold at zero, so *any* movement is
        surfaced -- but only movement past half a percent fails the run:

        >>> nudged = deepcopy(base); nudged["counts"]["n_subjects"] = 100_200
        >>> compare_summary_stats(base, nudged).status
        'warn'
        >>> dropped = deepcopy(base); dropped["counts"]["n_subjects"] = 98_000
        >>> report = compare_summary_stats(base, dropped)
        >>> report.status, report.findings[0].path, report.findings[0].relative_delta
        ('fail', 'counts.n_subjects', -0.02)

        A schema change fails regardless of magnitude, because there is no tolerable amount of it:

        >>> retyped = deepcopy(base); retyped["layout"]["data_columns"]["subject_id"] = "Int32"
        >>> report = compare_summary_stats(base, retyped)
        >>> report.status, report.findings[0].message
        ('fail', "changed from 'Int64' to 'Int32'")

        A path that disappears is an error; one that appears is a warning. The asymmetry is
        deliberate: a new field is usually a tool upgrade, while a vanished one means a statistic
        stopped being computed and is silently no longer checked. A rule may override either, for
        the paths where key churn is routine rather than alarming.

        >>> pruned = deepcopy(base); del pruned["counts"]["n_measurements"]
        >>> report = compare_summary_stats(base, pruned)
        >>> report.status, report.findings[0].kind, report.findings[0].message
        ('fail', 'removed', 'present in baseline but absent from current')
        >>> added = deepcopy(base); added["counts"]["n_notes"] = 17
        >>> report = compare_summary_stats(base, added)
        >>> report.status, report.findings[0].kind
        ('warn', 'added')

        The motivating workflow, end to end: the ETL changed *and* the dataset version moved, so the
        volume drift is expected. Under the default policy that still fails --

        >>> new_data = deepcopy(base)
        >>> new_data["dataset"]["dataset_version"] = "2.0"
        >>> new_data["counts"] = {"n_subjects": 140_000, "n_measurements": 7_000_000}
        >>> compare_summary_stats(base, new_data).status
        'fail'

        -- but under ``relax`` the volume findings drop to informational while the schema check stays
        live, which is exactly the distinction a version bump justifies:

        >>> relaxed_policy = ComparePolicy.from_dict({"on_dataset_version_change": "relax"})
        >>> report = compare_summary_stats(base, new_data, relaxed_policy)
        >>> report.status, report.relaxed, report.counts
        ('pass', True, {'error': 0, 'info': 3, 'warning': 0})
        >>> both = deepcopy(new_data); both["layout"]["data_columns"]["code"] = "Int64"
        >>> report = compare_summary_stats(base, both, relaxed_policy)
        >>> report.status, [f.path for f in report.findings if f.severity is Severity.ERROR]
        ('fail', ['layout.data_columns.code'])

        Documents extracted under different configs are refused outright rather than diffed into
        meaningless noise:

        >>> other_cfg = deepcopy(base); other_cfg["config"]["min_cell_size"] = 5
        >>> report = compare_summary_stats(base, other_cfg)
        >>> report.status, report.findings[0].kind
        ('fail', 'precondition')

        ``status`` describes the findings and ``failed`` applies the policy, so ``fail_on: warning``
        turns a warning-only run into a failing one without misreporting it as an error:

        >>> strict = ComparePolicy.from_dict({"fail_on": "warning"})
        >>> report = compare_summary_stats(base, added, strict)
        >>> report.status, report.failed
        ('warn', True)
        >>> report = compare_summary_stats(base, added)
        >>> report.status, report.failed
        ('warn', False)
    """
    policy = policy or ComparePolicy.default()
    findings: list[Finding] = list(_check_preconditions(baseline, current))

    version_finding = _dataset_version_finding(baseline, current, policy)
    base_version = (baseline.get("dataset") or {}).get("dataset_version")
    curr_version = (current.get("dataset") or {}).get("dataset_version")
    dataset_version_changed = base_version != curr_version
    relaxed = dataset_version_changed and (
        policy.on_dataset_version_change is DatasetVersionChangeAction.RELAX
    )
    if version_finding is not None:
        findings.append(version_finding)

    base_flat = flatten(baseline)
    curr_flat = flatten(current)
    n_checked = 0
    n_ignored = 0

    for path in sorted(set(base_flat) | set(curr_flat)):
        if path.startswith(_CONFIG_PREFIX) or path == "config":
            continue
        if path == "dataset.dataset_version":
            # Handled above by the dedicated dataset-version policy, which knows how to relax the
            # rest of the comparison; comparing it again here would double-report it.
            continue
        rule = policy.rule_for(path)
        if rule.kind is Kind.IGNORE:
            n_ignored += 1
            continue

        in_base, in_curr = path in base_flat, path in curr_flat
        finding: Finding | None
        if in_base and not in_curr:
            finding = Finding(
                path=path,
                severity=rule.on_removed or policy.on_removed_key,
                kind="removed",
                baseline=base_flat[path],
                message="present in baseline but absent from current",
            )
        elif in_curr and not in_base:
            finding = Finding(
                path=path,
                severity=rule.on_added or policy.on_added_key,
                kind="added",
                current=curr_flat[path],
                message="present in current but absent from baseline",
            )
        else:
            n_checked += 1
            finding = _compare_leaf(path, base_flat[path], curr_flat[path], rule)

        if finding is None:
            continue
        if relaxed and rule.is_relaxable:
            # A data change legitimately moves volume, and moves cells across the suppression floor,
            # so keys appearing and disappearing is expected here too. What stays hard is whatever
            # the policy marks as not relaxable: schema, dtypes, and vocabulary set identity.
            finding = dataclasses.replace(finding, severity=Severity.INFO)
        findings.append(finding)

    findings.sort(key=lambda f: (-f.severity._rank, f.path))
    worst = max((f.severity for f in findings), default=Severity.INFO)
    if any(f.severity is Severity.ERROR for f in findings):
        status = "fail"
    elif any(f.severity is Severity.WARNING for f in findings):
        status = "warn"
    else:
        status = "pass"

    return ComparisonReport(
        status=status,
        failed=bool(findings) and policy.fail_on is not None and worst >= policy.fail_on,
        fail_on="never" if policy.fail_on is None else str(policy.fail_on),
        findings=tuple(findings),
        n_paths_checked=n_checked,
        n_paths_ignored=n_ignored,
        dataset_version_changed=dataset_version_changed,
        relaxed=relaxed,
    )


__all__ = [
    "STATS_SCHEMA_VERSION",
    "ComparisonReport",
    "Finding",
    "compare_summary_stats",
    "relative_delta",
]
