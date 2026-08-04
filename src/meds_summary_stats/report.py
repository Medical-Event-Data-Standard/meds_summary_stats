"""Human-readable rendering of a comparison report.

Two renderings are provided: a plain-text table for a terminal, and GitHub-flavored markdown for a
CI job summary. Both are deliberately lossy -- they truncate to the findings a person will act on.
The full detail lives in the JSON report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .policy import Severity

if TYPE_CHECKING:
    from .compare import ComparisonReport, Finding

#: Maximum findings rendered per severity before the rest are summarized as a count.
DEFAULT_MAX_PER_SEVERITY = 20

_STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


def _truncate(value: object, width: int = 28) -> str:
    """Render a value for a fixed-width table cell.

    Examples:
        >>> _truncate(100000), _truncate(None), _truncate(True)
        ('100000', '-', 'True')
        >>> _truncate("sha256:" + "a" * 64)
        'sha256:aaaaaaaaaaaaaaaaaaaa…'
        >>> _truncate(0.123456789)
        '0.123456789'
    """
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _grouped(report: ComparisonReport, max_per_severity: int) -> list[tuple[Severity, list[Finding], int]]:
    """Group findings by descending severity, capping each group.

    Returns:
        Triples of ``(severity, shown findings, number omitted)``, skipping empty groups.

    Examples:
        >>> from .compare import ComparisonReport, Finding
        >>> findings = tuple(
        ...     Finding(path=f"p{i}", severity=Severity.WARNING, kind="exact") for i in range(5)
        ... )
        >>> report = ComparisonReport(status="warn", failed=False, fail_on="error", findings=findings,
        ...                           n_paths_checked=5, n_paths_ignored=0,
        ...                           dataset_version_changed=False, relaxed=False)
        >>> [(str(s), len(shown), omitted) for s, shown, omitted in _grouped(report, 3)]
        [('warning', 3, 2)]
        >>> [(str(s), len(shown), omitted) for s, shown, omitted in _grouped(report, 99)]
        [('warning', 5, 0)]
    """
    out = []
    for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
        group = [f for f in report.findings if f.severity is severity]
        if group:
            out.append((severity, group[:max_per_severity], max(0, len(group) - max_per_severity)))
    return out


def _verdict_note(report: ComparisonReport) -> str | None:
    """Explain the exit code when it disagrees with the finding severities.

    ``status`` describes what was found and ``failed`` applies the policy; when the two diverge a
    reader would otherwise be left staring at a red X above a list of warnings, or a green check
    above a list of errors.

    Examples:
        >>> from .compare import ComparisonReport
        >>> def report(status, failed, fail_on):
        ...     return ComparisonReport(status=status, failed=failed, fail_on=fail_on, findings=(),
        ...                             n_paths_checked=1, n_paths_ignored=0,
        ...                             dataset_version_changed=False, relaxed=False)
        >>> _verdict_note(report("fail", True, "error")) is None
        True
        >>> _verdict_note(report("warn", False, "error")) is None
        True
        >>> _verdict_note(report("warn", True, "warning"))
        'Failing the run because fail_on=warning.'
        >>> _verdict_note(report("fail", False, "never"))
        'Errors were found, but fail_on=never, so the run is not failed.'
    """
    if report.failed and report.status != "fail":
        return f"Failing the run because fail_on={report.fail_on}."
    if not report.failed and report.status == "fail":
        return f"Errors were found, but fail_on={report.fail_on}, so the run is not failed."
    return None


def render_text(report: ComparisonReport, max_per_severity: int = DEFAULT_MAX_PER_SEVERITY) -> str:
    """Render a report as a plain-text summary.

    Args:
        report: The report to render.
        max_per_severity: Findings shown per severity before truncation.

    Returns:
        The rendered text, without a trailing newline.

    Examples:
        >>> from .compare import ComparisonReport, Finding
        >>> clean = ComparisonReport(status="pass", failed=False, fail_on="error", findings=(),
        ...                          n_paths_checked=418, n_paths_ignored=5,
        ...                          dataset_version_changed=False, relaxed=False)
        >>> print(render_text(clean))
        PASS - 418 paths checked, 5 ignored, no findings.

        >>> failed = ComparisonReport(
        ...     status="fail", failed=True, fail_on="error", n_paths_checked=418, n_paths_ignored=5,
        ...     dataset_version_changed=False, relaxed=False,
        ...     findings=(
        ...         Finding(path="counts.n_subjects", severity=Severity.ERROR, kind="relative",
        ...                 baseline=100000, current=98000,
        ...                 message="100000 -> 98000 (-2.00%), over relative threshold 0.005"),
        ...         Finding(path="codes.digests.by_subject_frequency", severity=Severity.WARNING,
        ...                 kind="exact", baseline="sha256:aaa", current="sha256:bbb",
        ...                 message="changed from 'sha256:aaa' to 'sha256:bbb'"),
        ...     ))
        >>> print(render_text(failed))
        FAIL - 418 paths checked, 5 ignored; 1 error, 1 warning.
        <BLANKLINE>
        ERROR (1)
          counts.n_subjects
              100000 -> 98000 (-2.00%), over relative threshold 0.005
        <BLANKLINE>
        WARNING (1)
          codes.digests.by_subject_frequency
              changed from 'sha256:aaa' to 'sha256:bbb'

        A relaxed run says so, since a reader would otherwise wonder why large drifts passed:

        >>> relaxed = ComparisonReport(status="pass", failed=False, fail_on="error", findings=(),
        ...                            n_paths_checked=10, n_paths_ignored=0,
        ...                            dataset_version_changed=True, relaxed=True)
        >>> print(render_text(relaxed))
        PASS - 10 paths checked, 0 ignored, no findings.
        Note: dataset_version changed; numeric findings were downgraded to informational.
    """
    counts = report.counts
    header = (
        f"{report.status.upper()} - {report.n_paths_checked} paths checked, {report.n_paths_ignored} ignored"
    )
    if not report.findings:
        lines = [f"{header}, no findings."]
    else:
        tally = ", ".join(
            f"{counts[str(s)]} {s!s}"
            for s in (Severity.ERROR, Severity.WARNING, Severity.INFO)
            if counts[str(s)]
        )
        lines = [f"{header}; {tally}.", ""]
        for severity, shown, omitted in _grouped(report, max_per_severity):
            lines.append(f"{str(severity).upper()} ({counts[str(severity)]})")
            for finding in shown:
                lines.append(f"  {finding.path}")
                lines.append(f"      {finding.message}")
            if omitted:
                lines.append(f"  ... and {omitted} more")
            lines.append("")
        lines.pop()

    if report.relaxed:
        lines.append("Note: dataset_version changed; numeric findings were downgraded to informational.")
    if (note := _verdict_note(report)) is not None:
        lines.append(note)
    return "\n".join(lines)


def render_markdown(report: ComparisonReport, max_per_severity: int = DEFAULT_MAX_PER_SEVERITY) -> str:
    """Render a report as GitHub-flavored markdown, suitable for a job summary.

    Args:
        report: The report to render.
        max_per_severity: Findings shown per severity before truncation.

    Returns:
        The rendered markdown, ending in a newline.

    Examples:
        >>> from .compare import ComparisonReport, Finding
        >>> clean = ComparisonReport(status="pass", failed=False, fail_on="error", findings=(),
        ...                          n_paths_checked=418, n_paths_ignored=5,
        ...                          dataset_version_changed=False, relaxed=False)
        >>> print(render_markdown(clean), end="")
        ## ✅ MEDS summary-stats comparison: pass
        <BLANKLINE>
        418 paths checked, 5 ignored. No findings.

        >>> failed = ComparisonReport(
        ...     status="fail", failed=True, fail_on="error", n_paths_checked=2, n_paths_ignored=0,
        ...     dataset_version_changed=False, relaxed=False,
        ...     findings=(Finding(path="counts.n_subjects", severity=Severity.ERROR, kind="relative",
        ...                       baseline=100000, current=98000, threshold=0.005,
        ...                       message="over threshold"),))
        >>> print(render_markdown(failed), end="")
        ## ❌ MEDS summary-stats comparison: fail
        <BLANKLINE>
        2 paths checked, 0 ignored. **1 error**, 0 warnings, 0 info.
        <BLANKLINE>
        ### Errors (1)
        <BLANKLINE>
        | Path | Baseline | Current | Detail |
        | --- | --- | --- | --- |
        | `counts.n_subjects` | 100000 | 98000 | over threshold |
    """
    counts = report.counts
    icon = _STATUS_ICON.get(report.status, "❔")
    lines = [f"## {icon} MEDS summary-stats comparison: {report.status}", ""]

    if not report.findings:
        lines.append(
            f"{report.n_paths_checked} paths checked, {report.n_paths_ignored} ignored. No findings."
        )
    else:
        error_text = f"**{counts['error']} error{'s' if counts['error'] != 1 else ''}**"
        lines.append(
            f"{report.n_paths_checked} paths checked, {report.n_paths_ignored} ignored. "
            f"{error_text}, {counts['warning']} warnings, {counts['info']} info."
        )
        for severity, shown, omitted in _grouped(report, max_per_severity):
            lines += [
                "",
                f"### {str(severity).capitalize()}s ({counts[str(severity)]})",
                "",
                "| Path | Baseline | Current | Detail |",
                "| --- | --- | --- | --- |",
            ]
            for f in shown:
                lines.append(
                    f"| `{f.path}` | {_truncate(f.baseline)} | {_truncate(f.current)} | {f.message} |"
                )
            if omitted:
                lines.append(f"| _… and {omitted} more_ | | | |")

    if report.relaxed:
        lines += [
            "",
            "> [!NOTE]",
            "> `dataset_version` changed, so numeric findings were downgraded to informational.",
        ]
    if (note := _verdict_note(report)) is not None:
        lines += ["", "> [!NOTE]", f"> {note}"]
    return "\n".join(lines) + "\n"
