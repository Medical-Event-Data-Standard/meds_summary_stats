"""The comparison policy: which drift matters, and how much of it is too much.

A policy is a list of glob-matched rules over flattened stats paths. Every leaf gets exactly one rule
-- the most specific one that matches it -- so there is never ambiguity about why a given number was
or was not flagged.

The shipped default policy encodes the motivating use case: with the dataset version held fixed, an
ETL revision must not move the subject count by more than half a percent, must not change the data
schema at all, and must not silently rewrite the code vocabulary.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

import yaml

from .flatten import matches, specificity


class Severity(enum.StrEnum):
    """How seriously to take a difference.

    Examples:
        >>> Severity("error") > Severity("warning") > Severity("info")
        True
        >>> sorted([Severity.ERROR, Severity.INFO, Severity.WARNING])
        [<Severity.INFO: 'info'>, <Severity.WARNING: 'warning'>, <Severity.ERROR: 'error'>]
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def _rank(self) -> int:
        return ["info", "warning", "error"].index(self.value)

    def __lt__(self, other: object) -> bool:
        """Order by seriousness, so that ``max()`` over findings yields the worst one."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self._rank < other._rank

    def __gt__(self, other: object) -> bool:
        """Order by seriousness."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self._rank > other._rank

    def __le__(self, other: object) -> bool:
        """Order by seriousness."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self._rank <= other._rank

    def __ge__(self, other: object) -> bool:
        """Order by seriousness."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self._rank >= other._rank


class Kind(enum.StrEnum):
    """How a path's two values should be compared.

    Attributes:
        EXACT: Any inequality is a finding. For schema, dtypes, names, and digests.
        RELATIVE: Flag when ``|new - old| / max(|old|, 1)`` exceeds a threshold.
        ABSOLUTE: Flag when ``|new - old|`` exceeds a threshold.
        IGNORE: Never compared.
    """

    EXACT = "exact"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    IGNORE = "ignore"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Rule:
    """One glob-matched comparison rule.

    Attributes:
        path: Glob over flattened stats paths. ``*`` matches one segment, ``**`` matches many.
        kind: How to compare.
        warn: Threshold above which a difference is a warning. ``None`` disables the warning tier.
        error: Threshold above which a difference is an error. ``None`` disables the error tier.
        on_change: Severity for an ``exact`` mismatch.

    Examples:
        >>> Rule(path="counts.*", kind=Kind.RELATIVE, error=0.01)
        Rule(path='counts.*', kind=<Kind.RELATIVE: 'relative'>, warn=None, error=0.01, ...)

        A numeric rule with no thresholds could never fire, which is almost always a config typo:

        >>> Rule(path="counts.*", kind=Kind.RELATIVE)
        Traceback (most recent call last):
            ...
        ValueError: Rule for 'counts.*' is 'relative' but sets neither warn nor error

        Thresholds must be ordered, or the warning tier would be unreachable:

        >>> Rule(path="counts.*", kind=Kind.RELATIVE, warn=0.5, error=0.1)
        Traceback (most recent call last):
            ...
        ValueError: Rule for 'counts.*' has warn=0.5 above error=0.1
    """

    path: str
    kind: Kind
    warn: float | None = None
    error: float | None = None
    on_change: Severity = Severity.ERROR

    def __post_init__(self) -> None:
        """Reject rules that cannot fire or whose tiers are inverted."""
        if self.kind in (Kind.RELATIVE, Kind.ABSOLUTE) and self.warn is None and self.error is None:
            raise ValueError(f"Rule for {self.path!r} is {self.kind.value!r} but sets neither warn nor error")
        if self.warn is not None and self.error is not None and self.warn > self.error:
            raise ValueError(f"Rule for {self.path!r} has warn={self.warn} above error={self.error}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Rule:
        """Build a rule from its YAML mapping form.

        Examples:
            >>> Rule.from_dict({"path": "a.*", "kind": "exact", "on_change": "warning"}).on_change
            <Severity.WARNING: 'warning'>

            >>> Rule.from_dict({"path": "a.*", "kind": "approximate", "error": 1})
            Traceback (most recent call last):
                ...
            ValueError: Unknown comparison kind 'approximate' for path 'a.*'

            >>> Rule.from_dict({"kind": "ignore"})
            Traceback (most recent call last):
                ...
            ValueError: Every rule needs a 'path'
        """
        if "path" not in raw:
            raise ValueError("Every rule needs a 'path'")
        if unknown := sorted(set(raw) - {"path", "kind", "warn", "error", "on_change"}):
            raise ValueError(f"Unknown rule keys for path {raw['path']!r}: {unknown}")
        try:
            kind = Kind(raw.get("kind", "exact"))
        except ValueError:
            raise ValueError(
                f"Unknown comparison kind {raw.get('kind')!r} for path {raw['path']!r}"
            ) from None
        return cls(
            path=raw["path"],
            kind=kind,
            warn=raw.get("warn"),
            error=raw.get("error"),
            on_change=Severity(raw.get("on_change", "error")),
        )


class DatasetVersionChangeAction(enum.StrEnum):
    """What to do when the two documents describe different dataset versions.

    ``RELAX`` is the interesting one and the reason this setting exists: the whole premise of the
    check is "same data, new ETL". When the data itself moved, volume drift is expected and flagging
    it is noise -- but schema and vocabulary regressions are still real bugs, so those stay hard.

    Attributes:
        IGNORE: Say nothing.
        WARN: Note it as a warning; leave every other rule untouched.
        ERROR: Treat a version change as a failure in itself.
        RELAX: Downgrade every finding except ``exact`` mismatches to ``info`` -- including keys
            that appeared or disappeared, since a data change moves cells across the suppression
            floor. Schema and vocabulary identity stay at full severity.
    """

    IGNORE = "ignore"
    WARN = "warn"
    ERROR = "error"
    RELAX = "relax"


#: The default rule set, ordered from general to specific for readability only -- match order is by
#: specificity, not by position.
DEFAULT_RULES: tuple[dict[str, Any], ...] = (
    # Provenance that legitimately changes on every run.
    {"path": "extracted_at", "kind": "ignore"},
    {"path": "meds_summary_stats_version", "kind": "ignore"},
    {"path": "dataset.created_at", "kind": "ignore"},
    {"path": "dataset.etl_version", "kind": "ignore"},
    {"path": "dataset.location_uri", "kind": "ignore"},
    # Identity and schema: any change is a regression until proven otherwise.
    {"path": "stats_schema_version", "kind": "exact"},
    {"path": "dataset.**", "kind": "exact"},
    {"path": "layout.**", "kind": "exact"},
    {"path": "layout.shard_names.**", "kind": "exact", "on_change": "warning"},
    # Headline volume. The subject count is the tightest, since it is the number an ETL change is
    # least entitled to move.
    {"path": "**", "kind": "relative", "warn": 0.001, "error": 0.01},
    {"path": "counts.n_subjects", "kind": "relative", "warn": 0.0, "error": 0.005},
    {"path": "counts.n_measurements", "kind": "relative", "warn": 0.001, "error": 0.005},
    {"path": "counts.n_events", "kind": "relative", "warn": 0.001, "error": 0.005},
    {"path": "counts.quality.**", "kind": "absolute", "warn": 0, "error": 0},
    # Distributions drift more freely than counts do.
    {"path": "**.quantiles.*", "kind": "relative", "warn": 0.005, "error": 0.02},
    {"path": "**.mean", "kind": "relative", "warn": 0.005, "error": 0.02},
    {"path": "**.std", "kind": "relative", "warn": 0.01, "error": 0.05},
    # Vocabulary. A changed set of codes is an error; a changed *ordering* of codes is a warning,
    # because frequency ranks reshuffle on any small count change and would otherwise fire constantly.
    {"path": "codes.digests.alphabetical", "kind": "exact"},
    {"path": "codes.digests.by_measurement_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.by_subject_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.numeric_moments.digest", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.thresholded.*.alphabetical", "kind": "exact"},
    {"path": "codes.digests.thresholded.*.by_measurement_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.thresholded.*.by_subject_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "code_metadata.vocabulary_digest", "kind": "exact"},
    {"path": "code_metadata.ontology.by_vocabulary.*.keys_digest", "kind": "exact"},
    {"path": "code_metadata.ontology.by_vocabulary.*.recognized", "kind": "exact"},
    # Referential integrity between the data and its metadata.
    {"path": "code_metadata.n_data_codes_missing_from_metadata", "kind": "absolute", "error": 0},
    {"path": "splits.n_subjects_in_data_without_split", "kind": "absolute", "error": 0},
    {"path": "splits.n_subjects_in_splits_without_data", "kind": "absolute", "error": 0},
    # Suppression rollups move whenever a cell crosses the floor; that is expected, not a defect.
    {"path": "**.<suppressed>.**", "kind": "relative", "warn": 0.05, "error": 0.25},
)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ComparePolicy:
    """A complete comparison policy.

    Attributes:
        rules: The rule set, matched by specificity.
        on_added_key: Severity for a path present in the current document but not the baseline.
        on_removed_key: Severity for a path present in the baseline but not the current document.
        on_dataset_version_change: What to do when ``dataset.dataset_version`` differs.
        fail_on: Minimum severity that makes the overall comparison fail, or ``None`` to report
            findings without ever failing -- useful when first rolling the check out on an existing
            repo, where the aim is to see the noise before enforcing on it.

    Examples:
        >>> policy = ComparePolicy.default()
        >>> policy.on_dataset_version_change
        <DatasetVersionChangeAction.WARN: 'warn'>
        >>> policy.fail_on
        <Severity.ERROR: 'error'>

        Rule lookup picks the most specific match, not the first or last one written:

        >>> policy.rule_for("counts.n_subjects").error
        0.005
        >>> policy.rule_for("counts.n_unique_codes").error
        0.01
        >>> policy.rule_for("layout.data_columns.code").kind
        <Kind.EXACT: 'exact'>
        >>> policy.rule_for("extracted_at").kind
        <Kind.IGNORE: 'ignore'>

        Every conceivable path resolves to something, because the default set contains a ``**``
        catch-all -- a path with no rule would be silently unchecked, which is the one failure mode a
        regression checker must not have:

        >>> policy.rule_for("some.path.nobody.anticipated").kind
        <Kind.RELATIVE: 'relative'>
    """

    rules: tuple[Rule, ...]
    on_added_key: Severity = Severity.WARNING
    on_removed_key: Severity = Severity.ERROR
    on_dataset_version_change: DatasetVersionChangeAction = DatasetVersionChangeAction.WARN
    fail_on: Severity | None = Severity.ERROR

    @classmethod
    def default(cls) -> ComparePolicy:
        """The built-in policy.

        Examples:
            >>> len(ComparePolicy.default().rules) == len(DEFAULT_RULES)
            True
        """
        return cls(rules=tuple(Rule.from_dict(r) for r in DEFAULT_RULES))

    def rule_for(self, path: str) -> Rule:
        """Return the most specific rule matching ``path``.

        Args:
            path: A flattened stats path.

        Returns:
            The winning rule.

        Raises:
            LookupError: If no rule matches. A policy without a catch-all can leave paths unchecked,
                so this is surfaced rather than defaulted.

        Examples:
            >>> narrow = ComparePolicy(rules=(Rule(path="counts.*", kind=Kind.EXACT),))
            >>> narrow.rule_for("counts.n_subjects").kind
            <Kind.EXACT: 'exact'>
            >>> narrow.rule_for("layout.n_data_shards")
            Traceback (most recent call last):
                ...
            LookupError: No policy rule matches path 'layout.n_data_shards'
        """
        best: Rule | None = None
        best_key: tuple[int, int, int] | None = None
        for rule in self.rules:
            if not matches(path, rule.path):
                continue
            key = specificity(rule.path)
            # `>=` so that a later rule of equal specificity wins: user rules are appended to the
            # defaults, and an override must beat the default it is written to replace.
            if best_key is None or key >= best_key:
                best, best_key = rule, key
        if best is None:
            raise LookupError(f"No policy rule matches path {path!r}")
        return best

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ComparePolicy:
        """Build a policy from its YAML mapping form.

        User rules are appended to the defaults rather than replacing them, so a config only has to
        state its overrides. Set ``extends: none`` to start from an empty rule set instead.

        Examples:
            >>> policy = ComparePolicy.from_dict({
            ...     "fail_on": "warning",
            ...     "rules": [{"path": "counts.n_subjects", "kind": "relative", "error": 0.5}],
            ... })
            >>> policy.fail_on
            <Severity.WARNING: 'warning'>

            The override wins over the default of the same specificity, because later rules break
            specificity ties:

            >>> policy.rule_for("counts.n_subjects").error
            0.5

            Unrelated defaults are still in force:

            >>> policy.rule_for("layout.data_columns.code").kind
            <Kind.EXACT: 'exact'>

            With ``extends: none`` only the stated rules apply:

            >>> bare = ComparePolicy.from_dict({"extends": "none", "rules": [
            ...     {"path": "counts.n_subjects", "kind": "exact"}]})
            >>> len(bare.rules)
            1

            ``never`` is spelled out rather than left to the caller to fake with an impossible
            threshold:

            >>> ComparePolicy.from_dict({"fail_on": "never"}).fail_on is None
            True

            >>> ComparePolicy.from_dict({"fail_on": "catastrophe"})
            Traceback (most recent call last):
                ...
            ValueError: 'catastrophe' is not a valid Severity

            >>> ComparePolicy.from_dict({"typo_key": 1})
            Traceback (most recent call last):
                ...
            ValueError: Unknown policy keys: ['typo_key']
        """
        known = {"extends", "rules", "on_added_key", "on_removed_key", "on_dataset_version_change", "fail_on"}
        if unknown := sorted(set(raw) - known):
            raise ValueError(f"Unknown policy keys: {unknown}")

        base = () if raw.get("extends") == "none" else tuple(Rule.from_dict(r) for r in DEFAULT_RULES)
        user = tuple(Rule.from_dict(r) for r in raw.get("rules", []))
        return cls(
            rules=base + user,
            on_added_key=Severity(raw.get("on_added_key", "warning")),
            on_removed_key=Severity(raw.get("on_removed_key", "error")),
            on_dataset_version_change=DatasetVersionChangeAction(
                raw.get("on_dataset_version_change", "warn")
            ),
            fail_on=None if raw.get("fail_on") == "never" else Severity(raw.get("fail_on", "error")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ComparePolicy:
        """Load a policy from a YAML file.

        Examples:
            >>> spec = {"policy.yaml": {
            ...     "on_dataset_version_change": "relax",
            ...     "rules": [{"path": "counts.n_subjects", "kind": "relative", "error": 0.2}],
            ... }}
            >>> with yaml_disk(spec) as root:
            ...     policy = ComparePolicy.from_yaml(root / "policy.yaml")
            >>> policy.on_dataset_version_change, policy.rule_for("counts.n_subjects").error
            (<DatasetVersionChangeAction.RELAX: 'relax'>, 0.2)
        """
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
