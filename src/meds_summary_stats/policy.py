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

        Severity *strings* compare by seriousness too, which matters because this is a ``StrEnum``
        and callers naturally reach for the string form. Left to the base class these would compare
        alphabetically and answer wrongly -- ``"error" < "warning"``, so ``Severity.ERROR >
        "warning"`` would be ``False``:

        >>> Severity.ERROR > "warning", Severity.INFO > "warning"
        (True, False)
        >>> Severity.WARNING >= "warning", Severity.WARNING <= "info"
        (True, False)

        A string that is not a severity at all is a mistake, and says so:

        >>> Severity.ERROR > "banana"
        Traceback (most recent call last):
            ...
        ValueError: 'banana' is not a valid Severity

        As is comparing against a non-string:

        >>> Severity.ERROR < 3
        Traceback (most recent call last):
            ...
        TypeError: '<' not supported between instances of 'Severity' and 'int'
        >>> Severity.ERROR >= None
        Traceback (most recent call last):
            ...
        TypeError: '>=' not supported between instances of 'Severity' and 'NoneType'
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def _rank(self) -> int:
        return ["info", "warning", "error"].index(self.value)

    @staticmethod
    def _rank_of(other: object) -> int | None:
        """Rank whatever we are being compared against, or ``None`` if it is not a severity.

        Severity strings are accepted and ranked by seriousness. Without this, the ``StrEnum`` base
        would quietly answer with an *alphabetical* comparison -- ``Severity.ERROR > "warning"``
        would be ``False``, because ``"e" < "w"`` -- which is a wrong answer rather than an error.
        """
        if isinstance(other, Severity):
            return other._rank
        if isinstance(other, str):
            return Severity(other)._rank
        return None

    def __lt__(self, other: object) -> bool:
        """Order by seriousness, so that ``max()`` over findings yields the worst one."""
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self._rank < rank

    def __gt__(self, other: object) -> bool:
        """Order by seriousness."""
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self._rank > rank

    def __le__(self, other: object) -> bool:
        """Order by seriousness."""
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self._rank <= rank

    def __ge__(self, other: object) -> bool:
        """Order by seriousness."""
        rank = self._rank_of(other)
        return NotImplemented if rank is None else self._rank >= rank


class Kind(enum.StrEnum):
    """How a path's two values should be compared.

    Attributes:
        EXACT: Any inequality is a finding. For schema, dtypes, names, and digests.
        RELATIVE: Flag when ``|new - old| / max(|old|, 1)`` exceeds a threshold.
        ABSOLUTE: Flag when ``|new - old|`` exceeds a threshold. For timestamps, such as
            ``time.quantiles.*``, the difference is measured in days.
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
        on_added: Severity when this path appears in the current document but not the baseline.
            ``None`` defers to the policy's ``on_added_key``.
        on_removed: Severity when this path is in the baseline but not the current document.
            ``None`` defers to the policy's ``on_removed_key``. Override it where key churn is
            routine rather than alarming -- per-shard file digests come and go whenever an ETL
            changes its parallelism, which is not the same event as a statistic disappearing.
        relaxable: Whether a dataset-version change excuses this finding, under
            ``on_dataset_version_change: relax``. ``None`` means "decide by kind": numeric
            comparisons are excused, ``exact`` ones are not. Set it explicitly for the cases where
            that default is wrong -- the thresholded vocabulary digests are ``exact``, but their
            membership depends on absolute subject counts, so a cohort that simply got larger
            changes them without the vocabulary having changed at all.

    Examples:
        >>> Rule(path="counts.*", kind=Kind.RELATIVE, error=0.01)
        Rule(path='counts.*', kind=<Kind.RELATIVE: 'relative'>, warn=None, error=0.01, ...)

        By default, numeric findings are excused by a data refresh and exact ones are not:

        >>> Rule(path="counts.*", kind=Kind.RELATIVE, error=0.01).is_relaxable
        True
        >>> Rule(path="layout.*", kind=Kind.EXACT).is_relaxable
        False

        An explicit setting wins either way:

        >>> Rule(path="codes.digests.thresholded.*", kind=Kind.EXACT, relaxable=True).is_relaxable
        True
        >>> Rule(path="counts.n_subjects", kind=Kind.RELATIVE, error=0.01, relaxable=False).is_relaxable
        False

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
    on_added: Severity | None = None
    on_removed: Severity | None = None
    relaxable: bool | None = None

    @property
    def is_relaxable(self) -> bool:
        """Whether a dataset-version change excuses findings from this rule."""
        if self.relaxable is not None:
            return self.relaxable
        return self.kind is not Kind.EXACT

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

            A misspelled key is rejected rather than ignored -- a silently dropped ``errors`` would
            leave the rule unable to fire while looking correct:

            >>> Rule.from_dict({"path": "a.*", "kind": "relative", "errors": 0.1})
            Traceback (most recent call last):
                ...
            ValueError: Unknown rule keys for path 'a.*': ['errors']
        """
        if "path" not in raw:
            raise ValueError("Every rule needs a 'path'")
        known_keys = {"path", "kind", "warn", "error", "on_change", "on_added", "on_removed", "relaxable"}
        if unknown := sorted(set(raw) - known_keys):
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
            on_added=Severity(raw["on_added"]) if "on_added" in raw else None,
            on_removed=Severity(raw["on_removed"]) if "on_removed" in raw else None,
            relaxable=raw.get("relaxable"),
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
    # The dataset checksum. Deliberately a warning rather than an error: it is a digest of bytes,
    # so the same rows rewritten by a different parquet writer move it without the data having
    # changed. When that happens the digest moves and every statistic stays put, which is itself
    # the diagnosis. Promote it to `error` once you have confirmed your ETL is byte-reproducible.
    {"path": "dataset_digest.**", "kind": "exact", "on_change": "warning"},
    {"path": "dataset_digest.method", "kind": "exact"},
    # Per-shard digests appear and disappear whenever an ETL changes its parallelism, which is a
    # layout event rather than a data one -- `layout` reports it, and reporting it twice at error
    # severity would just be noise.
    {
        "path": "dataset_digest.by_file.*",
        "kind": "exact",
        "on_change": "warning",
        "on_added": "warning",
        "on_removed": "warning",
    },
    # dataset.json carries created_at and etl_version, so its digest changes on every rerun.
    {"path": "dataset_digest.dataset_metadata", "kind": "ignore"},
    # Identity and schema: any change is a regression until proven otherwise.
    {"path": "stats_schema_version", "kind": "exact"},
    {"path": "dataset.**", "kind": "exact"},
    {"path": "layout.**", "kind": "exact"},
    {"path": "layout.shard_names", "kind": "exact", "on_change": "warning"},
    # Headline volume. The subject count is the tightest, since it is the number an ETL change is
    # least entitled to move.
    {"path": "**", "kind": "relative", "warn": 0.001, "error": 0.01},
    {"path": "counts.n_subjects", "kind": "relative", "warn": 0.0, "error": 0.005},
    {"path": "counts.n_measurements", "kind": "relative", "warn": 0.001, "error": 0.005},
    {"path": "counts.n_events", "kind": "relative", "warn": 0.001, "error": 0.005},
    {"path": "counts.quality.**", "kind": "absolute", "warn": 0, "error": 0},
    # Distributions drift more freely than counts do.
    {"path": "**.quantiles.*", "kind": "relative", "warn": 0.005, "error": 0.02},
    # Timestamp quantiles, in days: a relative threshold on a timestamp would be relative to the
    # epoch. The same data gives identical timestamps, so any movement warns -- a sub-day shift is
    # what a timezone regression looks like -- and a move of more than a day is an error.
    {"path": "time.quantiles.*", "kind": "absolute", "warn": 0.0, "error": 1.0},
    {"path": "**.mean", "kind": "relative", "warn": 0.005, "error": 0.02},
    {"path": "**.std", "kind": "relative", "warn": 0.01, "error": 0.05},
    # Vocabulary. A changed set of codes is an error; a changed *ordering* of codes is a warning,
    # because frequency ranks reshuffle on any small count change and would otherwise fire constantly.
    {"path": "codes.digests.alphabetical", "kind": "exact"},
    {"path": "codes.digests.by_measurement_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.by_subject_frequency", "kind": "exact", "on_change": "warning"},
    {"path": "codes.digests.numeric_moments.digest", "kind": "exact", "on_change": "warning"},
    # Thresholded digests cover only the codes above a subject floor, so their membership moves
    # with cohort size. They are hard checks when the data is held fixed, but a genuine data refresh
    # changes them without the vocabulary having changed -- hence `relaxable`.
    {"path": "codes.digests.thresholded.*.alphabetical", "kind": "exact", "relaxable": True},
    {
        "path": "codes.digests.thresholded.*.by_measurement_frequency",
        "kind": "exact",
        "on_change": "warning",
        "relaxable": True,
    },
    {
        "path": "codes.digests.thresholded.*.by_subject_frequency",
        "kind": "exact",
        "on_change": "warning",
        "relaxable": True,
    },
    {"path": "codes.digests.numeric_moments.n_codes", "kind": "relative", "warn": 0.0, "error": 0.05},
    {"path": "code_metadata.vocabulary_digest", "kind": "exact"},
    {"path": "code_metadata.ontology.by_vocabulary.*.keys_digest", "kind": "exact"},
    # `keys` is only emitted under `emit_ontology_keys`, and then it explains a `keys_digest` change
    # rather than being a second check of it. The digest carries the error in both modes; left to
    # the catch-all, the same changed key would be reported as two errors.
    {"path": "code_metadata.ontology.by_vocabulary.*.keys", "kind": "exact", "on_change": "info"},
    {"path": "code_metadata.ontology.by_vocabulary.*.recognized", "kind": "exact"},
    # Ontology coverage is the axis that says *where* the vocabulary moved, and a mapping table
    # that stops resolving shows up here and nowhere else.
    {
        "path": "code_metadata.ontology.coverage.frac_measurements_mapped",
        "kind": "relative",
        "warn": 0.0,
        "error": 0.01,
    },
    {
        "path": "code_metadata.ontology.coverage.n_observed_codes_without_parents",
        "kind": "relative",
        "warn": 0.0,
        "error": 0.05,
    },
    {"path": "code_metadata.ontology.n_vocabularies", "kind": "exact"},
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
