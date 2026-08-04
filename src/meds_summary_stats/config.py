"""Extraction configuration and the privacy guarantees it encodes.

Every knob that can affect the emitted numbers lives here, and the resolved config is written into the stats
file itself. Comparison then refuses to trust a diff between two files extracted under different configs,
because (for example) digests taken at different subject thresholds are simply different quantities and their
inequality means nothing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

#: Bucket label for parent codes that carry no ``/`` vocabulary separator.
NO_VOCABULARY = "<no-vocabulary>"

#: Bucket label for entries rolled up because they fell below the suppression threshold.
SUPPRESSED = "<suppressed>"

#: Quantiles reported for every distributional summary.
DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

#: Subject-count floors at which additional vocabulary digests are taken.
DEFAULT_DIGEST_THRESHOLDS: tuple[int, ...] = (20, 100, 1000)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ExtractConfig:
    """Controls what extraction computes and how aggressively it suppresses.

    Attributes:
        min_cell_size: Minimum number of *distinct subjects* backing a cell for that cell to be reported
            at all. Namespace, split, calendar-year, and ontology-vocabulary entries below this are
            dropped and rolled into a ``<suppressed>`` aggregate.
        min_numeric_cell_size: Minimum number of finite numeric values required before numeric moments
            are reported for a cell.
        numeric_detail: ``"moments"`` reports counts, mean, and standard deviation only.
            ``"quantiles"`` additionally computes full-column quantiles over ``numeric_value`` and
            ``time``, which requires materializing those columns and is markedly more expensive on
            large datasets.
        quantiles: The quantiles reported for distributional summaries.
        digest_subject_thresholds: Subject-count floors at which extra vocabulary digests are taken. A
            digest restricted to common codes is far less noisy than one over the full tail.
        moment_digest_sig_digits: Significant digits retained when hashing per-code numeric moments.
        code_metadata_only_digests: Whether to also digest the vocabulary of ``metadata/codes.parquet``
            independently of the codes actually observed in the data.
        emit_ontology_keys: Whether to list, in full, the keys observed within each external
            vocabulary referenced by ``parent_codes``. Those keys name published terminology concepts
            rather than anything about a subject, so this is off by default for output size, not for
            privacy. Suppression of low-subject vocabularies still applies.

    Examples:
        The defaults are deliberately conservative:

        >>> cfg = ExtractConfig()
        >>> cfg.min_cell_size, cfg.min_numeric_cell_size, cfg.numeric_detail
        (20, 20, 'moments')

        Configs are frozen, so nothing downstream can weaken a suppression threshold in flight:

        >>> cfg.min_cell_size = 1
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'min_cell_size'

        Thresholds below 1 are rejected outright rather than silently clamped:

        >>> ExtractConfig(min_cell_size=0)
        Traceback (most recent call last):
            ...
        ValueError: min_cell_size must be >= 1; got 0

        As is an unknown detail level:

        >>> ExtractConfig(numeric_detail="everything")
        Traceback (most recent call last):
            ...
        ValueError: numeric_detail must be one of ('moments', 'quantiles'); got 'everything'

        And out-of-range quantiles:

        >>> ExtractConfig(quantiles=(0.5, 1.5))
        Traceback (most recent call last):
            ...
        ValueError: quantiles must lie in (0, 1); got 1.5
    """

    min_cell_size: int = 20
    min_numeric_cell_size: int = 20
    numeric_detail: str = "moments"
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    digest_subject_thresholds: tuple[int, ...] = DEFAULT_DIGEST_THRESHOLDS
    moment_digest_sig_digits: int = 6
    code_metadata_only_digests: bool = True
    emit_ontology_keys: bool = False

    VALID_NUMERIC_DETAIL: tuple[str, ...] = dataclasses.field(
        default=("moments", "quantiles"), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the configuration, normalizing sequence fields to tuples."""
        for field in ("min_cell_size", "min_numeric_cell_size", "moment_digest_sig_digits"):
            value = getattr(self, field)
            if value < 1:
                raise ValueError(f"{field} must be >= 1; got {value}")
        if self.numeric_detail not in self.VALID_NUMERIC_DETAIL:
            raise ValueError(
                f"numeric_detail must be one of {self.VALID_NUMERIC_DETAIL}; got {self.numeric_detail!r}"
            )
        for q in self.quantiles:
            if not (0 < q < 1):
                raise ValueError(f"quantiles must lie in (0, 1); got {q}")
        object.__setattr__(self, "quantiles", tuple(sorted(self.quantiles)))
        object.__setattr__(self, "digest_subject_thresholds", tuple(sorted(self.digest_subject_thresholds)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain-JSON form embedded in every stats file.

        Examples:
            >>> from pprint import pprint
            >>> pprint(ExtractConfig(quantiles=(0.5,)).to_dict())
            {'code_metadata_only_digests': True,
             'digest_subject_thresholds': [20, 100, 1000],
             'emit_ontology_keys': False,
             'min_cell_size': 20,
             'min_numeric_cell_size': 20,
             'moment_digest_sig_digits': 6,
             'numeric_detail': 'moments',
             'quantiles': [0.5]}

            The round trip is exact, which is what lets comparison detect a config mismatch:

            >>> cfg = ExtractConfig(min_cell_size=50, numeric_detail="quantiles")
            >>> ExtractConfig.from_dict(cfg.to_dict()) == cfg
            True
        """
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}
        return {k: list(v) if isinstance(v, tuple) else v for k, v in sorted(out.items())}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExtractConfig:
        """Build a config from a plain dict, rejecting unknown keys.

        Args:
            raw: The mapping to read. Sequence values may be lists or tuples.

        Returns:
            The parsed config.

        Raises:
            ValueError: If ``raw`` contains keys this version does not understand.

        Examples:
            >>> ExtractConfig.from_dict({"min_cell_size": 5}).min_cell_size
            5

            Typos fail loudly rather than silently leaving the default in place -- a silently ignored
            ``min_cell_sise`` would mean shipping data at a threshold the caller did not choose:

            >>> ExtractConfig.from_dict({"min_cell_sise": 5})
            Traceback (most recent call last):
                ...
            ValueError: Unknown ExtractConfig keys: ['min_cell_sise']

            >>> ExtractConfig.from_dict({}) == ExtractConfig()
            True
        """
        known = {f.name for f in dataclasses.fields(cls) if f.init}
        if unknown := sorted(set(raw) - known):
            raise ValueError(f"Unknown ExtractConfig keys: {unknown}")
        kwargs = dict(raw)
        for seq_field in ("quantiles", "digest_subject_thresholds"):
            if seq_field in kwargs:
                kwargs[seq_field] = tuple(kwargs[seq_field])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExtractConfig:
        """Load a config from a YAML file.

        Args:
            path: The YAML file to read. An empty file yields the default config.

        Returns:
            The parsed config.

        Examples:
            >>> with yaml_disk({"cfg.yaml": {"min_cell_size": 30, "numeric_detail": "quantiles"}}) as root:
            ...     cfg = ExtractConfig.from_yaml(root / "cfg.yaml")
            >>> cfg.min_cell_size, cfg.numeric_detail
            (30, 'quantiles')

            An empty document is treated as "all defaults" rather than an error:

            >>> with yaml_disk({"cfg.yaml": None}) as root:
            ...     ExtractConfig.from_yaml(root / "cfg.yaml") == ExtractConfig()
            True
        """
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)
