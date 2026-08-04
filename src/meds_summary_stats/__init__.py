"""Privacy-safe summary statistics over MEDS datasets, for ETL validation and regression testing."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - trivial packaging fallback
    __version__ = version("meds_summary_stats")
except PackageNotFoundError:  # pragma: no cover - only hit in an un-installed source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
