"""Module entry point for ``python -m meds_summary_stats``.

The implementation lives in :mod:`meds_summary_stats.cli`; pytest excludes ``__main__.py`` from
doctest collection, so keeping this file empty of logic keeps the CLI fully covered.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
