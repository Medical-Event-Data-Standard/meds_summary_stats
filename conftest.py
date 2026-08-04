"""Test set-up and fixtures code."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from meds_summary_stats.config import SUPPRESSED


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
):
    """Pre-populate the doctest namespace.

    ``yaml_disk``, ``print_directory``, and ``PrintConfig`` are registered by their own pytest
    plugins, so they do not need to be added here. The two MEDS dataset fixtures come from
    ``meds_testing_helpers``; binding them into the namespace lets doctests read as
    ``extract_summary_stats(simple_static_MEDS)`` rather than carrying dataset construction
    boilerplate.
    """
    doctest_namespace.update(
        {
            # Absolute, so README and docs examples resolve regardless of the working directory
            # pytest was invoked from.
            "EXAMPLES": Path(__file__).parent / "examples",
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "json": json,
            "pl": pl,
            "pa": pa,
            "SUPPRESSED": SUPPRESSED,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
        }
    )
