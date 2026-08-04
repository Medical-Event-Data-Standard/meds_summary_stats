"""Test set-up and fixtures code."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "json": json,
            "pl": pl,
        }
    )
