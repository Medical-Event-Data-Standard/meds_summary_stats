"""Regenerate the static example MEDS datasets under ``examples/datasets/``.

The datasets themselves are committed, so the commands in the README run against a fresh clone with
no setup. This script exists so those committed parquet files are reviewable rather than opaque: it
is the readable source of truth for what they contain, and
``tests/test_examples.py::test_committed_datasets_match_the_builder`` fails if the two drift apart.

Three variants, chosen to exercise the three outcomes a regression check has to distinguish:

``v1_baseline``
    A healthy ETL run. This is the fingerprint you would commit as your baseline.

``v1_etl_regression``
    The *same* source data (``dataset_version`` unchanged at ``1.0``) put through a broken ETL: a
    unit conversion applied to one measurement, one code renamed, and a handful of subjects dropped
    on the floor. Every one of those must be caught.

``v2_data_refresh``
    A genuine data refresh: ``dataset_version`` bumped to ``2.0``, more subjects, same schema and
    same vocabulary. Volume drift here is expected, not a regression -- which is what
    ``on_dataset_version_change: relax`` exists to express.

Run with ``uv run python examples/build_datasets.py``.
"""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

EXAMPLES_DIR = Path(__file__).parent
DATASETS_DIR = EXAMPLES_DIR / "datasets"

#: Fixed seed. The datasets are committed, so generation must be reproducible.
SEED = 20260804

#: The code vocabulary, with the ontology mapping each code carries in ``metadata/codes.parquet``.
#:
#: Deliberately mixed in shape: some codes use the ``//`` convention many ETLs adopt, some are flat.
#: Nothing in the package may treat the two differently, and an example that only showed one shape
#: would hide that.
VOCABULARY: dict[str, dict] = {
    "MEDS_BIRTH": {"description": "Birth", "parents": None, "numeric": None},
    "EYE_COLOR//BROWN": {"description": "Brown eyes", "parents": None, "numeric": None},
    "EYE_COLOR//BLUE": {"description": "Blue eyes", "parents": None, "numeric": None},
    "ADMISSION": {"description": "Hospital admission", "parents": None, "numeric": None},
    "DISCHARGE": {"description": "Hospital discharge", "parents": None, "numeric": None},
    "HR": {
        "description": "Heart rate",
        "parents": ["LOINC/8867-4"],
        "numeric": (75.0, 12.0),
    },
    "TEMP": {
        "description": "Body temperature, degrees Fahrenheit",
        "parents": ["LOINC/8310-5"],
        "numeric": (98.6, 0.8),
    },
    "LAB//GLUCOSE": {
        "description": "Serum glucose",
        "parents": ["LOINC/2345-7"],
        "numeric": (100.0, 25.0),
    },
    "LAB//SODIUM": {
        "description": "Serum sodium",
        "parents": ["LOINC/2951-2"],
        "numeric": (140.0, 3.0),
    },
    # Mapped into two vocabularies at once, so the example exercises the case where per-vocabulary
    # measurement counts deliberately over-sum against the dataset total.
    "DIAGNOSIS//E11.9": {
        "description": "Type 2 diabetes mellitus without complications",
        "parents": ["ICD10CM/E11.9", "SNOMED/44054006"],
        "numeric": None,
    },
}

_DYNAMIC_CODES = ["ADMISSION", "HR", "TEMP", "LAB//GLUCOSE", "LAB//SODIUM", "DIAGNOSIS//E11.9", "DISCHARGE"]
#: Split proportions. Assignment is by subject index rather than by sampling, so the split sizes
#: are exact -- an example whose splits sometimes fell below the suppression floor would demonstrate
#: something different every time it was regenerated.
_SPLIT_FRACTIONS = [("train", 0.70), ("tuning", 0.15), ("held_out", 0.15)]


def _subject_rows(rng: random.Random, subject_id: int) -> list[dict]:
    """Generate one subject's measurements: a birth, a static eye colour, then several visits."""
    birth = datetime(1940, 1, 1) + timedelta(days=rng.randrange(0, 22_000))
    rows: list[dict] = [
        {
            "subject_id": subject_id,
            "time": None,
            "code": rng.choice(["EYE_COLOR//BROWN", "EYE_COLOR//BLUE"]),
            "numeric_value": None,
        },
        {"subject_id": subject_id, "time": birth, "code": "MEDS_BIRTH", "numeric_value": None},
    ]

    for _ in range(rng.randrange(1, 4)):
        admitted = datetime(2018, 1, 1) + timedelta(
            days=rng.randrange(0, 1460), hours=rng.randrange(0, 24), minutes=rng.randrange(0, 60)
        )
        rows.append({"subject_id": subject_id, "time": admitted, "code": "ADMISSION", "numeric_value": None})
        for step in range(rng.randrange(2, 7)):
            code = rng.choice(_DYNAMIC_CODES[1:-1])
            spec = VOCABULARY[code]["numeric"]
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": admitted + timedelta(hours=step + 1),
                    "code": code,
                    "numeric_value": None if spec is None else round(rng.gauss(*spec), 3),
                }
            )
        rows.append(
            {
                "subject_id": subject_id,
                "time": admitted + timedelta(days=rng.randrange(1, 8)),
                "code": "DISCHARGE",
                "numeric_value": None,
            }
        )
    return rows


def _split_for(index: int, n_subjects: int) -> str:
    """Assign subject ``index`` to a split, deterministically and in exact proportion."""
    position = index / n_subjects
    cumulative = 0.0
    for name, fraction in _SPLIT_FRACTIONS:
        cumulative += fraction
        if position < cumulative:
            return name
    return _SPLIT_FRACTIONS[-1][0]


def build(n_subjects: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the measurement table and subject-split table for a dataset of ``n_subjects``."""
    rng = random.Random(seed)
    rows: list[dict] = []
    splits: list[dict] = []
    for i in range(n_subjects):
        subject_id = 100_000 + i
        rows.extend(_subject_rows(rng, subject_id))
        splits.append({"subject_id": subject_id, "split": _split_for(i, n_subjects)})

    data = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
        },
    ).sort("subject_id", "time", nulls_last=False)
    return data, pl.DataFrame(splits, schema={"subject_id": pl.Int64, "split": pl.String})


def code_metadata(vocabulary: dict[str, dict]) -> pl.DataFrame:
    """Build ``metadata/codes.parquet`` for a vocabulary."""
    return pl.DataFrame(
        [
            {"code": code, "description": spec["description"], "parent_codes": spec["parents"]}
            for code, spec in sorted(vocabulary.items())
        ],
        schema={"code": pl.String, "description": pl.String, "parent_codes": pl.List(pl.String)},
    )


def write_dataset(
    root: Path, data: pl.DataFrame, splits: pl.DataFrame, vocabulary: dict[str, dict], metadata: dict
) -> Path:
    """Write a complete MEDS dataset, sharded by split.

    Sharding by split keeps every subject whole within a shard, which MEDS requires.
    """
    if root.exists():
        shutil.rmtree(root)
    (root / "metadata").mkdir(parents=True)

    with_split = data.join(splits, on="subject_id", how="left")
    for split in sorted(splits["split"].unique().to_list()):
        shard_dir = root / "data" / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        (
            with_split.filter(pl.col("split") == split)
            .drop("split")
            .sort("subject_id", "time", nulls_last=False)
            .write_parquet(shard_dir / "0.parquet")
        )

    splits.sort("subject_id").write_parquet(root / "metadata" / "subject_splits.parquet")
    code_metadata(vocabulary).write_parquet(root / "metadata" / "codes.parquet")
    (root / "metadata" / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return root


def build_v1_baseline(root: Path) -> Path:
    """A healthy ETL run.

    The fingerprint of this is what you would commit as a baseline.
    """
    data, splits = build(n_subjects=160, seed=SEED)
    return write_dataset(
        root,
        data,
        splits,
        VOCABULARY,
        {
            "dataset_name": "example_hospital",
            "dataset_version": "1.0",
            "etl_name": "example_etl",
            "etl_version": "0.3.1",
            "meds_version": "0.4.1",
        },
    )


def build_v1_etl_regression(root: Path) -> Path:
    """The same source data through a broken ETL. Three independent bugs, all of which must be caught.

    1. ``TEMP`` is emitted in Celsius rather than Fahrenheit -- a unit conversion nobody asked for.
    2. ``LAB//SODIUM`` is renamed to ``LAB//NA``, silently changing the vocabulary.
    3. Every twentieth subject is dropped on the floor.

    ``dataset_version`` is unchanged, so none of this is excusable as a data refresh.
    """
    data, splits = build(n_subjects=160, seed=SEED)

    dropped = set(splits["subject_id"].to_list()[::20])
    data = data.filter(~pl.col("subject_id").is_in(dropped))
    splits = splits.filter(~pl.col("subject_id").is_in(dropped))

    data = data.with_columns(
        numeric_value=pl.when(pl.col("code") == "TEMP")
        .then((pl.col("numeric_value") - 32.0) * 5.0 / 9.0)
        .otherwise(pl.col("numeric_value")),
        code=pl.when(pl.col("code") == "LAB//SODIUM").then(pl.lit("LAB//NA")).otherwise(pl.col("code")),
    )

    vocabulary = {("LAB//NA" if k == "LAB//SODIUM" else k): v for k, v in VOCABULARY.items()}
    return write_dataset(
        root,
        data,
        splits,
        vocabulary,
        {
            "dataset_name": "example_hospital",
            "dataset_version": "1.0",
            "etl_name": "example_etl",
            "etl_version": "0.4.0",
            "meds_version": "0.4.1",
        },
    )


def build_v2_data_refresh(root: Path) -> Path:
    """A genuine data refresh: more subjects, same schema, same vocabulary, new dataset version."""
    data, splits = build(n_subjects=220, seed=SEED)
    return write_dataset(
        root,
        data,
        splits,
        VOCABULARY,
        {
            "dataset_name": "example_hospital",
            "dataset_version": "2.0",
            "etl_name": "example_etl",
            "etl_version": "0.3.1",
            "meds_version": "0.4.1",
        },
    )


BUILDERS = {
    "v1_baseline": build_v1_baseline,
    "v1_etl_regression": build_v1_etl_regression,
    "v2_data_refresh": build_v2_data_refresh,
}


def main() -> None:
    """Regenerate every example dataset in place."""
    for name, builder in BUILDERS.items():
        root = builder(DATASETS_DIR / name)
        n_rows = sum(pl.read_parquet(p).height for p in sorted((root / "data").rglob("*.parquet")))
        print(f"{name}: {n_rows} measurements")


if __name__ == "__main__":
    main()
