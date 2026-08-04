# The stats surface

Every example on this page is executed as a doctest, so what is printed here is what the tool
actually emits.

All examples use `simple_static_MEDS`, a six-subject dataset from
[`meds-testing-helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers), with
suppression turned down to 1 so the structure is visible. On real data you would leave the default
floor of 20 in place, and most of the keyed cells below would collapse into `<suppressed>` rollups.

```python
>>> from meds_summary_stats.config import ExtractConfig
>>> from meds_summary_stats.extract import extract_summary_stats
>>> cfg = ExtractConfig(min_cell_size=1, min_numeric_cell_size=1)
>>> stats = extract_summary_stats(simple_static_MEDS, cfg)
>>> sorted(stats)
['code_metadata', 'codes', 'config', 'counts', 'dataset', 'dataset_digest', 'extracted_at',
 'layout', 'meds_summary_stats_version', 'per_subject', 'splits', 'stats_schema_version',
 'time']

```

## The privacy contract

Three rules, enforced structurally rather than by convention:

1. **No code string is ever emitted.** Not at any suppression threshold, not under any flag. The
    vocabulary is represented only by digests and by aggregate distributions over it.
2. **No cell is reported below the subject floor.** Split, calendar-year, and ontology-vocabulary
    entries backed by fewer than `min_cell_size` distinct subjects are dropped — the *key itself* is
    never written, only a `<suppressed>` rollup of how many were dropped and their pooled volume.
    The rollup deliberately omits a pooled subject count, since summing distinct-subject counts
    across keys would overstate the cohort.
3. **No extremes.** There is no `min` or `max` anywhere in the document — not on `numeric_value`,
    not on `time`. The subject with the longest record or the most extreme lab value is precisely
    where re-identification starts; quantiles replace them, and quantiles are themselves suppressed
    below `min_numeric_cell_size` observations.

```python
>>> import json
>>> blob = json.dumps(stats)
>>> any(code in blob for code in ("EYE_COLOR//BROWN", "MEDS_BIRTH", "HR", "TEMP"))
False
>>> any(str(subject_id) in blob for subject_id in (239684, 1195293, 68729))
False
>>> "max" in blob or '"min"' in blob
False

```

## `counts` — headline volume

The numbers the motivating use case turns on. `n_events` counts distinct `(subject_id, time)` pairs,
with all of a subject's static (null-time) rows counting as one event.

```python
>>> print(json.dumps(stats["counts"], indent=2, sort_keys=True))
{
  "n_events": 34,
  "n_measurements": 62,
  "n_static_measurements": 12,
  "n_subjects": 6,
  "n_subjects_with_birth": 6,
  "n_subjects_with_death": 0,
  "n_subjects_with_static": 6,
  "n_unique_codes": 11,
  "numeric": {
    "mean": 113.994,
    "n_finite": 38,
    "n_infinite": 0,
    "n_nan": 0,
    "n_negative": 0,
    "n_present": 38,
    "n_zero": 0,
    "std": 26.385
  },
  "quality": {
    "n_empty_code": 0,
    "n_null_code": 0,
    "n_null_subject_id": 0
  }
}

```

`numeric.mean` and `numeric.std` are computed over *finite* values only. A single NaN would
otherwise propagate through the mean and erase the statistic; NaN and infinity are counted
separately instead. The `quality` block holds invariants that should never move off zero, which is
why the default policy compares them with an absolute threshold of 0.

## `codes` — the vocabulary, without the vocabulary

The hardest part of the design: be sensitive to vocabulary drift without publishing the vocabulary.
Three complementary views, which fail differently.

### Digests

```python
>>> digests = stats["codes"]["digests"]
>>> sorted(digests)
['alphabetical', 'by_measurement_frequency', 'by_subject_frequency', 'numeric_moments',
 'thresholded']
>>> digests["alphabetical"]
'sha256:...'

```

| Digest                     | Changes when                                              | Default severity |
| -------------------------- | --------------------------------------------------------- | ---------------- |
| `alphabetical`             | the **set** of codes changes — and only then              | error            |
| `by_measurement_frequency` | the set changes, or codes reorder by occurrence count     | warning          |
| `by_subject_frequency`     | the set changes, or codes reorder by subject count        | warning          |
| `numeric_moments`          | any code's mean or std shifts beyond 6 significant digits | warning          |
| `thresholded.<k>.*`        | as above, restricted to codes with ≥ *k* subjects         | error / warning  |

The frequency-ordered digests are sensitive tripwires but noisy ones: two codes with counts 1000 and
1001 swap places if either moves by two. That is exactly why the thresholded variants exist —
restricted to codes common enough that their ordering is stable, they trip on real reshuffling
rather than on counting noise. Each thresholded entry reports its own `n_codes`, so a digest change
can be read against a changing denominator.

```python
>>> {k: v["n_codes"] for k, v in sorted(digests["thresholded"].items())}
{'100': 0, '1000': 0, '20': 0}

```

(All zero here: a six-subject dataset has no code with 20 subjects behind it. On real data these
cover most of the vocabulary.)

`numeric_moments` is a change *detector* rather than a change *description*: it hashes each
sufficiently-common code's rounded mean and standard deviation, so a unit-conversion regression on
any single lab trips it without the report saying which lab. Combined with the per-vocabulary
moments in the [ontology block](#code_metadata-and-ontology--the-only-grouping-axis),
you get "something shifted, and it was among the LOINC-mapped codes" — enough to start debugging,
and no more than that.

### Shape and concentration

```python
>>> print(json.dumps(stats["codes"]["concentration"], indent=2, sort_keys=True))
{
  "frac_measurements_in_top_10": 0.983871,
  "frac_measurements_in_top_100": 1.0,
  "frac_measurements_in_top_1000": 1.0,
  "n_codes_covering_50pct": 2,
  "n_codes_covering_90pct": 8,
  "n_codes_covering_99pct": 11
}

```

A mapping regression that collapses many codes into one, or explodes one into many, moves these
sharply even when the total measurement count is unchanged. `codes.occurrence` complements them with
the distribution of per-code measurement and subject counts.

### What is deliberately absent: any partition by code structure

Many MEDS ETLs write codes as `SPACE//CODE` — `LAB//GLUCOSE`, `DIAGNOSIS//ICD10CM//E11.9`. Splitting
on `//` would give a cheap "which part of the vocabulary moved" axis, and this package does not do
it. Two reasons, either sufficient:

- **MEDS does not require the convention.** An axis that exists on some datasets and not others is
    not something a comparison policy can be written against, and its silent absence would read as
    "nothing changed here" rather than "this was never measured".
- **For a flat vocabulary it is not a partition at all.** If codes are `HR`, `HEIGHT`, `TEMP`, then
    the "namespace" of each code is the code, and emitting the namespace list emits the vocabulary.

So the entire `codes` block is structure-blind:

```python
>>> sorted(stats["codes"])
['concentration', 'digests', 'n_codes_below_min_cell_size', 'n_unique_codes', 'occurrence']

```

The grouping axis comes from `parent_codes` instead, where the groups are declared by the dataset
rather than inferred from string shape. That is the next section.

## `code_metadata` and `ontology` — the only grouping axis

`parent_codes` entries point into published external terminologies — `LOINC/8867-4`,
`ICD10CM/E11.9`. Unlike a dataset's own code strings, the identity of those vocabularies is a
property of the terminology, not of any patient, and — decisively — the grouping is *declared by the
dataset* rather than inferred from how the codes happen to be spelled. That is why this is both the
one block that reports names and the one axis along which the vocabulary is partitioned.

### Per-vocabulary statistics

```python
>>> print(json.dumps(stats["code_metadata"]["ontology"]["by_vocabulary"], indent=2, sort_keys=True))
{
  "LOINC": {
    "keys_digest": "sha256:...",
    "n_child_codes": 2,
    "n_distinct_keys": 2,
    "n_measurements": 32,
    "n_subjects": 6,
    "numeric": {
      "mean": 104.691,
      "n_finite": 32,
      "std": 16.0519
    },
    "recognized": true
  }
}

```

`numeric` is the finest-grained "where did it move" signal the document offers, and the direct
replacement for a per-namespace view: a unit conversion applied to the LOINC-mapped labs moves this
mean while leaving every global count untouched.

`recognized` flags whether the vocabulary is a known OMOP CDM `vocabulary_id`. Suppression applies
to the statistics as everywhere else — a vocabulary reached by too few subjects is rolled up into
`<suppressed>`, because its measurement count would otherwise describe a tiny cohort. Set
`emit_ontology_keys: true` to list every key per vocabulary; that is off by default for output
size, not for privacy.

A code with parents in two vocabularies is counted under both, so `n_measurements` across
vocabularies deliberately over-sums relative to the dataset total.

### Coverage — where an unresolved mapping table announces itself

These are dataset-level ratios and counts, so they are never suppressed.

```python
>>> print(json.dumps(stats["code_metadata"]["ontology"]["coverage"], indent=2, sort_keys=True))
{
  "frac_measurements_mapped": 0.516129,
  "frac_observed_codes_with_parents": 0.181818,
  "n_measurements_mapped": 32,
  "n_measurements_unmapped": 30,
  "n_observed_codes": 11,
  "n_observed_codes_with_parents": 2,
  "n_observed_codes_without_parents": 9
}

```

This is the block that catches a class of regression nothing else can. When a concept-mapping join
silently stops resolving, the ETL still emits every row: the subject count, measurement count, code
count, and schema are all identical. What changes is that codes stop reaching an ontology —
`frac_measurements_mapped` falls, `n_observed_codes_without_parents` rises. The default policy
treats any downward movement in the former as at least a warning.

### Structure of the linkage

```python
>>> ontology = stats["code_metadata"]["ontology"]
>>> {k: ontology[k] for k in ("n_parent_links", "n_distinct_parents", "n_vocabularies",
...                           "n_vocabularies_recognized", "n_codes_in_multiple_vocabularies",
...                           "n_parents_without_vocabulary")}
{'n_parent_links': 2, 'n_distinct_parents': 2, 'n_vocabularies': 1, 'n_vocabularies_recognized': 1,
 'n_codes_in_multiple_vocabularies': 0, 'n_parents_without_vocabulary': 0}

```

`parents_per_code` and `vocabularies_per_code` report the distributions of those two counts across
the metadata vocabulary. `n_parents_without_vocabulary` counts parent codes with no `/` separator at
all — those cannot be attributed to a terminology and are pooled rather than guessed at.

Alongside all of this, `code_metadata` checks the MEDS invariant that every code in the data appears
in `metadata/codes.parquet`:

```python
>>> {k: stats["code_metadata"][k] for k in
...  ("n_codes", "n_data_codes_missing_from_metadata", "n_metadata_codes_absent_from_data")}
{'n_codes': 5, 'n_data_codes_missing_from_metadata': 6, 'n_metadata_codes_absent_from_data': 0}

```

(This particular fixture is deliberately incomplete; on a valid dataset the middle number is 0, and
the default policy treats any nonzero value as an error.)

## `dataset_digest` — "is this the same dataset at all?"

Everything else on this page measures the data. This block asks the prior question, and answers it
exactly: a sha256 over the bytes of every file the MEDS schema defines.

```python
>>> digest = stats["dataset_digest"]
>>> digest["method"], digest["n_files"], digest["data"]["n_files"]
('files', 7, 4)
>>> digest["combined"]
'sha256:...'

```

Three deliberate choices, each of which would otherwise make it useless:

- **Only schema files count.** `data/**/*.parquet`, `metadata/codes.parquet`,
    `metadata/subject_splits.parquet`, `metadata/dataset.json`. Task labels, caches, notes, and
    editor droppings in the dataset root are ignored, so they cannot make two copies of the same
    dataset compare unequal.
- **`metadata/dataset.json` is excluded from `combined`.** It carries `created_at` and
    `etl_version`, which change on every run by design; folding them in would invalidate the digest
    every time and mean nothing. It gets its own `dataset_metadata` digest, which the default policy
    ignores.
- **`combined` hashes the set of file *contents*, not their paths.** Renaming or re-nesting a shard
    without changing what is in it leaves it alone. Shard naming is tracked separately by
    `layout.shard_names_digest`, so a rename is reported once, in the right place.

```python
>>> sorted(digest)
['by_file', 'code_metadata', 'combined', 'data', 'dataset_metadata', 'method', 'n_files',
 'subject_splits']
>>> sorted(digest["by_file"])
['data/held_out/0.parquet', 'data/train/0.parquet', 'data/train/1.parquet',
 'data/tuning/0.parquet', 'metadata/codes.parquet', 'metadata/dataset.json',
 'metadata/subject_splits.parquet']

```

`by_file` pinpoints *which* file moved, and is dropped above 64 files so a thousand-shard dataset
does not produce a thousand-line block.

> [!IMPORTANT]
> This is a digest of **bytes**, which makes it the strictest identity check available and also
> means it is sensitive to re-encoding. The same rows rewritten by a different parquet writer, or
> with different compression or row-group sizes, produce a different digest. That is why the default
> policy treats a digest change as a **warning** rather than an error.
>
> The false positive is self-diagnosing: when only the encoding changed, the digest moves and every
> single statistic stays put. When the data changed, both move. Once you have confirmed your ETL is
> byte-reproducible, promote it in your policy:
>
> ```yaml
> rules:
>   - {path: dataset_digest.combined, kind: exact, on_change: error}
> ```

Set `dataset_digest: none` (or `--dataset-digest none`) to skip it, which also skips the extra pass
over the raw bytes.

## `layout` — schema and physical structure

Schema drift is the cheapest regression to detect and among the most damaging to miss.

```python
>>> layout = stats["layout"]
>>> for name, dtype in layout["data_columns"].items():
...     print(f"{name:15s} {dtype}")
code            String
numeric_value   Float32
subject_id      Int64
time            Datetime(time_unit='us', time_zone=None)
>>> layout["n_data_shards"], layout["column_dtype_conflicts"], layout["missing_core_columns"]
(4, {}, [])

```

`column_presence` counts how many shards carry each column, so a column present in only some shards
— a real ETL bug that a union-only view hides — is visible. `column_dtype_conflicts` does the same
for dtypes. Shard names are reported in full up to 64 shards and as a digest always, so a
thousand-shard dataset does not produce a thousand-line document.

## `per_subject` and `time` — distributions, never extremes

```python
>>> stats["per_subject"]["n_measurements"]
{'n': 6, 'mean': 10.3333, 'std': 4.13118, 'quantiles': {'p1': 7.0, 'p5': 7.0, 'p25': 7.0,
 'p50': 9.0, 'p75': 12.5, 'p95': 16.0, 'p99': 16.8}}
>>> sorted(stats["per_subject"])
['n_events', 'n_measurements', 'n_unique_codes', 'span_days']

```

`time.by_year` is the workhorse of the time block: cheap, high-count, and immediately legible when
something goes wrong, since a timezone or epoch-parsing regression relocates whole years of data at
once. Suppression also disposes of the stray year-1900 and year-2099 rows that bad date parsing
produces — they land in the rollup rather than as reportable cells.

```python
>>> stats["time"]["by_year"]["2010"]
{'n_measurements': 44, 'n_subjects': 6}
>>> stats["time"]["n_years_observed"], stats["time"]["n_measurements_with_time"]
(6, 50)

```

Full-column quantiles over `numeric_value` and `time` are opt-in via `numeric_detail: quantiles`,
because unlike the per-subject and per-code distributions (whose frames are small) they require
materializing a column of the full dataset.

## `dataset`, `config`, and provenance

`dataset` is a verbatim passthrough of `metadata/dataset.json` — every field in the MEDS dataset
metadata schema is dataset-level provenance, not subject data. `config` records the exact extraction
settings, and comparison refuses to diff two documents whose configs differ: digests taken at
different subject thresholds are simply different quantities, and their inequality carries no
information.

```python
>>> stats["config"]["min_cell_size"], stats["stats_schema_version"]
(1, 1)

```
