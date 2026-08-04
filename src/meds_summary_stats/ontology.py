"""Parsing and classification of ontology references found in ``parent_codes``.

MEDS ``parent_codes`` entries point into external, published vocabularies -- ``LOINC/8867-4``,
``ICD10CM/E11.9``, and so on. Unlike the dataset's own code strings, the identity of those vocabularies
is a property of the terminology, not of any patient, so reporting *which* vocabularies an ETL mapped
into (and how densely) is both safe and a sharp regression signal: a mapping table that silently stops
resolving shows up here as a vocabulary that lost most of its links.
"""

from __future__ import annotations

import polars as pl

from .config import NO_VOCABULARY

#: OMOP CDM ``vocabulary_id`` values common in MEDS ETLs.
#:
#: Used only to annotate each observed vocabulary with a ``recognized`` flag; an unrecognized
#: vocabulary is reported the same way, just flagged. The list is intentionally partial -- it is a
#: readability aid, not a validation gate.
KNOWN_VOCABULARIES: frozenset[str] = frozenset(
    {
        "ATC",
        "CPT4",
        "CVX",
        "Cancer Modifier",
        "Condition Type",
        "Currency",
        "Death Type",
        "Device Type",
        "Domain",
        "Drug Type",
        "Ethnicity",
        "Gender",
        "HCPCS",
        "HemOnc",
        "ICD10",
        "ICD10CM",
        "ICD10PCS",
        "ICD9CM",
        "ICD9Proc",
        "ICDO3",
        "LOINC",
        "MedDRA",
        "Meas Type",
        "Medicare Specialty",
        "NDC",
        "NUCC",
        "Nebraska Lexicon",
        "None",
        "Note Type",
        "OMOP Extension",
        "OSM",
        "Obs Period Type",
        "Observation Type",
        "PPI",
        "Plan",
        "Procedure Type",
        "Provider",
        "Race",
        "Relationship",
        "Revenue Code",
        "RxNorm",
        "RxNorm Extension",
        "SNOMED",
        "SPL",
        "Specimen Type",
        "Sponsor",
        "Supplier",
        "Type Concept",
        "UCUM",
        "Visit",
        "Visit Type",
        "Vocabulary",
    }
)


def split_parent_code(parent: str) -> tuple[str, str]:
    """Split a parent code into its ``(vocabulary, key)`` parts.

    The vocabulary is everything before the first ``/``. Any immediately following ``/`` characters are
    consumed as part of the separator, so both the OMOP-style ``VOCAB/KEY`` and the MEDS-style
    ``VOCAB//KEY`` spellings resolve identically.

    Args:
        parent: A single ``parent_codes`` entry.

    Returns:
        A ``(vocabulary, key)`` pair. Parents with no separator get
        :data:`~meds_summary_stats.config.NO_VOCABULARY` and are reported only in aggregate.

    Examples:
        >>> split_parent_code("LOINC/8867-4")
        ('LOINC', '8867-4')
        >>> split_parent_code("LOINC//8867-4")
        ('LOINC', '8867-4')

        Keys may themselves contain slashes; only the first run is the separator:

        >>> split_parent_code("SNOMED/123/456")
        ('SNOMED', '123/456')

        A parent with no separator cannot be attributed to a vocabulary, so it is bucketed rather than
        guessed at:

        >>> split_parent_code("8867-4")
        ('<no-vocabulary>', '8867-4')

        Degenerate inputs bucket too, rather than producing an empty vocabulary name:

        >>> split_parent_code("/8867-4")
        ('<no-vocabulary>', '8867-4')
        >>> split_parent_code("")
        ('<no-vocabulary>', '')
    """
    vocabulary, separator, key = parent.partition("/")
    if not separator or not vocabulary:
        return (NO_VOCABULARY, parent.lstrip("/"))
    return (vocabulary, key.lstrip("/"))


def vocabulary_expr(column: str = "parent_code") -> pl.Expr:
    """Polars expression extracting the vocabulary from a parent-code column.

    This mirrors :func:`split_parent_code` but runs vectorized over a whole column. The two are kept in
    sync by :func:`~tests.test_ontology` (property-based) rather than by inspection.

    Args:
        column: Name of the string column holding one parent code per row.

    Returns:
        A string expression yielding the vocabulary for each row.

    Examples:
        >>> df = pl.DataFrame({"parent_code": ["LOINC/8867-4", "ICD10CM//E11.9", "bare", "/x", ""]})
        >>> df.select(vocab=vocabulary_expr())["vocab"].to_list()
        ['LOINC', 'ICD10CM', '<no-vocabulary>', '<no-vocabulary>', '<no-vocabulary>']
    """
    head = pl.col(column).str.splitn("/", 2).struct.field("field_0")
    return (
        pl.when(pl.col(column).str.contains("/", literal=True) & (head.str.len_bytes() > 0))
        .then(head)
        .otherwise(pl.lit(NO_VOCABULARY))
        .alias("vocabulary")
    )


def key_expr(column: str = "parent_code") -> pl.Expr:
    """Polars expression extracting the within-vocabulary key from a parent-code column.

    Args:
        column: Name of the string column holding one parent code per row.

    Returns:
        A string expression yielding the key for each row.

    Examples:
        >>> df = pl.DataFrame({"parent_code": ["LOINC/8867-4", "ICD10CM//E11.9", "bare", "/x"]})
        >>> df.select(key=key_expr())["key"].to_list()
        ['8867-4', 'E11.9', 'bare', 'x']
    """
    tail = pl.col(column).str.splitn("/", 2).struct.field("field_1")
    head = pl.col(column).str.splitn("/", 2).struct.field("field_0")
    has_vocabulary = pl.col(column).str.contains("/", literal=True) & (head.str.len_bytes() > 0)
    return (
        pl.when(has_vocabulary)
        .then(tail)
        .otherwise(pl.col(column))
        .str.strip_chars_start("/")
        .fill_null("")
        .alias("key")
    )


def is_recognized(vocabulary: str) -> bool:
    """Whether ``vocabulary`` is a known OMOP CDM vocabulary identifier.

    Args:
        vocabulary: The vocabulary name to check. Matching is case-insensitive, since ETLs vary on
            whether they emit ``SNOMED`` or ``Snomed``.

    Returns:
        True if recognized.

    Examples:
        >>> is_recognized("LOINC"), is_recognized("loinc"), is_recognized("RxNorm")
        (True, True, True)
        >>> is_recognized("MY_CUSTOM_VOCAB"), is_recognized("<no-vocabulary>")
        (False, False)
    """
    lowered = {v.lower() for v in KNOWN_VOCABULARIES}
    return vocabulary.lower() in lowered
