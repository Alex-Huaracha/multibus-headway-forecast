# Reproducibility Manifest Specification

## Purpose

A reviewer must be able to rebuild every recertified table and figure from pinned Kaggle datasets/kernels without contacting the author, and trust that rerunning local report builders on identical inputs reproduces the same numbers (or a documented tolerance).

## Requirements

### Requirement: README Reproduction Section

`README.md` MUST include a reproduction section with ordered steps (environment setup, Kaggle download commands, local rebuild commands) sufficient to regenerate the multihorizon tables/figures from a clean checkout.

#### Scenario: Reviewer follows documented steps

- GIVEN a clean checkout of the repository
- WHEN the reviewer follows the README reproduction steps in order
- THEN the multihorizon CSVs and figures under `docs/resultados/` are regenerated locally

### Requirement: Dataset/Kernel Version Pinning

`docs/dataset-manifest.md` MUST record, for each dataset/kernel consumed by the 6 recertified notebook families, the dataset or kernel ID, version/`lastUpdated` timestamp, and the builder/commit that produced it.

#### Scenario: Manifest entry exists per family

- GIVEN the 6 recertified notebook families (11, 12, 13, 17, 18, 19)
- WHEN the manifest is inspected
- THEN each family's upstream Kaggle kernel/dataset has a pinned entry with version and producing builder/commit

#### Scenario: Manifest re-emitted on version change

- GIVEN a pinned dataset/kernel's `lastUpdated` changes
- WHEN the manifest is updated
- THEN a new entry replaces the stale one and the prior entry is added to the revision history table

### Requirement: Non-Versioned Artifact Inventory

The reproducibility docs MUST enumerate gitignored/non-committed artifacts (raw data, processed Parquet, notebook execution outputs) and state which pinned Kaggle dataset/kernel is authoritative for each.

#### Scenario: Reviewer resolves artifact provenance

- GIVEN any path under `data/` or a non-committed path under `docs/resultados/`
- WHEN the reviewer consults the reproducibility docs
- THEN they can determine whether the path is committed or must be rebuilt, and from which pinned Kaggle source

### Requirement: Deterministic Report Builders Or Documented Tolerance

Aggregation report builders that produce `docs/resultados/csv-multihorizon/*.csv` (significance, volatility, ex-ante correlation, paired audit) MUST produce byte-identical output across repeated runs on identical input, OR the reproducibility docs MUST state the expected numeric tolerance and affected columns when determinism cannot be guaranteed.

#### Scenario: Rerun produces identical bytes

- GIVEN a fixed set of residual/input CSVs
- WHEN a report builder (e.g. `src/build_paired_audit.py`) is run twice
- THEN both output CSVs are byte-identical

#### Scenario: Tolerance documented when non-deterministic

- GIVEN a report builder whose parallel reduction order is not pinned
- WHEN reruns produce numeric noise (e.g. differing beyond the 12th significant digit)
- THEN the reproducibility docs state the tolerance and affected builder/columns, and a test asserts reruns stay within it
