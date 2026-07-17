# Notebook Generation Integrity Specification

## Purpose

The 6 recertified notebook families (11, 12, 13 multihorizon; 17, 18, 19 E4) must be regenerated from their corrected builders after the ex-ante calibration and paired-audit fixes land, and must never reintroduce the train-only winsorization pattern (`non_train`) already known to affect the out-of-scope legacy notebooks (05-09, 14, 15).

## Requirements

### Requirement: Regeneration From Corrected Builders

The 6 notebook families MUST be regenerated exclusively from their current `src/build_notebook_{11,12,13,17,18,19}_*.py` builders so committed `.ipynb` cell source matches builder output.

#### Scenario: Regeneration is reproducible

- GIVEN a fixed builder module version
- WHEN the notebook is regenerated twice
- THEN both runs produce identical cell source (nbformat cell `source` fields)

#### Scenario: Committed notebook matches current builder

- GIVEN a committed `.ipynb` under `notebooks/{11,12,13,17,18,19}_*/`
- WHEN it is compared against a fresh regeneration from the corresponding builder
- THEN cell source is identical (no manual post-generation edits)

### Requirement: No Train-Only Winsorization Guard

An automated test MUST scan each of the 6 regenerated `.ipynb` files' cell source via `nbformat` and MUST fail if any cell contains the `non_train` train-only-winsorization identifier/pattern.

#### Scenario: Guard passes on clean notebooks

- GIVEN the 6 regenerated notebooks contain no `non_train` occurrences
- WHEN the guard test runs
- THEN it passes

#### Scenario: Guard fails with actionable location

- GIVEN a regenerated notebook whose cell source contains `non_train`
- WHEN the guard test runs
- THEN it fails and reports the notebook path and cell index

### Requirement: Frozen Input-Hash Gate With Required Atypical Feature

Every generated DL notebook MUST resolve its training inputs (headways parquets and `atypical_days.csv`) through a hash-verifying resolver pinned to frozen SHA-256 values, MUST declare `alexhuaracha/02-eda-corridors` as a kernel source so `atypical_days.csv` mounts, and MUST NOT contain a silent fallback that leaves the atypical-day feature inert.

Rationale: all pre-recertification Kaggle runs trained with `atypical_flag=0` because the CSV never mounted and the notebooks fell back silently; the recertified runs activate the documented DL-2 feature and pin input bytes.

#### Scenario: Inputs are hash-pinned

- GIVEN a generated notebook of any of the 6 families
- WHEN its code cells are scanned
- THEN they contain the frozen SHA-256 for each required input and resolve every input through the hash-verifying resolver

#### Scenario: Atypical days CSV is required

- GIVEN a generated notebook of any of the 6 families
- WHEN its code cells are scanned
- THEN `atypical_days.csv` resolves through the hash gate, the silent `atypical_path = None` fallback is absent, and an empty parsed date set raises before training

#### Scenario: Mismatched input stops before training

- GIVEN a required input file whose bytes differ from the frozen snapshot
- WHEN the notebook runs
- THEN input resolution raises before any split, window, or training step executes

### Requirement: Out-of-Scope Notebooks Excluded From Guard And Regeneration

Notebooks 05-09, 14, and 15 MUST NOT be regenerated or modified by this change, and MUST be excluded from the guard's scanned set.

#### Scenario: Guard scope is limited to recertified families

- GIVEN the guard's configured notebook set is `{11, 12, 13, 17, 18, 19}`
- WHEN the guard test runs
- THEN notebooks 05-09, 14, 15 are not scanned and their known `non_train` pattern does not fail the guard
