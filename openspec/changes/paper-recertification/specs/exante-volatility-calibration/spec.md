# Ex-ante Volatility Calibration Specification

## Purpose

Ex-ante volatility (σ) terciles stratify evaluation into live-executable low/mid/high regimes. Thresholds computed on the same split being evaluated leak test-set information into an "operational" claim. This capability freezes tercile thresholds from train+val volatility and applies them, unchanged, to test.

## Requirements

### Requirement: Frozen Train+Val Tercile Thresholds

The system MUST compute p33/p66 percentile thresholds for ex-ante volatility exclusively from the concatenated train+val subset, in both `compute_stratification` (`src/build_exante_volatility.py`) and `compute_exante_terciles` (`src/build_exante_correlation.py`).

#### Scenario: Thresholds derived from train+val only

- GIVEN train, val, and test ex-ante σ arrays for a corridor/horizon
- WHEN tercile thresholds are computed
- THEN p33 and p66 are percentiles of the concatenated train+val array only
- AND test values are not included in the percentile computation

#### Scenario: Test-only extreme values do not shift thresholds

- GIVEN a test split containing σ values far outside the train+val range
- WHEN thresholds are computed
- THEN p33/p66 remain unchanged from the train+val-only computation

### Requirement: NaN Exclusion Before Threshold Computation

NaN entries in the train+val calibration array MUST be excluded before computing p33/p66 (ex-ante σ is NaN by construction for windows with fewer than 2 valid input timesteps — `src/build_exante_volatility.py:178-189`). Threshold computation MUST fail loudly, not return NaN, if the filtered calibration array is empty or too small to define three terciles.

#### Scenario: NaN entries excluded from calibration array

- GIVEN a train+val ex-ante σ array containing NaN entries
- WHEN p33/p66 thresholds are computed
- THEN NaN entries are excluded before the percentile computation
- AND the resulting p33/p66 are finite, non-NaN values

#### Scenario: Empty or too-small calibration array fails loudly

- GIVEN a train+val calibration array whose non-NaN entries are empty or fewer than the minimum required to define three terciles
- WHEN threshold computation is invoked
- THEN it raises an explicit error instead of returning NaN or degenerate thresholds

### Requirement: Frozen Thresholds Applied To Test Classification

Test-set rows MUST be classified into low/mid/high tercile using the frozen train+val thresholds; thresholds MUST NOT be recomputed from the test split.

#### Scenario: Test row classified against frozen threshold

- GIVEN a frozen (p33, p66) pair from train+val
- AND a test row with σ above the frozen p66 but below what test's own p66 would be
- WHEN the row is classified
- THEN it is labeled "high"

#### Scenario: NaN ex-ante rows excluded consistently

- GIVEN test rows with NaN `ex_ante_std`
- WHEN classification runs
- THEN those rows are excluded from classified output and do not affect the frozen thresholds

### Requirement: Shared Frozen-Threshold Contract Across Both Call Sites

`compute_stratification` and `compute_exante_terciles` MUST accept or derive thresholds from the same frozen-threshold computation so the two modules cannot silently diverge for the same corridor/horizon.

#### Scenario: Identical thresholds across both functions

- GIVEN the same train+val ex-ante σ array passed to both threshold paths
- WHEN each computes (p33, p66)
- THEN both return the same pair

#### Scenario: Integration-level equality across both consumers

- GIVEN identical fixture train+val ex-ante σ data for the same corridor/horizon
- WHEN the threshold path in `src/build_exante_volatility.py` and the threshold path in `src/build_exante_correlation.py` are each exercised through their real public entrypoint (not just a shared internal helper in isolation)
- THEN an integration-level test asserts both modules produce identical (p33, p66) for that corridor/horizon
