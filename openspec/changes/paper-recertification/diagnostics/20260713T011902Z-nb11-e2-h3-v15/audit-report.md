# NB11 E2/h=3 Version 15 Terminal Audit

## Scope and Submission

- Kernel: `alexhuaracha/11-lstm-multihorizon-h3`
- Explicit assigned version: `15`
- Scope: E2, h=3, `HeadwayLSTM` only
- Frozen E2 configuration: `(hidden_size=32, num_layers=1, dropout=0.0, lr=5e-4)`
- Terminal status: `ERROR`
- Retry: prohibited; none submitted.

## Pre-Submission Gates

| Gate | Result |
|---|---|
| h3 metadata contains `alexhuaracha/02-eda-corridors` | Pass |
| h3 manifest declares `02-eda-corridors/atypical_days.csv` | Pass |
| h3 atypical-days SHA-256 equals `2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86` | Pass |
| Frozen E2 winner remains `(32, 1, 0.0, 5e-4)` | Pass |
| Generator parity and focused source tests | Pass (`70 passed`) |
| Explicit user authorization and replacement rationale | Recorded in immutable ledger amendment |

## Terminal Failure

Kaggle stopped in the fail-closed manifest gate before training with:

```text
ValueError: Missing declared input path: headways_E2.parquet (04-preprocessing/headways_E2.parquet)
```

The complete relevant remote-log excerpt is preserved in `kaggle-error-log.md`.
The runtime sidecar and output artifacts were not produced; outputs were not
downloaded.

## Audit and Promotion

| Gate | Result |
|---|---|
| Source checksums | Pass (local submission source) |
| Declared input contract | Pass (local source contract) |
| Runtime sidecar / observed input hashes | Unavailable; fail closed |
| Residual/results inventory and checksums | Unavailable; no download |
| Six-field primary-key uniqueness | Unavailable; no residual export |
| Keyed direct-horizon reconstruction | Unavailable; no residual export |

**Validation status**: `failed`  
**Promotion**: `quarantined`  
**Production action**: none

No production outputs, residuals, paper claims, or Phase 5/9/10 artifacts were
changed.
