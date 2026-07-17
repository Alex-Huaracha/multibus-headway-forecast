# NB11 E2/h=3 Kaggle Completion Audit

**Audit mode:** read-only remote verification. No kernel was pushed, no production
residual was overwritten, and no task checkbox or source file was changed.

## Verdict

**completed but insufficient provenance**

Kaggle kernel version 6 completed and produced E2/h=3 residuals, using source
semantically identical to the expected regenerated NB11 h3 notebook and the
frozen LSTM winner configuration. It is **not a validated paper result**: the
downloaded residual CSV has no immutable `(corridor, direction, pair_rank,
input_end_t, target_t)` window identity keys, and the upstream kernel inputs are
unversioned. The old production residual remains untouched.

## Remote identity and completion evidence

| Field | Evidence |
|---|---|
| Kernel | `alexhuaracha/11-lstm-multihorizon-h3` |
| Latest completed version | `6` (`KernelWorkerStatus.COMPLETE`) |
| Latest run time (Kaggle list, UTC) | `2026-07-12 15:42:30.540000` |
| Kernel numeric identifier | `123170788` |
| Pulled metadata | `remote-version-6-source/kernel-metadata.json` |
| Container image | `gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c` |
| Log completion evidence | Results at 1057 s; residual export at 1058 s; notebook/HTML conversion ends at 1067.8 s |

The latest pull and explicit `/6` pull have identical notebook SHA-256 values,
so the downloaded source is specifically version 6, not merely an unpinned
latest alias.

## GPU evidence: requested is not assigned

| Layer | Value | Interpretation |
|---|---|---|
| Expected local metadata | `accelerator: GPU_T4X2` | Push-time request/configuration only. |
| Pulled remote metadata | `machine_shape: NvidiaTeslaT4`, `enable_gpu: true` | Remote metadata selection, not a runtime device attestation. |
| Runtime log | `Device: cuda`; E2 and E59 training completed | Proves CUDA was available and usable. |
| Runtime GPU SKU | Not emitted | Cannot be determined from available metadata/logs. |

There is no `AcceleratorError`, CUDA compatibility failure, or GPU-selection
failure in the completed version-6 log. The available evidence therefore does
not substantiate a push-time GPU-selection error for this completed run; it
also does not prove the exact runtime GPU SKU. Do not infer a Tesla P100, T4X2,
or T4 assignment from the request/metadata alone.

## Source and configuration comparison

Raw notebook files differ only because Kaggle serializes cell `source` values
as strings while the local generated notebook serializes them as line arrays.
After normalizing every cell source to text, both the all-cell and code-cell
SHA-256 digests match exactly:

| Comparison | SHA-256 | Match |
|---|---:|---|
| Local raw notebook | `04a59cab99b6949fa1601ea3880a05d7374ac5e9ee23e52c912d2b59186fe0d5` | — |
| Remote raw notebook | `b43c91bc19f65007d1d62771940084bca9d55f5d905211334bbf4bd1856b7155` | No (serialization only) |
| Normalized all cell sources | `624854b37254930f12c5785ba0a63d45d335b9fd77a534586d5f555e37ee3293` | Yes |
| Normalized code-cell sources | `ee440805d5aaa09c65ea9097a61607c875daaf2a825a40b15df2bd58366567a1` | Yes |

Verified contract:

- Model: `HeadwayLSTM` / LSTM.
- Horizon: `HORIZON = 3`.
- Corridor E2 winner: `hidden=32`, `layers=1`, `dropout=0.0`, `lr=0.0005`.
- E59 winner also matches the frozen configuration: `hidden=32`, `layers=2`,
  `dropout=0.2`, `lr=0.0005`.
- Preprocessing: temporal split precedes `winsorize_train_p99(df_split)`;
  logs show E2 train-p99 `28.4679` minutes and the expected split counts
  (train `1,127,043`, val `240,960`, test `222,656`).
- The completed log reports E2 h=3 DL MAE `4.8580`, persistence MAE `5.7622`,
  `n=599,117`.

## Inputs and provenance

Pulled source metadata lists kernel sources
`alexhuaracha/04-preprocessing` and
`alexhuaracha/10-baselines-multi-horizonte`; the log identifies the baseline
file path under the latter. Neither source is pinned to a Kaggle version in
the metadata/log. No dataset source/version identifier is recorded.

The log supplies useful provenance signals (split counts, train-p99 thresholds,
normalization statistics, context columns, window counts, frozen winner
hyperparameters, and output row counts), but they cannot replace immutable
per-window residual keys or versioned upstream inputs.

## Download inventory

All remote downloads are contained under this new diagnostic directory.

| File | Bytes | SHA-256 |
|---|---:|---|
| `remote-output/11-lstm-multihorizon-h3.log` | 19,652 | `98df04ca801167510785acf124eaf699093660df56b78bb3d3c00cccd8137645` |
| `remote-output/lstm_residuals_h3.csv` | 175,675,703 | `e77dd15c0f4f9e8bf8e7c9a4ff9def8af4ce60e757cc86bdb1190d73e8a61d8b` |
| `remote-output/lstm_results_h3.csv` | 510 | `78fdb758d2cdb486d2783d83e9d7bfac2294485f4ff23652c5389db3b84a064a` |
| `remote-version-6-source/11-lstm-multihorizon-h3.ipynb` | 99,077 | `b43c91bc19f65007d1d62771940084bca9d55f5d905211334bbf4bd1856b7155` |
| `remote-version-6-source/kernel-metadata.json` | 688 | `c6170f6971f5643270a48bf0823ae13dfd6ac67c67eb1735e5de6c3872e5c570` |

`remote-source/` is a duplicate unversioned-latest pull whose two checksums
are identical to the explicit version-6 source files above.

## Residual identity check

`lstm_residuals_h3.csv` has 2,768,950 data rows and exactly these columns:

```text
corridor,direction,horizon,y_true,y_pred_dl,y_pred_persist
```

It contains `corridor` and `direction`, but does **not** contain `pair_rank`,
`input_end_t`, or `target_t`. It therefore cannot prove immutable window
alignment and must not replace or be merged into a production residual path.

## Commands executed

```bash
uv run kaggle kernels status alexhuaracha/11-lstm-multihorizon-h3
uv run kaggle kernels logs alexhuaracha/11-lstm-multihorizon-h3
uv run kaggle kernels pull alexhuaracha/11-lstm-multihorizon-h3 --path <diagnostic>/remote-source --metadata
uv run kaggle kernels output alexhuaracha/11-lstm-multihorizon-h3 --path <diagnostic>/remote-output --force
uv run kaggle kernels list --user alexhuaracha --search "11-lstm-multihorizon-h3" --page-size 20 --sort-by dateRun -v
uv run kaggle kernels status alexhuaracha/11-lstm-multihorizon-h3/6
uv run kaggle kernels logs alexhuaracha/11-lstm-multihorizon-h3/6
uv run kaggle kernels pull alexhuaracha/11-lstm-multihorizon-h3/6 --path <diagnostic>/remote-version-6-source --metadata
```

The installed Kaggle CLI is 2.1.2. Its `status` and `logs` help expose no
version-history option; explicit `/6` status/pull establishes the completed
version audited here. No push command was executed during this audit.
