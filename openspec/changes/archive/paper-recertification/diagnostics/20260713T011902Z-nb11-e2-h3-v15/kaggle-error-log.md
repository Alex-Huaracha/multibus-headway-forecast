# Kaggle Terminal Error Log — NB11 E2/h=3 Version 15

## Retrieval

- Retrieved at: `2026-07-13T01:19:02Z`
- Command: `uv run kaggle kernels logs alexhuaracha/11-lstm-multihorizon-h3`
- Assigned version: `15`
- Terminal status: `ERROR`

## Relevant Remote Log Excerpt

```text
Output dir: /kaggle/working
Horizon:    3
Device:     cuda

papermill.exceptions.PapermillExecutionError:
Exception encountered at "In [10]":

ValueError: Missing declared input path: headways_E2.parquet (04-preprocessing/headways_E2.parquet)
```

The failure occurred in `_resolve_closed_input_manifest` before any training-input
read, split/window construction, or model training call. No retry was submitted.
