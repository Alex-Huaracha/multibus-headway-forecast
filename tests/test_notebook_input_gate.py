"""Input-gate guard for the 6 DL notebook families (paper recertification).

Guards the frozen training-input contract on the GENERATED notebooks, so a
stale regeneration cannot silently reintroduce unverified inputs:

  1. Every DL notebook pins the frozen SHA-256 of each required training input
     (INPUT_HASHES) and resolves inputs through the hash-verifying
     ``_resolve_input`` helper — a run stops before training when a required
     file is missing or its bytes differ from the frozen snapshot.
  2. ``atypical_days.csv`` is a REQUIRED input: the silent fallback to
     ``atypical_flag=0`` (which left the feature inert in every previous
     Kaggle run) must not appear in any generated notebook.
  3. ``kernel-metadata.json`` declares ``alexhuaracha/02-eda-corridors`` as a
     kernel source, so atypical_days.csv actually mounts under /kaggle/input.

Frozen hashes were computed from the pinned Kaggle kernel outputs
(alexhuaracha/04-preprocessing, alexhuaracha/16-e4-data-baselines,
alexhuaracha/02-eda-corridors) and cross-checked against local copies.
"""
import json
from pathlib import Path

import nbformat as nbf
import pytest

ROOT = Path(__file__).resolve().parent.parent

ATYPICAL_SHA256 = "2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86"
HEADWAYS_E2_SHA256 = "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4"
HEADWAYS_E59_SHA256 = "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448"
HEADWAYS_E4_SHA256 = "1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273"

E2_E59_HASHES = (HEADWAYS_E2_SHA256, HEADWAYS_E59_SHA256, ATYPICAL_SHA256)
E4_HASHES = (HEADWAYS_E4_SHA256, ATYPICAL_SHA256)

# (family dir, notebook filename template, required frozen hashes)
FAMILIES = [
    ("11_lstm_multihorizon", "11_lstm_h{h}.ipynb", E2_E59_HASHES),
    ("12_spatial_conv_lstm_multihorizon", "12_spatial_conv_lstm_h{h}.ipynb", E2_E59_HASHES),
    ("13_spatial_transformer_multihorizon", "13_spatial_transformer_h{h}.ipynb", E2_E59_HASHES),
    ("17_e4_lstm", "17_e4_lstm_h{h}.ipynb", E4_HASHES),
    ("18_e4_convlstm", "18_e4_convlstm_h{h}.ipynb", E4_HASHES),
    ("19_e4_transformer", "19_e4_transformer_h{h}.ipynb", E4_HASHES),
]
HORIZONS = [1, 3, 5, 10]

CASES = [
    pytest.param(family, template.format(h=h), hashes, h, id=f"{family}-h{h}")
    for family, template, hashes in FAMILIES
    for h in HORIZONS
]


def _notebook_code(nb_path: Path) -> str:
    nb = nbf.read(nb_path, as_version=4)
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


@pytest.mark.parametrize("family, filename, hashes, horizon", CASES)
class TestNotebookInputGate:
    def _nb_dir(self, family: str, horizon: int) -> Path:
        nb_dir = ROOT / "notebooks" / family / f"h{horizon}"
        if not nb_dir.exists():
            pytest.skip(f"family dir absent: {nb_dir}")
        return nb_dir

    def test_inputs_are_hash_pinned(self, family, filename, hashes, horizon):
        """Notebook pins frozen SHA-256 hashes and resolves via _resolve_input."""
        src = _notebook_code(self._nb_dir(family, horizon) / filename)
        assert "INPUT_HASHES" in src, "setup cell must pin frozen input hashes"
        assert "_resolve_input(" in src, "inputs must resolve through the hash gate"
        for digest in hashes:
            assert digest in src, f"frozen hash missing from notebook: {digest[:12]}…"

    def test_atypical_days_is_required(self, family, filename, hashes, horizon):
        """No silent fallback: atypical_days.csv resolves through the hash gate."""
        src = _notebook_code(self._nb_dir(family, horizon) / filename)
        assert 'atypical_path = _resolve_input("atypical_days.csv")' in src
        assert "atypical_path = None" not in src, (
            "silent atypical fallback must not survive regeneration"
        )
        assert "if not atypical_dates:" in src, (
            "notebook must hard-fail when the atypical set parses empty"
        )

    def test_kernel_metadata_declares_eda_source(self, family, filename, hashes, horizon):
        """02-eda-corridors must be a kernel source so atypical_days.csv mounts."""
        meta_path = self._nb_dir(family, horizon) / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "alexhuaracha/02-eda-corridors" in meta["kernel_sources"], (
            f"{meta_path}: kernel_sources must include alexhuaracha/02-eda-corridors, "
            f"got {meta['kernel_sources']!r}"
        )


def _extract_setup_namespace(nb_path: Path) -> dict:
    """Exec the emitted setup cell (minus torch/DEVICE lines) in a fresh namespace."""
    nb = nbf.read(nb_path, as_version=4)
    src = next(
        c.source for c in nb.cells
        if c.cell_type == "code" and "INPUT_HASHES" in c.source
    )
    src = "\n".join(
        line for line in src.splitlines()
        if not line.startswith("DEVICE") and "Device:" not in line
    )
    namespace: dict = {}
    exec(src, namespace)
    return namespace


class TestResolverBehavior:
    """Behavior tests for the EMITTED _resolve_input — not its textual shape.

    A regression that keeps the source looking right but disables the gate
    (flipped hash comparison, swallowed error, unconditional candidates[0])
    must fail here."""

    REPRESENTATIVES = [
        ROOT / "notebooks" / "11_lstm_multihorizon" / "h3" / "11_lstm_h3.ipynb",
        ROOT / "notebooks" / "17_e4_lstm" / "h3" / "17_e4_lstm_h3.ipynb",
    ]

    @pytest.fixture(params=REPRESENTATIVES, ids=["nb11", "nb17"])
    def resolver(self, request, tmp_path, monkeypatch):
        namespace = _extract_setup_namespace(request.param)
        monkeypatch.chdir(tmp_path)
        # Inject a synthetic frozen hash so the behavior test needs no real data.
        good_bytes = b"synthetic frozen input"
        namespace["INPUT_HASHES"]["probe.bin"] = __import__("hashlib").sha256(good_bytes).hexdigest()
        return namespace["_resolve_input"], tmp_path, good_bytes

    def test_missing_input_fails_closed(self, resolver):
        resolve, _tmp, _good = resolver
        with pytest.raises(FileNotFoundError):
            resolve("probe.bin")

    def test_tampered_input_fails_closed(self, resolver):
        resolve, tmp, _good = resolver
        (tmp / "a").mkdir()
        (tmp / "a" / "probe.bin").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            resolve("probe.bin")

    def test_matching_copy_wins_among_duplicates(self, resolver):
        resolve, tmp, good = resolver
        (tmp / "bad").mkdir()
        (tmp / "bad" / "probe.bin").write_bytes(b"tampered")
        (tmp / "ok").mkdir()
        (tmp / "ok" / "probe.bin").write_bytes(good)
        assert Path(resolve("probe.bin")).resolve() == (tmp / "ok" / "probe.bin").resolve()


def test_nb12_h10_keeps_replacement_slug():
    """NB12 h10 must keep the h10b slug — the original h10 kernel is corrupt
    on Kaggle (see commit e0757b6); regenerating with plain h10 would push to
    the corrupt kernel instead of versioning the replacement."""
    meta_path = ROOT / "notebooks" / "12_spatial_conv_lstm_multihorizon" / "h10" / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == "alexhuaracha/12-spatialconvlstm-multihorizon-h10b"
    assert meta["title"] == "12 SpatialConvLSTM Multihorizon h10b"
