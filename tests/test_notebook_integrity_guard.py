"""Artifact-level integrity guard for recertified generated notebooks."""
from __future__ import annotations

from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
RECERTIFIED_FAMILIES = (11, 12, 13, 17, 18, 19)
PROHIBITED_PATTERNS = (
    "non_train",
    "winsorize_train_p99(train_df)",
    "pl.concat([df_winsor, non_train])",
)


@pytest.mark.parametrize("family", RECERTIFIED_FAMILIES)
def test_recertified_notebooks_do_not_contain_stale_winsorization(family: int) -> None:
    """NGI2/NGI3: scan each present recertified family at the shipped artifact level."""
    family_dirs = sorted(NOTEBOOKS_DIR.glob(f"{family}_*"))
    if not family_dirs:
        pytest.skip(f"Notebook family {family} is absent")

    violations: list[str] = []
    for family_dir in family_dirs:
        for notebook_path in sorted(family_dir.rglob("*.ipynb")):
            with notebook_path.open(encoding="utf-8") as notebook_file:
                notebook = nbformat.read(notebook_file, as_version=4)
            for cell_index, cell in enumerate(notebook.cells):
                for prohibited_pattern in PROHIBITED_PATTERNS:
                    if prohibited_pattern in cell.source:
                        violations.append(
                            f"{prohibited_pattern!r}: {notebook_path}: cell {cell_index}"
                        )

    assert not violations, (
        "Prohibited stale winsorization pattern found in notebook cells:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "prohibited_pattern",
    (
        "non_train",
        "winsorize_train_p99(train_df)",
        "pl.concat([df_winsor, non_train])",
    ),
)
def test_guard_reports_path_and_zero_based_cell_for_each_stale_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prohibited_pattern: str
) -> None:
    """D3: every prohibited stale form reports its shipped-artifact location."""
    notebook_path = tmp_path / "11_lstm_multihorizon" / "h1" / "stale.ipynb"
    notebook_path.parent.mkdir(parents=True)
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(prohibited_pattern)]),
        notebook_path,
    )
    monkeypatch.setitem(globals(), "NOTEBOOKS_DIR", tmp_path)

    with pytest.raises(AssertionError) as error:
        test_recertified_notebooks_do_not_contain_stale_winsorization(11)

    assert prohibited_pattern in str(error.value)
    assert f"{notebook_path}: cell 0" in str(error.value)


def test_guard_skips_an_absent_recertified_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NGI3: fresh checkouts without a family remain runnable."""
    monkeypatch.setitem(globals(), "NOTEBOOKS_DIR", tmp_path)

    with pytest.raises(pytest.skip.Exception, match="Notebook family 11 is absent"):
        test_recertified_notebooks_do_not_contain_stale_winsorization(11)
