"""Research notebooks: the committed notebooks match their builder and execute cleanly.

Execution is marked ``slow`` (each notebook generates its own synthetic market; about a minute
each). Run it with ``pytest -m slow`` or as part of the full suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_notebooks", NB_DIR / "build_notebooks.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BUILDER = _builder()
NAMES = sorted(BUILDER.NOTEBOOKS)


@pytest.mark.parametrize("name", NAMES)
def test_committed_notebook_matches_builder_and_ran_cleanly(name: str) -> None:
    committed = nbformat.read(NB_DIR / name, as_version=4)
    fresh = BUILDER.NOTEBOOKS[name]()
    assert [c.source for c in committed.cells] == [c.source for c in fresh.cells], "re-run build_notebooks.py"
    code = [c for c in committed.cells if c.cell_type == "code"]
    assert code and all(c.get("execution_count") for c in code), "committed notebooks are saved executed"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors[0].get("ename") if errors else None


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_notebook_executes(name: str, tmp_path: Path) -> None:
    nb = BUILDER.NOTEBOOKS[name]()
    client = nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    client.execute()  # raises CellExecutionError on the first failing cell
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert outputs and not any(o.get("output_type") == "error" for o in outputs)
