"""The tutorials must keep running against the current code (executed, not just parsed)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")

TUTORIALS = sorted((Path(__file__).parent.parent / "tutorials").glob("*.ipynb"))


@pytest.mark.parametrize("path", TUTORIALS, ids=[p.stem for p in TUTORIALS])
def test_tutorial_runs(path):
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                                     resources={"metadata": {"path": str(path.parent)}})
    client.execute()  # raises CellExecutionError on the first failing cell
