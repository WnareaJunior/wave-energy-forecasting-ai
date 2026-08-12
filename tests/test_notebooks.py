"""Static checks on the notebooks.

These do not execute anything - the notebooks need network access to NDBC, so
end-to-end execution happens in the `notebooks` GitHub Actions workflow. What
these catch is the class of breakage that comes from refactoring library code
and forgetting the notebooks call it: a renamed function, a moved module, a
deleted symbol.

That failure mode is easy to miss because a broken notebook does not fail any
test, does not fail CI, and is only discovered when someone opens it.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def code_cells(path: Path):
    notebook = json.loads(path.read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebook:
    def test_is_valid_json_with_cells(self, path):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["cells"]

    def test_every_cell_has_an_id(self, path):
        """Required by nbformat 4.5+; missing ids become a hard error."""
        notebook = json.loads(path.read_text())
        assert all(cell.get("id") for cell in notebook["cells"])

    def test_no_stored_outputs(self, path):
        """Outputs bloat diffs and leak absolute paths from whoever ran them."""
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), f"cell {cell['id']} has stored output"

    def test_all_code_parses(self, path):
        for i, source in enumerate(code_cells(path)):
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"cell {i} does not parse: {e}")

    def test_every_src_import_resolves(self, path):
        """The regression guard: a renamed library symbol breaks this test
        rather than being discovered when someone opens the notebook."""
        unresolved = []
        for source in code_cells(path):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("src."):
                    continue
                module = importlib.import_module(node.module)
                for alias in node.names:
                    if not hasattr(module, alias.name):
                        unresolved.append(f"{node.module}.{alias.name}")
        assert not unresolved, f"missing symbols: {unresolved}"

    def test_has_narrative(self, path):
        """A notebook that is only code belongs in a script."""
        notebook = json.loads(path.read_text())
        markdown = [c for c in notebook["cells"] if c["cell_type"] == "markdown"]
        assert len(markdown) >= 3


def test_no_stale_double_extension():
    """The original analysis notebook was named `...ipynb.ipynb`."""
    assert not list(NOTEBOOK_DIR.glob("*.ipynb.ipynb"))


def test_notebooks_do_not_reimplement_the_physics():
    """Analysis logic belongs in tested modules, not in cells.

    The original notebook inlined `0.49 * H**2 * T` with the wrong period
    variable, and the error survived precisely because nothing imported it and
    nothing tested it.
    """
    offenders = []
    for path in NOTEBOOKS:
        for source in code_cells(path):
            if "0.49" in source and "**2" in source:
                offenders.append(path.name)
    assert not offenders, (
        f"{offenders} inline the power-flux formula; call "
        "src.processing.wave_power_flux.calculate_wave_power_flux instead"
    )
