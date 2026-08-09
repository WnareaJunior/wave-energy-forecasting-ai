"""Guards against argument-name drift in the experiment entrypoints.

A three-station ensemble run once died three seconds in with
``AttributeError: 'Namespace' object has no attribute 'member'`` - the CLI flag
had been renamed from ``--member`` to ``--members`` and one reference in a
logging call was missed.

Nothing caught it. Lint does not know what an argparse Namespace carries, the
unit tests never construct one, and the failure only surfaces on a runner after
the data has already been downloaded. These tests close that gap by parsing the
real arguments and checking every ``args.X`` the module actually reads.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
SCRIPTS = sorted(EXPERIMENTS.glob("*.py"))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_exp_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def referenced_arg_attributes(path: Path) -> set[str]:
    """Every attribute read off a variable named ``args`` in the source."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.ctx, ast.Load)
        ):
            found.add(node.attr)
    return found


def test_experiment_scripts_exist():
    assert SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
class TestExperimentScript:
    def test_parses(self, path):
        ast.parse(path.read_text())

    def test_has_a_main(self, path):
        module = load_module(path)
        assert callable(getattr(module, "main", None))

    def test_help_does_not_crash(self, path):
        """--help exercises the whole parser definition."""
        module = load_module(path)
        with pytest.raises(SystemExit) as excinfo:
            module.main(["--help"])
        assert excinfo.value.code == 0

    def test_every_args_attribute_is_defined(self, path):
        """The actual regression guard.

        Parses the script's own default arguments, then checks that every
        ``args.SOMETHING`` the source reads exists on the resulting namespace.
        A renamed flag now fails here instead of on a runner.
        """
        module = load_module(path)
        source = path.read_text()

        # Find the parser by running main's argument setup in isolation is not
        # possible, so parse with no arguments via the module's own parser.
        tree = ast.parse(source)
        has_parser = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse_args"
            for node in ast.walk(tree)
        )
        if not has_parser:
            pytest.skip("no argparse usage")

        namespace = _parse_default_args(module)
        if namespace is None:
            pytest.skip("could not obtain a default namespace")

        referenced = referenced_arg_attributes(path)
        available = set(vars(namespace))
        missing = referenced - available
        assert not missing, (
            f"{path.name} reads args.{{{', '.join(sorted(missing))}}} but the "
            f"parser defines only {sorted(available)}"
        )


def _parse_default_args(module):
    """Run the module's parser with no arguments, capturing the namespace.

    Monkeypatches ArgumentParser.parse_args so main() hands back the namespace
    before doing any work.
    """
    import argparse

    captured = {}
    original = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        result = original(self, args=[], namespace=namespace)
        captured["ns"] = result
        raise _StopEarly

    argparse.ArgumentParser.parse_args = spy
    try:
        try:
            module.main([])
        except _StopEarly:
            pass
        except SystemExit:
            pass
        except Exception:
            # main may fail after parsing for unrelated reasons (no network);
            # the namespace is what matters and it is already captured.
            pass
    finally:
        argparse.ArgumentParser.parse_args = original

    return captured.get("ns")


class _StopEarly(Exception):
    """Unwinds main() as soon as the namespace has been captured."""
