"""Execute the project notebooks and report failures usefully.

Static tests catch renamed symbols and syntax errors. Only execution catches a
cell that raises, and the notebooks need network access to NDBC, so this runs
in CI rather than locally.

Written as a script rather than inline workflow YAML for two reasons: it can be
run by hand the same way CI runs it, and it can report *which cell* failed with
its source. ``jupyter nbconvert`` prints a traceback without the cell that
caused it, which turns a two-minute fix into a hunt.

(An earlier version of the workflow invoked ``python -m nbclient``. That package
ships no ``__main__``, so all three notebooks "failed" in under a second without
a kernel ever starting.)

Usage::

    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py --output-dir executed --timeout 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TIMEOUT = 1200


def execute_notebook(path: Path, output_dir: Path, timeout: int) -> tuple[bool, str]:
    """Run one notebook. Returns (succeeded, message).

    The executed copy is written even on failure, so the partial outputs up to
    the failing cell are available for inspection.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        # Run with the repo root as cwd so the notebooks' path bootstrap and
        # any relative cache directories behave as they do locally.
        resources={"metadata": {"path": str(Path.cwd())}},
    )

    try:
        client.execute()
        return True, "ok"
    except CellExecutionError as e:
        return False, _describe_failure(notebook, e)
    except Exception as e:  # kernel death, timeout, anything else
        return False, f"{type(e).__name__}: {e}"
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_dir / path.name)


def _describe_failure(notebook, error) -> str:
    """Find the cell that raised and show its source alongside the traceback."""
    lines = [str(error)]

    for i, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                source = "".join(cell["source"])
                lines = [
                    f"--- failing cell (index {i}) ---",
                    source,
                    "--- traceback ---",
                    "\n".join(output.get("traceback", [])),
                ]
                return "\n".join(lines)

    return "\n".join(lines)


def print_outputs(path: Path, max_chars: int = 6000) -> None:
    """Dump a notebook's cell outputs to stdout.

    Executed notebooks are uploaded as artifacts, but artifact storage lives on
    a host that some egress policies block - including the one this project is
    developed under, where every download route redirects to a denied domain.
    The job log is readable in that situation and the artifact is not, so the
    numbers are printed here as well as stored.

    Images are skipped; only text output is useful in a log.
    """
    import nbformat

    notebook = nbformat.read(path, as_version=4)

    print("\n" + "#" * 78)
    print(f"# OUTPUTS: {path.name}")
    print("#" * 78)

    for i, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue

        chunks = []
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                chunks.append("".join(output.get("text", [])))
            elif kind == "execute_result":
                chunks.append("".join(output.get("data", {}).get("text/plain", [])))
            elif kind == "error":
                chunks.append("\n".join(output.get("traceback", [])))

        text = "".join(chunks).strip()
        if not text:
            continue

        print(f"\n----- cell {i} -----")
        print(text[:max_chars])
        if len(text) > max_chars:
            print(f"... [{len(text) - max_chars} more characters truncated]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-dir", default="notebooks")
    parser.add_argument("--output-dir", default="executed")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--print-outputs",
        action="store_true",
        help="Also print each notebook's cell outputs to stdout, so the "
        "results are readable from a job log when artifact storage is blocked.",
    )
    args = parser.parse_args(argv)

    notebook_dir = Path(args.notebook_dir)
    output_dir = Path(args.output_dir)
    notebooks = sorted(notebook_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        return 1

    results = []
    for path in notebooks:
        print("=" * 78, flush=True)
        print(f"Executing {path}", flush=True)
        print("=" * 78, flush=True)

        ok, message = execute_notebook(path, output_dir, args.timeout)
        results.append((path.name, ok))

        if ok:
            print(f"OK   {path.name}", flush=True)
        else:
            # Every notebook is attempted even after one fails, so a single
            # broken cell does not hide the state of the others.
            print(f"FAIL {path.name}\n{message}\n", flush=True)

    if args.print_outputs:
        for path in notebooks:
            executed = output_dir / path.name
            if executed.exists():
                print_outputs(executed)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} of {len(results)} notebooks failed: {failed}")
        return 1

    print(f"\nAll {len(results)} notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
