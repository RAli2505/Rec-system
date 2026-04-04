from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import contextmanager
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

import jupyter_client.connect as jc_connect
import jupyter_core.paths as jc_paths


@contextmanager
def _insecure_secure_write(fname: str, binary: bool = False):
    """Fallback for Windows environments where ACL hardening is denied."""
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    with open(path, mode, encoding=encoding) as handle:
        yield handle


def _patch_jupyter_secure_write() -> None:
    jc_paths.win32_restrict_file_to_user = lambda fname: None
    jc_paths.secure_write = _insecure_secure_write
    jc_connect.secure_write = _insecure_secure_write


def execute_notebook(notebook_path: Path, timeout: int, kernel_name: str) -> None:
    _patch_jupyter_secure_write()
    project_root = notebook_path.parent.parent
    runtime_dir = project_root / ".jupyter_runtime"
    ipython_dir = project_root / ".ipython"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ipython_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["IPYTHONDIR"] = str(ipython_dir)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    with notebook_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
        allow_errors=False,
    )

    try:
        client.execute()
    except CellExecutionError:
        with notebook_path.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        raise

    with notebook_path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a notebook in place.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()

    execute_notebook(args.notebook.resolve(), args.timeout, args.kernel_name)


if __name__ == "__main__":
    main()
