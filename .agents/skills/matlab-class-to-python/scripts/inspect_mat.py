#!/usr/bin/env python3
"""Inspect MATLAB .mat test fixture files and print variable shapes, dtypes, and struct fields.

Usage:
    python inspect_mat.py path/to/fixture.mat
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


def _describe_value(val: Any, indent: int = 2) -> str:
    prefix = " " * indent
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            fields = ", ".join(val.dtype.names)
            return f"struct ndarray shape={val.shape}, fields=[{fields}]"
        if val.dtype == object and val.size > 0:
            first = val.flat[0]
            return f"object ndarray shape={val.shape}, first_elem={type(first).__name__}"
        if np.issubdtype(val.dtype, np.number) and val.size > 0:
            return (
                f"ndarray shape={val.shape}, dtype={val.dtype}, "
                f"min={np.nanmin(val):.6g}, max={np.nanmax(val):.6g}"
            )
        return f"ndarray shape={val.shape}, dtype={val.dtype}"
    if hasattr(val, "_fieldnames"):
        lines = [f"mat_struct fields={val._fieldnames}"]
        for fname in val._fieldnames:
            fval = getattr(val, fname)
            lines.append(f"{prefix}  .{fname}: {_describe_value(fval, indent + 4)}")
        return "\n".join(lines)
    return f"{type(val).__name__}: {repr(val)[:80]}"


def inspect_mat_file(mat_path: Path) -> None:
    if not mat_path.exists():
        print(f"Error: File not found: {mat_path}", file=sys.stderr)
        sys.exit(1)

    data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    print(f"=== {mat_path.name} ===")
    for key, val in sorted(data.items()):
        if key.startswith("__"):
            continue
        print(f"- {key}: {_describe_value(val, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mat.py <path_to_mat_file> [...]", file=sys.stderr)
        sys.exit(1)
    for arg in sys.argv[1:]:
        inspect_mat_file(Path(arg))

