"""Config and data loading. Unit conversion from lab units happens here and in
the analysis scripts, never inside the models.

Configs are TOML. Any value still equal to the string "MEASURE" makes loading
fail with the list of missing entries, so an analysis can never run on a
placeholder.
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import numpy as np

PLACEHOLDER = "MEASURE"


def _find_placeholders(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _find_placeholders(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_placeholders(v, f"{path}[{i}]")
    elif obj == PLACEHOLDER:
        yield path


def load_config(path):
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    missing = list(_find_placeholders(cfg))
    if missing:
        raise ValueError(f"{path}: fill in the measured values first:\n  " + "\n  ".join(missing))
    cfg["_path"] = str(path)
    return cfg


def read_table(path, required):
    """Read a CSV with a header row (lines starting with '#' are comments).

    Returns a dict of float arrays for the required columns. Missing columns,
    empty cells and non-numeric cells raise.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(line for line in f if not line.lstrip().startswith("#"))]
    header = [h.strip() for h in rows[0]]
    for col in required:
        if col not in header:
            raise KeyError(f"{path}: column {col!r} missing (have {header})")
    data = {}
    for col in required:
        j = header.index(col)
        vals = []
        for i, r in enumerate(rows[1:], start=2):
            cell = r[j].strip() if j < len(r) else ""
            if cell == "":
                raise ValueError(f"{path}: empty value in column {col!r}, data row {i - 1}")
            vals.append(float(cell))
        data[col] = np.array(vals)
    return data


def check_time_increasing(t, name):
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: time column must be strictly increasing")


def write_table(path, columns, header_comment):
    """Write equal-length arrays as a CSV with a '#' comment line on top."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    n = len(columns[names[0]])
    assert all(len(columns[c]) == n for c in names)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {header_comment}\n")
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([f"{columns[c][i]:.6g}" for c in names])
