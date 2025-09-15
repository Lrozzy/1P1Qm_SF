"""
Minimal analysis for trash_modes sweep (autoencoder).

- Reads logs in a directory with filenames: dim{dim}_wires{w}_trash{t}
- Prints confirmation when reading each file and whether an AUC was found
- Builds a DataFrame: index=trash_modes, columns=wires, values=AUC
- Plots a heatmap (seaborn if available, otherwise matplotlib)

Usage:
    from analysis_trash_sweep import collect_trash_sweep_matrix, plot_trash_sweep_heatmap
    df = collect_trash_sweep_matrix(dim=4)
    fig, ax = plot_trash_sweep_heatmap(df, title="Dim 4: AUC vs Trash Modes and Wires")
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting deps
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None
import matplotlib.pyplot as plt

# Default directory for trash sweep logs
DEFAULT_TRASH_SWEEP_LOG_DIR = \
    "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_auto_logs/trash_sweep"

_auc_line_patterns = [
    re.compile(r"Final test AUC \(anomaly score\):\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"Final test AUC:\s*([0-9]*\.?[0-9]+)"),
]

# Accept optional trailing suffix (e.g., "_trash2") or extension after the pattern
_fname_regex = re.compile(r"^dim(?P<dim>\d+)_wires(?P<wires>\d+)_trash(?P<trash>\d+).*")


def extract_auc_from_log(log_file: str, *, debug: bool = False) -> Optional[float]:
    """Extract final test AUC from a log file.

    If debug is True, prints a line when reading the file and whether an AUC was found.
    Returns None if not found or file missing.
    """
    if debug:
        print(f"[read] {log_file}")
    if not os.path.exists(log_file):
        if debug:
            print(f"[auc ] {log_file} -> MISSING")
        return None
    try:
        with open(log_file, "r") as f:
            content = f.read()
    except Exception as e:
        if debug:
            print(f"[auc ] {log_file} -> ERROR: {e}")
        return None

    # Search from end for robustness
    lines = content.splitlines()[::-1]
    for line in lines:
        for pat in _auc_line_patterns:
            m = pat.search(line)
            if m:
                try:
                    auc = float(m.group(1))
                    if debug:
                        print(f"[auc ] {log_file} -> {auc}")
                    return auc
                except Exception as e:
                    if debug:
                        print(f"[auc ] {log_file} -> PARSE_ERROR: {e}")
                    return None
    if debug:
        print(f"[auc ] {log_file} -> None")
    return None


def collect_trash_sweep_matrix(
    dim: int,
    log_dir: str = DEFAULT_TRASH_SWEEP_LOG_DIR,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame of AUC values by directly scanning the log_dir.

    - Keeps it simple: for each file matching dim{dim}_wires{w}_trash{t}, read it,
      print what was read, parse AUC (printing the outcome), and populate the matrix.
    - Missing entries are NaN.
    """
    wires_set = set()
    trash_set = set()
    values = {}  # (t, w) -> auc or np.nan
    files_present = set()  # set[(w, t)]
    auc_present = set()    # set[(w, t)] where an AUC was parsed

    if not os.path.isdir(log_dir):
        print(f"[trash_sweep] Log directory not found: {log_dir}")
        return pd.DataFrame()

    for name in sorted(os.listdir(log_dir)):
        if debug:
            print(f"[scan] {name}")
        m = _fname_regex.match(name)
        if not m:
            if debug:
                print(f"[skip] {name} (pattern)")
            continue
        if int(m.group("dim")) != dim:
            if debug:
                print(f"[skip] {name} (dim {m.group('dim')} != {dim})")
            continue
        w = int(m.group("wires"))
        t = int(m.group("trash"))
        if debug:
            print(f"[keep] {name} -> wires={w}, trash={t}")
        wires_set.add(w)
        trash_set.add(t)
        files_present.add((w, t))
        log_file = os.path.join(log_dir, name)
        auc = extract_auc_from_log(log_file, debug=debug)
        values[(t, w)] = auc if auc is not None else np.nan
        if auc is not None:
            auc_present.add((w, t))

    wires = sorted(wires_set)
    trash = sorted(trash_set)

    # Always print a global coverage summary: expected trash 1..w-1 per wire
    total_expected = 0
    total_found = 0
    missing_file_pairs = []  # (w, t)
    missing_auc_pairs = []   # (w, t)
    for w in wires:
        expected = set(range(1, w))
        total_expected += len(expected)
        for t in expected:
            if (w, t) in auc_present:
                total_found += 1
            elif (w, t) in files_present:
                missing_auc_pairs.append((w, t))
            else:
                missing_file_pairs.append((w, t))

    if total_expected > 0:
        print(f"[summary] dim={dim}: {total_found}/{total_expected} runs")
    else:
        print(f"[summary] dim={dim}: 0/0 runs (no expectations)")

    for w, t in missing_file_pairs:
        print(f"\tmissing: wires {w}, trash modes {t} (no file)")
    for w, t in missing_auc_pairs:
        print(f"\tmissing: wires {w}, trash modes {t} (no AUC)")
    df = pd.DataFrame(index=trash, columns=wires, dtype=float)
    for t in trash:
        for w in wires:
            df.loc[t, w] = values.get((t, w), np.nan)

    df.index.name = "trash_modes"
    df.columns.name = "wires"
    return df


def plot_trash_sweep_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Plot a heatmap for the AUC matrix.

    Returns (fig, ax). Notebook-friendly: displays when in interactive backends.
    """
    # Determine figure size: allow override, else scale by matrix dimensions
    if figsize is None:
        default_size = (1.4 * max(6, len(df.columns)) + 3, 1.2 * max(6, len(df.index)) + 3)
        figsize = default_size
    fig, ax = plt.subplots(figsize=figsize)

    if sns is not None:
        sns.heatmap(
            df, ax=ax, cmap=cmap, annot=annotate, fmt=fmt, vmin=vmin, vmax=vmax,
            cbar_kws={"label": "AUC"}
        )
    else:
        # Fallback: matplotlib imshow
        im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("AUC")
        if annotate:
            for (i, j), val in np.ndenumerate(df.values):
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", color="w")

    ax.set_xlabel("Number of wires (n)")
    ax.set_ylabel("Number of trash modes")
    if title:
        ax.set_title(title)
    ax.invert_yaxis()  # Invert y-axis to put lowest trash_modes at the bottom
    fig.tight_layout()
    return fig, ax


def collect_and_plot(dim: int, log_dir: str = DEFAULT_TRASH_SWEEP_LOG_DIR, title: Optional[str] = None):
    """Convenience: collect matrix for a dim and plot heatmap.

    Returns (df, (fig, ax)).
    """
    df = collect_trash_sweep_matrix(dim=dim, log_dir=log_dir)
    if title is None:
        title = f"Dim {dim}: AUC vs Trash Modes and Wires"
    fig, ax = plot_trash_sweep_heatmap(df, title=title)
    return df, (fig, ax)
