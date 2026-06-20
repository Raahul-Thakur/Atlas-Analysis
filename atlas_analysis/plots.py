"""Plotting and histogram table exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogram_table(values, weights=None, bins=50, range=None) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights[finite]
    raw_counts, edges = np.histogram(values, bins=bins, range=range)
    weighted_counts, _ = np.histogram(values, bins=edges, weights=weights) if weights is not None else (raw_counts, edges)
    return pd.DataFrame(
        {
            "bin_low": edges[:-1],
            "bin_high": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "raw_count": raw_counts,
            "weighted_count": weighted_counts,
        }
    )


def save_histogram(values, output_path: Path, title: str, xlabel: str, weights=None, bins=50, range=None, label="Events"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.hist(values, bins=bins, range=range, weights=weights, histtype="stepfilled", alpha=0.72, color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    if "Z" in title:
        ax.axvline(91.1876, color="#c43b3b", linestyle="--", linewidth=1.5, label="Z mass")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_overlay_histogram(series, output_path: Path, title: str, xlabel: str, bins=50, range=None, label="Events"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for name, values in series.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, range=range, histtype="step", linewidth=1.8, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    if "Z" in title or "dilepton" in title:
        ax.axvline(91.1876, color="#c43b3b", linestyle="--", linewidth=1.5, label="Z mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
