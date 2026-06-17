#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the token-scaling chart (Tufte style) from the scaling tiers.

Reads each scaling tier's summary.csv and plots, on log-log axes, the per-query
context cost of full-context replay (which grows with the memory size) against
Artesian (which stays flat). Output: benchmarks/results/scaling.svg.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = Path(__file__).resolve().parents[1]
TIERS = ["xl-corpus", "session-corpus", "mid-corpus", "mega-corpus"]

RC = {
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "font.size": 12,
    "figure.facecolor": "#fffff8",
    "axes.facecolor": "#fffff8",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#666666",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.facecolor": "#fffff8",
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
}

FULL = "#999999"     # full replay: muted gray (the cost we beat)
MDOKF = "#e1812c"    # md/OKF index-first: middle ground
ARTESIAN = "#4e79a7"   # Artesian: accent


def human(v: float) -> str:
    return f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")


def load():
    pts = []
    for tier in TIERS:
        rows = {r["arm"]: r for r in csv.DictReader(open(BENCH / "results" / tier / "summary.csv"))}
        pts.append((float(rows["A-full-replay"]["mean_total_tokens"]),
                    float(rows["E-md-okf-index-first"]["mean_total_tokens"]),
                    float(rows["B-default-artesian"]["mean_total_tokens"])))
    return sorted(pts)


def main():
    plt.rcParams.update(RC)
    pts = load()
    x = [f for f, _, _ in pts]
    full = [f for f, _, _ in pts]
    mdokf = [m for _, m, _ in pts]
    artesian = [b for _, _, b in pts]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x, full, color=FULL, lw=1.6, marker="o", ms=4)
    ax.plot(x, mdokf, color=MDOKF, lw=1.8, marker="o", ms=4)
    ax.plot(x, artesian, color=ARTESIAN, lw=2.0, marker="o", ms=4)

    ax.set_xlim(x[0] * 0.7, x[-1] * 2.2)
    ax.set_ylim(min(artesian) * 0.5, max(full) * 2)
    ax.spines["bottom"].set_bounds(x[0], x[-1])
    ax.spines["left"].set_bounds(min(artesian), max(full))
    ax.tick_params(direction="in", length=3, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([human(v) for v in x])
    ax.set_yticks([1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["1k", "10k", "100k", "1M"])
    ax.set_xlabel("Durable memory / conversation history (tokens)", color="#666666")
    ax.set_ylabel("Tokens per query", color="#666666")

    # direct labels (no legend)
    ax.annotate("Full-context replay", xy=(x[-1], full[-1]), xytext=(10, -2),
                textcoords="offset points", color=FULL, va="center", fontsize=12)
    ax.annotate("md / OKF index-first", xy=(x[-1], mdokf[-1]), xytext=(10, 0),
                textcoords="offset points", color=MDOKF, va="center", fontsize=12)
    ax.annotate("Artesian (memory.context)", xy=(x[-1], artesian[-1]), xytext=(10, 0),
                textcoords="offset points", color=ARTESIAN, va="center", fontsize=12)

    # annotate Artesian's saving vs full replay at each point
    for fx, _m, b in pts:
        ax.annotate(f"{100*(fx-b)/fx:.1f}% less", xy=(fx, b), xytext=(0, -16),
                    textcoords="offset points", color="#666666", ha="center",
                    fontsize=10, fontstyle="italic")

    fig.text(0.09, 0.97, "One query stays ~1,000 tokens, however large the memory",
             fontsize=17, fontfamily="serif", color="#111111")
    fig.text(0.09, 0.925,
             "Full replay and a markdown/OKF index both grow with the history; only Artesian stays flat (log–log)",
             fontsize=12, fontfamily="serif", color="#666666")

    plt.subplots_adjust(top=0.86, left=0.09, right=0.78, bottom=0.13)
    svg = BENCH / "results" / "scaling.svg"
    png = BENCH / "results" / "scaling.png"
    fig.savefig(svg, format="svg")
    fig.savefig(png, format="png", dpi=200)
    print(f"wrote {svg} and {png}  points={[(human(f), round(m), round(b)) for f, m, b in pts]}")


if __name__ == "__main__":
    main()
