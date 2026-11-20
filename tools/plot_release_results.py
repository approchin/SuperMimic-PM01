#!/usr/bin/env python3
"""Regenerate release figures from the public aggregate CSV files.

Requires Python 3 and matplotlib. Run from any directory:
    python tools/plot_release_results.py

This is a plotting entry point. It does not run policy inference, simulation,
training, network requests, or access the publisher's private experiment files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
INK = "#172A3A"
MUTED = "#536778"
BLUE = "#276C9B"
TEAL = "#168178"
ORANGE = "#CE6A2A"
PALE = "#E9EFF3"
GRID = "#D8E2E8"


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    return [float(row[key]) for row in rows]


def style_axes(ax, *, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def save_figure(fig, output_dir: Path, stem: str):
    # Fixed canvas and fonts keep the PNG and SVG layouts aligned. SVG text is
    # converted to paths so a reader's installed fonts do not alter the figure.
    for extension in ("png", "svg"):
        metadata = {"Title": stem.replace("_", " ")}
        if extension == "svg":
            metadata["Date"] = None
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=220, facecolor="white", metadata=metadata)
    plt.close(fig)


def plot_retests(rows, output_dir: Path):
    labels = ["0 ms repeat A", "0 ms repeat B", "Per-clip fixed\nrandom delay"]
    y = list(range(len(rows)))
    fig = plt.figure(figsize=(14.8, 5.5))
    grid = fig.add_gridspec(1, 3, left=0.165, right=0.965, top=0.715, bottom=0.255, wspace=0.32)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.text(0.045, 0.94, "Checkpoint 65k: matched native simulation retests", fontsize=19, weight="bold", color=INK)
    fig.text(0.045, 0.865, "Publisher-reported historical results | same 200-reference development set | one training seed (13)", fontsize=11, color=MUTED)

    ax = axes[0]
    completion = [100 * float(row["completion_fraction"]) for row in rows]
    ax.barh(y, [100] * len(rows), height=0.46, color=PALE)
    ax.barh(y, completion, height=0.46, color=BLUE)
    for pos, row, fraction in zip(y, rows, completion):
        ax.text(3, pos, f"{row['completed_clips']}/{row['evaluated_clips']}  ({fraction:.1f}%)", va="center", color="white", fontsize=10.6, weight="bold")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="Completed clips (%)")
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_title("a  Clip completion", loc="left", fontsize=12, weight="bold", color=INK, pad=20)

    ax = axes[1]
    mae = values(rows, "controlled_joint_mae_deg")
    rmse = values(rows, "controlled_joint_rmse_deg")
    ax.barh([i - 0.15 for i in y], mae, height=0.24, color=TEAL, label="MAE")
    ax.barh([i + 0.15 for i in y], rmse, height=0.24, color=BLUE, label="RMSE")
    for i, a, r in zip(y, mae, rmse):
        ax.text(a + 0.2, i - 0.15, f"{a:.3f}", va="center", fontsize=9.2, color=INK)
        ax.text(r + 0.2, i + 0.15, f"{r:.3f}", va="center", fontsize=9.2, color=INK)
    ax.set(yticks=y, yticklabels=[], xlim=(0, 13.8), xlabel="Joint angle error (degrees)")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_title("b  Controlled-joint error", loc="left", fontsize=12, weight="bold", color=INK, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04), ncol=2, frameon=False, fontsize=10, borderaxespad=0, handlelength=1.1)

    ax = axes[2]
    coverage = [100 * float(row["executed_frame_fraction"]) for row in rows]
    ax.barh(y, [100] * len(rows), height=0.46, color=PALE)
    ax.barh(y, coverage, height=0.46, color=TEAL)
    for pos, row, fraction in zip(y, rows, coverage):
        ax.text(3, pos, f"{int(row['executed_frames']):,}/{int(row['planned_frames']):,}  ({fraction:.3f}%)", va="center", color="white", fontsize=9.7, weight="bold")
    ax.set(yticks=y, yticklabels=[], xlim=(0, 100), xlabel="Executed / planned frames (%)")
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_title("c  Executed-frame coverage", loc="left", fontsize=12, weight="bold", color=INK, pad=20)

    for ax in axes:
        ax.set_ylim(2.6, -0.6)
        style_axes(ax)
    fig.text(0.045, 0.115, "Native MuJoCo-Warp | CUDA BF16 MLP, TF32 enabled | 50 Hz | random delay fixed per clip: 0 / 5 / 10 / 15 / 20 ms", fontsize=10, color=MUTED)
    fig.text(0.045, 0.065, "Error covers executed frames and 23 controlled joints; terminal frame included, no zero-filled tail. No multi-seed uncertainty estimated.", fontsize=10, color=MUTED)
    save_figure(fig, output_dir, "native_bf16_retests_65k")


def plot_candidates(rows, output_dir: Path):
    rows = sorted(rows, key=lambda row: int(row["checkpoint_collection_step"]))
    steps = [int(row["checkpoint_collection_step"]) / 1000 for row in rows]
    release = next(i for i, row in enumerate(rows) if row["checkpoint_collection_step"] == "65000")
    fig = plt.figure(figsize=(12.4, 8.8))
    grid = fig.add_gridspec(3, 1, left=0.09, right=0.965, top=0.79, bottom=0.21, hspace=0.22)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    fig.text(0.045, 0.945, "Retained snapshots under the same random-delay protocol", fontsize=17, weight="bold", color=INK)
    fig.text(0.045, 0.892, "Eight available candidates | one training seed (13) | same frozen 200 clips and per-clip latency assignments", fontsize=10.7, color=MUTED)
    fig.text(0.045, 0.85, "Native MuJoCo-Warp | CUDA BF16 MLP, TF32 enabled | development set reused for monitoring and selection", fontsize=10.4, color=MUTED)

    series = [
        ([100 * x for x in values(rows, "completion_fraction")], "a  Completed clips (%)", BLUE, (82, 102)),
        (values(rows, "controlled_joint_mae_deg"), "b  Joint MAE (degrees)", TEAL, (6.6, 10.8)),
        ([100 * x for x in values(rows, "executed_frame_fraction")], "c  Frame coverage (%)", BLUE, (90.5, 100.6)),
    ]
    for ax, (metric, label, color, ylim) in zip(axes, series):
        # Connections are a reading guide between measured snapshots, not a
        # smoothed learning curve or estimates for unmeasured checkpoints.
        ax.plot(steps, metric, color=color, linewidth=1.4, marker="o", markersize=5.5)
        ax.axvline(65, color=ORANGE, linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.8)
        ax.scatter([steps[release]], [metric[release]], s=95, color=ORANGE, edgecolors="white", linewidths=1.1, zorder=4)
        ax.set(xlim=(50.5, 101.5), ylim=ylim, ylabel=label)
        ax.yaxis.label.set_fontsize(10)
        style_axes(ax, grid_axis="y")
        ax.set_xticks(steps, [str(int(x)) for x in steps])
        ax.tick_params(axis="x", labelbottom=False)

    for x, row, offset in zip(steps, rows, [8, 8, -18, -18, 8, -18, 8, 8]):
        score = 100 * float(row["completion_fraction"])
        axes[0].annotate(f"{row['completed_clips']}/200", (x, score), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9, color=INK)
    axes[0].text(0.015, 0.06, "65k release candidate", transform=axes[0].transAxes, fontsize=10, color=ORANGE, weight="bold")
    for panel, key, precision in [(1, "controlled_joint_mae_deg", 3), (2, "executed_frame_fraction", 3)]:
        factor = 100 if panel == 2 else 1
        for i in (release, len(rows) - 1):
            x, value = steps[i], float(rows[i][key]) * factor
            axes[panel].annotate(f"{value:.{precision}f}", (x, value), xytext=(7, 9), textcoords="offset points", ha="left", fontsize=9.3, color=ORANGE if i == release else INK)
    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Checkpoint collection steps (thousands; not optimizer updates)", labelpad=10)
    fig.text(0.045, 0.104, "Points are recorded candidates; connections only aid reading. Axes show stated ranges. This is not a dense or multi-seed learning curve.", fontsize=9.7, color=MUTED)
    fig.text(0.045, 0.061, "Error uses executed frames only; coverage is shown alongside it. 65k is the distributed candidate; no global-best claim is made.", fontsize=9.7, color=MUTED)
    save_figure(fig, output_dir, "native_bf16_candidate_snapshots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "media/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "path",
        "svg.hashsalt": "supermimic-pm01-release-results-v1",
        "axes.unicode_minus": False,
    })
    protocol = json.loads((args.results_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol["independent_training_seeds"] != 1:
        raise ValueError("Figure wording is defined for the released single-seed records.")
    retests = read_csv(args.results_dir / "native_bf16_retests_65k.csv")
    candidates = read_csv(args.results_dir / "native_bf16_candidate_snapshots.csv")
    if len(retests) != 3 or len(candidates) != 8:
        raise ValueError("Expected the three release-checkpoint retests and eight matched candidates.")
    plot_retests(retests, args.output_dir)
    plot_candidates(candidates, args.output_dir)
    print(f"Wrote two figures as PNG and SVG to {args.output_dir}")


if __name__ == "__main__":
    main()
