#!/usr/bin/env python3
"""Plot CGP run logs (.xls text files) written by cgp.cpp.

Only the first 4 tab-separated columns are used:
1) Generation
2) Bestfitness
3) Pop. fitness
4) #Blocks
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List


@dataclass
class LogData:
    path: str
    generation: List[int]
    best_fitness: List[int]
    pop_fitness: List[int]
    blocks: List[int]
    blocks_gen_filtered: List[int] = None
    blocks_val_filtered: List[int] = None
    
    def __post_init__(self):
        """Filter blocks to only points where value changes."""
        if self.blocks_gen_filtered is None:
            self.blocks_gen_filtered = []
            self.blocks_val_filtered = []
            if self.blocks:
                self.blocks_gen_filtered.append(self.generation[0])
                self.blocks_val_filtered.append(self.blocks[0])
                for i in range(1, len(self.blocks)):
                    if self.blocks[i] != self.blocks[i-1]:
                        self.blocks_gen_filtered.append(self.generation[i])
                        self.blocks_val_filtered.append(self.blocks[i])
                # Always add final generation to reach the end
                if self.blocks_gen_filtered[-1] != self.generation[-1]:
                    self.blocks_gen_filtered.append(self.generation[-1])
                    self.blocks_val_filtered.append(self.blocks[-1])


def parse_log(path: str) -> LogData:
    generation: List[int] = []
    best_fitness: List[int] = []
    pop_fitness: List[int] = []
    blocks: List[int] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Generation"):
                continue

            # cgp.cpp writes tab-separated rows and may include an empty field after #Blocks.
            cols = line.split("\t")
            if len(cols) < 4:
                continue

            try:
                generation.append(int(cols[0]))
                best_fitness.append(int(cols[1]))
                pop_fitness.append(int(cols[2]))
                blocks.append(int(cols[3]))
            except ValueError:
                # Skip malformed row fragments.
                continue

    return LogData(
        path=path,
        generation=generation,
        best_fitness=best_fitness,
        pop_fitness=pop_fitness,
        blocks=blocks,
    )


def plot_placeholder(outdir: str, filename: str, title: str, message: str) -> str:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc
    return plt


def plot_single_log(data: LogData, outdir: str) -> str:
    plt = _ensure_matplotlib()

    if not data.generation:
        return plot_placeholder(
            outdir,
            "single_log_overview.png",
            f"Single Log Overview: {os.path.basename(data.path)}",
            "No numeric data rows were parsed from this log.",
        )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9.5, 8),
        sharex=True,
        constrained_layout=True,
    )

    ax0, ax1 = axes
    ax0.plot(data.generation, data.best_fitness, linewidth=2.0, label="Bestfitness")
    ax0.plot(data.generation, data.pop_fitness, linewidth=1.6, label="Pop. fitness")
    ax0.set_ylabel("Fitness")
    ax0.set_title(f"Single Log Overview: {os.path.basename(data.path)}")
    ax0.margins(y=0.06)
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(data.blocks_gen_filtered, data.blocks_val_filtered, color="tab:green", linewidth=1.8, label="#Blocks")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Blocks")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "single_log_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_multi_log(logs: List[LogData], outdir: str) -> str:
    plt = _ensure_matplotlib()

    if not logs:
        return plot_placeholder(
            outdir,
            "multi_log_comparison.png",
            "Multi Log Comparison",
            "No parsable logs were found.",
        )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9.5, 9),
        sharex=False,
        constrained_layout=True,
    )

    ax0, ax1 = axes

    for data in logs:
        label = os.path.basename(data.path)
        ax0.plot(data.generation, data.best_fitness, linewidth=1.4, alpha=0.9, label=label)
        ax1.plot(data.blocks_gen_filtered, data.blocks_val_filtered, linewidth=1.2, alpha=0.85, label=label)

    ax0.set_title("Bestfitness Across Multiple Logs")
    ax0.set_xlabel("Generation")
    ax0.set_ylabel("Bestfitness")
    ax0.margins(y=0.06)
    ax0.grid(True, alpha=0.3)

    ax1.set_title("#Blocks Across Multiple Logs")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("#Blocks")
    ax1.grid(True, alpha=0.3)

    # Keep legend readable when many logs are included.
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "multi_log_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def discover_logs(logs_dir: str, pattern: str, max_logs: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(logs_dir, pattern)))
    if max_logs > 0:
        paths = paths[:max_logs]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize cgp.cpp .xls logs using only the first 4 columns."
    )
    parser.add_argument(
        "--logs-dir",
        default=".",
        help="Directory containing log_*.xls files (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="log_*.xls",
        help="Glob pattern for log files (default: log_*.xls).",
    )
    parser.add_argument(
        "--single",
        default="log_0.xls",
        help="Single file to plot in detail (default: log_0.xls).",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=10,
        help="Maximum number of logs for multi-log chart (default: 10, 0 means all).",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Output directory for PNG images (default: plots).",
    )

    args = parser.parse_args()

    single_path = args.single
    if not os.path.isabs(single_path):
        single_path = os.path.join(args.logs_dir, single_path)

    if not os.path.exists(single_path):
        raise SystemExit(f"Single log file not found: {single_path}")

    single_data = parse_log(single_path)

    log_paths = discover_logs(args.logs_dir, args.pattern, args.max_logs)
    if not log_paths:
        raise SystemExit(
            f"No log files found in {args.logs_dir} using pattern {args.pattern}"
        )

    logs: List[LogData] = []
    for p in log_paths:
        try:
            logs.append(parse_log(p))
        except ValueError:
            # Ignore files that do not parse as cgp logs.
            continue

    single_out = plot_single_log(single_data, args.outdir)
    multi_out = plot_multi_log(logs, args.outdir)

    print("Saved plots:")
    print(single_out)
    print(multi_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
